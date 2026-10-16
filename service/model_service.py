"""Model lifecycle and atomic publication of complete serving runtimes."""

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

import torch

from algorithm.utils.file_util import resolve_feature_file
from backends.torch_backend import select_device
from config import Config
from error_code import ErrorCode, ReException
from inference.runtime import ModelRuntime
from models.registry import MODEL_REGISTRY
from proto import Model


class ModelService:
    def __init__(self, feature_service, registry=None):
        self.feature_service = feature_service
        self.registry = MODEL_REGISTRY if registry is None else registry
        self.runtimes = {}
        self.model_lock = threading.RLock()
        self.load_lock = threading.RLock()
        self.refresh_lock = threading.Lock()
        self.refresh_workers = {}
        self.refresh_attempts = {}

    def current(self, target):
        with self.model_lock:
            return self.runtimes.get(target)

    def select_device(self, model_type):
        return select_device(model_type)

    def startup(self):
        for directory in (
            Path(Config.MODEL.ROOT) / "training",
            Path(Config.MODEL.ROOT) / "releases",
        ):
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o777 if directory.name == "training" else 0o755)
        failures = []
        for target_type in ("item", "user"):
            try:
                self.restore(target_type)
            except Exception as error:
                logging.exception("automatic %s model load failed", target_type)
                failures.append(error)
        if failures and Config.MODEL.REQUIRED:
            raise failures[0]

    def state_path(self, target_type):
        return Path(Config.MODEL.STATE_DIR) / (target_type + ".json")

    def _bootstrap_model_info(self, target_type):
        model_path = (
            Config.MODEL.USER_PATH if target_type == "user" else Config.MODEL.PATH
        )
        if not model_path:
            return None
        feature = (
            Config.MODEL.USER_FEATURE_PATH
            if target_type == "user"
            else Config.MODEL.FEATURE_PATH
        )
        return Model(
            type=Config.MODEL.TYPE,
            model=model_path,
            feature=feature,
            dim=Config.MODEL.DIM,
        )

    def _persist_model(self, info, loaded_info):
        target_type = loaded_info["target_type"]
        path = self.state_path(target_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        load = {
            "type": loaded_info["type"],
            "model": str(Path(info.model).resolve()),
            "feature": (
                str(Path(loaded_info["feature"]).resolve())
                if loaded_info["feature"]
                else None
            ),
            "dim": loaded_info["dim"],
            "factor_dim": loaded_info.get("factor_dim"),
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                json.dump(
                    {
                        "schema_version": 1,
                        "target_type": target_type,
                        "load": load,
                    },
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def restore(self, target_type):
        # Serialize recovery with publication and recheck after waiting for another
        # load.
        with self.load_lock:
            with self.model_lock:
                current = self.runtimes.get(target_type)
                if current is not None:
                    return
            path = self.state_path(target_type)
            if path.exists():
                try:
                    saved = json.loads(path.read_text())
                    if (
                        saved["schema_version"] != 1
                        or saved["target_type"] != target_type
                    ):
                        raise ValueError("invalid persisted model state")
                    info = Model(**saved["load"])
                    return self.load(
                        info, persist=False, expected_target=target_type
                    )
                except Exception:
                    logging.exception(
                        "persisted %s model is unusable; falling back to bootstrap",
                        target_type,
                    )
                    info = self._bootstrap_model_info(target_type)
                    if info is None:
                        raise
                    # Replace the stale state only after the bootstrap model has
                    # loaded successfully. A transient failure keeps the published
                    # state available for a later retry.
                    return self.load(
                        info, persist=True, expected_target=target_type
                    )
            info = self._bootstrap_model_info(target_type)
            if info is not None:
                return self.load(info, persist=False, expected_target=target_type)

    def load(self, info, persist=True, expected_target=None):
        model_type = info.type.strip().lower()
        if model_type not in self.registry:
            raise ReException(ErrorCode.INVALID_MODEL)
        with self.load_lock:
            feature_file = info.feature or resolve_feature_file(info.model)
            snapshot = self.feature_service.prepare_all_features(
                str(feature_file) if feature_file else None
            )
            target = snapshot.get("target_type", "item")
            if target not in ("item", "user") or (expected_target and target != expected_target):
                raise ValueError("feature space target does not match requested model")
            declared = snapshot.get("model_type")
            if declared and declared != model_type:
                raise ValueError("feature space belongs to %s, not %s" % (declared, model_type))
            dim = snapshot["dim"] if feature_file else info.dim
            device = self.select_device(model_type)
            definition = self.registry[model_type]
            loaded = definition.load(info, dim, device)
            executor = definition.prepare(loaded, snapshot)
            snapshot["loaded_at"] = time.monotonic()
            metadata = {
                **loaded.metadata,
                "feature_selection": snapshot.get("feature_selection"),
                "type": model_type, "path": info.model,
                "feature": str(feature_file) if feature_file else None,
                "dim": dim, "device": str(device),
                "feature_set": snapshot.get("feature_set"),
                "catalog_version": snapshot.get("catalog_version"),
                "catalog_sha256": snapshot.get("catalog_sha256"),
                "target_type": target,
            }
            runtime = ModelRuntime(loaded, definition, snapshot, executor, metadata)
            with self.model_lock:
                if persist:
                    self._persist_model(info, metadata)
                self.feature_service.activate(snapshot)
                self.runtimes[target] = runtime
            return metadata

    def refresh_features(self, target, force=False):
        with self.load_lock:
            active = self.current(target)
            snapshot = active.snapshot if active is not None else None
            if snapshot is None or not snapshot.get("feature_file"):
                return
            seconds = Config.MODEL.FEATURE_REFRESH_SECONDS
            if not force and (seconds <= 0 or time.monotonic() - snapshot.get("loaded_at", 0) < seconds):
                return
            prepared = self.feature_service.prepare_all_features(snapshot["feature_file"])
            if prepared["dim"] != active.dim:
                raise ValueError("refreshed features do not match model dimension")
            executor = active.definition.prepare(active.loaded, prepared)
            prepared["loaded_at"] = time.monotonic()
            replacement = ModelRuntime(active.loaded, active.definition, prepared, executor, active.info)
            with self.model_lock:
                if self.runtimes.get(target) is not active:
                    return
                self.feature_service.activate(prepared)
                self.runtimes[target] = replacement

    def schedule_feature_refresh(self, target):
        active = self.current(target)
        snapshot = active.snapshot if active is not None else None
        seconds = Config.MODEL.FEATURE_REFRESH_SECONDS
        now = time.monotonic()
        if (not snapshot or not snapshot.get("feature_file") or seconds <= 0
                or now - snapshot.get("loaded_at", 0) < seconds):
            return
        with self.refresh_lock:
            if target in self.refresh_workers:
                return
            previous, attempted_at = self.refresh_attempts.get(target, (None, 0))
            if previous is active and now - attempted_at < seconds:
                return
            self.refresh_attempts[target] = (active, now)

            def refresh():
                try:
                    self.refresh_features(target)
                except Exception:
                    logging.exception("feature refresh failed; retaining last usable snapshot")
                finally:
                    with self.refresh_lock:
                        self.refresh_workers.pop(target, None)

            worker = threading.Thread(target=refresh, name=f"feature-refresh-{target}", daemon=True)
            self.refresh_workers[target] = worker
            try:
                worker.start()
            except Exception:
                self.refresh_workers.pop(target, None)
                raise

    def clean(self):
        with self.model_lock:
            self.runtimes.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
