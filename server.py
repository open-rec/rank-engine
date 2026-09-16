import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
)

from algorithm.utils.file_util import resolve_feature_file
from config import Config
from error_code import ErrorCode, ReException
from model import model_func_map
from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.point_in_time import materialize_point_in_time_samples
from algorithm.feature.user_feature import UserFeature
from algorithm.feature.feature_space import FeatureSpace
from algorithm.feature.feature_catalog import feature_catalog, select_features
from algorithm.rank.lr import LRRecModel
from algorithm.rank.fm import FMRecModel
from proto import Model, ReResponse, TrainModel, UserItems
from service.feature_service import FeatureService
from util.redis_util import get_redis_client

app = FastAPI(title="OpenRec Rank Engine", version="1.0")
model = None
model_info = None
user_model = None
user_model_info = None
model_lock = threading.RLock()
load_lock = threading.RLock()
feature_service = FeatureService()
request_count = Counter(
    "openrec_rank_requests",
    "Rank Engine requests",
    ["method", "path", "status"],
)
request_latency = Histogram(
    "openrec_rank_request_latency_seconds",
    "Rank Engine request latency",
    ["method", "path"],
)
model_loaded = Gauge(
    "openrec_rank_model_loaded", "Whether a ranking model is loaded"
)
models_loaded = Gauge(
    "openrec_rank_models_loaded", "Loaded models by target", ["target_type"]
)
business_errors = Counter(
    "openrec_rank_business_errors",
    "Failed business requests",
    ["method", "path", "code"],
)


def request_path(request):
    route = request.scope.get("route")
    return route.path if route else "unmatched"


def _align_materialized_rows(events, sample_users, sample_items):
    """Align Spark materialized rows by immutable sample identity."""
    frames = (
        ("events", events),
        ("sample_users", sample_users),
        ("sample_items", sample_items),
    )
    for name, frame in frames:
        if "_sample_id" not in frame.columns:
            raise ValueError("%s is missing _sample_id" % name)
        if (
            frame["_sample_id"].isna().any()
            or frame["_sample_id"].duplicated().any()
        ):
            raise ValueError("%s contains null or duplicate _sample_id" % name)
    expected = set(events["_sample_id"])
    if (
        set(sample_users["_sample_id"]) != expected
        or set(sample_items["_sample_id"]) != expected
    ):
        raise ValueError(
            "materialized feature rows do not match event sample identities"
        )
    order = events["_sample_id"].tolist()
    return (
        sample_users.set_index("_sample_id").loc[order].reset_index(),
        sample_items.set_index("_sample_id").loc[order].reset_index(),
    )


def _latest_feature_rows(events, rows):
    """Select latest per-entity PIT rows regardless of part-file order."""
    if "_sample_id" in events.columns and "_sample_id" in rows.columns:
        timeline = events[["_sample_id", "time"]].rename(
            columns={"time": "_label_time"}
        )
        ordered = rows.merge(timeline, on="_sample_id", validate="one_to_one")
    else:
        if len(events) != len(rows):
            raise ValueError("feature rows are not aligned with label events")
        ordered = rows.copy()
        ordered["_label_time"] = events["time"].tolist()
    ordered["_sample_order"] = range(len(ordered))
    ordered = ordered.sort_values(
        ["_label_time", "_sample_order"], kind="mergesort"
    )
    return (
        ordered.drop_duplicates("id", keep="last")
        .drop(
            columns=["_sample_id", "_label_time", "_sample_order"],
            errors="ignore",
        )
        .reset_index(drop=True)
    )


@app.middleware("http")
async def observe_request(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    started = time.monotonic()
    status = "500"
    try:
        result = await call_next(request)
        status = str(result.status_code)
        return result
    finally:
        path = request_path(request)
        request_count.labels(request.method, path, status).inc()
        request_latency.labels(request.method, path).observe(
            time.monotonic() - started
        )


@app.get("/metrics")
def metrics():
    with model_lock:
        model_loaded.set(1 if model is not None else 0)
        models_loaded.labels("item").set(int(model is not None))
        models_loaded.labels("user").set(int(user_model is not None))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/features")
def features():
    return response(feature_catalog())


@app.post("/features/validate")
def validate_features(info: dict):
    try:
        return response(
            select_features(
                info.get("model_type", "lr"),
                info.get("target_type", "item"),
                info.get("feature_selection"),
            )
        )
    except ValueError as error:
        return JSONResponse(status_code=422, content={"detail": str(error)})


def model_device():
    configured = Config.MODEL.DEVICE
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "MODEL_DEVICE requests CUDA but no CUDA device is available"
        )
    return torch.device(configured)


def response(data=None, message=""):
    return ReResponse(
        code=0, status="success", data=data, message=message
    ).to_dict()


@app.exception_handler(ReException)
async def re_exception_handler(request: Request, exception: ReException):
    error = exception.error_code
    business_errors.labels(
        request.method, request_path(request), str(error.code)
    ).inc()
    return JSONResponse(
        status_code=200,
        content=ReResponse(
            code=error.code, status="fail", data=None, message=error.message
        ).to_dict(),
    )


@app.exception_handler(Exception)
async def unknown_exception_handler(request: Request, exception: Exception):
    logging.exception("unhandled request error")
    business_errors.labels(
        request.method,
        request_path(request),
        str(ErrorCode.UNKNOWN_ERROR.code),
    ).inc()
    return JSONResponse(
        status_code=503 if request.url.path == "/health" else 200,
        content=ReResponse(
            code=ErrorCode.UNKNOWN_ERROR.code,
            status="fail",
            data=None,
            message=ErrorCode.UNKNOWN_ERROR.message,
        ).to_dict(),
    )


@app.on_event("startup")
def startup():
    for directory in (
        Path(Config.MODEL.ROOT) / "training",
        Path(Config.MODEL.ROOT) / "releases",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o777 if directory.name == "training" else 0o755)
    failures = []
    for target_type in ("item", "user"):
        try:
            _restore_model(target_type)
        except Exception as error:
            logging.exception("automatic %s model load failed", target_type)
            failures.append(error)
    if failures and Config.MODEL.REQUIRED:
        raise failures[0]


def _state_path(target_type):
    return Path(Config.MODEL.STATE_DIR) / (target_type + ".json")


def _restore_model(target_type):
    # Serialize recovery with publication and recheck after waiting for another
    # load.
    with load_lock:
        with model_lock:
            current = user_model if target_type == "user" else model
            if current is not None:
                return
        path = _state_path(target_type)
        if path.exists():
            saved = json.loads(path.read_text())
            if (
                saved["schema_version"] != 1
                or saved["target_type"] != target_type
            ):
                raise ValueError("invalid persisted model state")
            info = Model(**saved["load"])
        else:
            model_path = (
                Config.MODEL.USER_PATH
                if target_type == "user"
                else Config.MODEL.PATH
            )
            if not model_path:
                return
            feature = (
                Config.MODEL.USER_FEATURE_PATH
                if target_type == "user"
                else Config.MODEL.FEATURE_PATH
            )
            info = Model(
                type=Config.MODEL.TYPE,
                model=model_path,
                feature=feature,
                dim=Config.MODEL.DIM,
            )
        _load_model(info, persist=False, expected_target=target_type)


def _persist_model(info, loaded_info):
    target_type = loaded_info["target_type"]
    path = _state_path(target_type)
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


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/health")
def health():
    try:
        redis_ok = bool(get_redis_client().ping())
    except Exception:
        redis_ok = False
    with model_lock:
        loaded = model is not None
        current = model_info
        user_loaded = user_model is not None
        user_current = user_model_info
        features = feature_service.stats()
    required = {"item"}
    if Config.MODEL.USER_PATH or _state_path("user").exists() or user_loaded:
        required.add("user")
    target_ready = {}
    for target, present, info in (
        ("item", loaded, current),
        ("user", user_loaded, user_current),
    ):
        stats = features[target]
        target_ready[target] = bool(
            present
            and info
            and stats["users"]
            and stats["candidates"]
            and stats["dim"] == info["dim"]
        )
    result = response(
        {
            "redis": redis_ok,
            "model_loaded": loaded,
            "model": current,
            "features": features,
            "user_model_loaded": user_loaded,
            "user_model": user_current,
            "targets_ready": target_ready,
            "required_targets": sorted(required),
            "ready": redis_ok
            and all(target_ready[target] for target in required),
        }
    )
    return JSONResponse(
        status_code=200 if result["data"]["ready"] else 503, content=result
    )


def _load_model(info, persist=True, expected_target=None):
    global model, model_info, user_model, user_model_info
    model_type = info.type.strip().lower()
    if model_type not in model_func_map:
        raise ReException(ErrorCode.INVALID_MODEL)
    with load_lock:
        feature_file = info.feature or resolve_feature_file(info.model)
        snapshot = feature_service.prepare_all_features(
            str(feature_file) if feature_file else None
        )
        target_type = snapshot.get("target_type", "item")
        if target_type not in ("item", "user") or (
            expected_target and target_type != expected_target
        ):
            raise ValueError(
                "feature space target does not match requested model"
            )
        declared_model_type = snapshot.get("model_type")
        if declared_model_type and declared_model_type != model_type:
            raise ValueError(
                "feature space belongs to %s, not %s"
                % (declared_model_type, model_type)
            )
        effective_dim = snapshot["dim"] if feature_file else info.dim
        device = model_device()
        state = torch.load(info.model, map_location=device)
        kwargs = {}
        if model_type == "fm":
            factors = state.get("factors")
            if (
                factors is None
                or factors.ndim != 2
                or factors.shape[0] != effective_dim
            ):
                raise ValueError(
                    "FM checkpoint factors do not match the feature dimension"
                )
            kwargs["factor_dim"] = info.factor_dim or factors.shape[1]
        loaded_model = model_func_map[model_type](effective_dim, **kwargs)
        loaded_model.load_state_dict(state)
        loaded_model.to(device)
        loaded_model.eval()
        snapshot["loaded_at"] = time.monotonic()
        loaded_model.feature_snapshot = snapshot
        loaded_info = {
            "feature_selection": snapshot.get("feature_selection"),
            "type": model_type,
            "path": info.model,
            "feature": str(feature_file) if feature_file else None,
            "dim": effective_dim,
            "device": str(device),
            "feature_set": snapshot.get("feature_set"),
            "catalog_version": snapshot.get("catalog_version"),
            "catalog_sha256": snapshot.get("catalog_sha256"),
            "target_type": snapshot.get("target_type", "item"),
            **(
                {"factor_dim": loaded_model.factor_dim}
                if model_type == "fm"
                else {}
            ),
        }
        with model_lock:
            if persist:
                _persist_model(info, loaded_info)
            feature_service.activate(snapshot)
            if snapshot.get("target_type", "item") == "user":
                user_model, user_model_info = loaded_model, loaded_info
            else:
                model, model_info = loaded_model, loaded_info
    return loaded_info


@app.post("/model/load")
def load_model(info: Model):
    try:
        return response(_load_model(info))
    except ReException:
        raise
    except FileNotFoundError:
        raise ReException(ErrorCode.MODEL_NOT_FOUND)
    except Exception:
        logging.exception("model load failed")
        raise ReException(ErrorCode.LOAD_MODEL_FAILED)


@app.post("/model/train")
def train_model(info: TrainModel):
    """Train an immutable artifact from a Spark-prepared dataset."""
    try:
        selection = select_features(
            info.model_type, info.target_type, info.feature_selection
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    dataset = Path(info.dataset_dir).resolve()
    training_root = (Path(Config.MODEL.ROOT) / "training").resolve()
    artifact_root = (Path(Config.MODEL.ROOT) / "releases").resolve()
    if training_root not in dataset.parents or not re.match(
        r"^[A-Za-z0-9_-]+$", info.scene
    ):
        raise ReException(ErrorCode.INVALID_MODEL)
    target = artifact_root / info.target_type / info.scene / info.version
    if target.exists():
        raise ReException(ErrorCode.LOAD_MODEL_FAILED)
    scene_root = artifact_root / info.target_type / info.scene
    scene_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".%s-" % info.version, dir=str(scene_root))
    )
    try:

        def read_records(path):
            files = (
                sorted(path.glob("part-*.json")) if path.is_dir() else [path]
            )
            frames = [
                pd.read_json(value, lines=True)
                for value in files
                if value.stat().st_size
            ]
            return (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame()
            )

        events = read_records(dataset / "events.jsonl")
        model_type = info.model_type.strip().lower()
        model_filename = "%s.pth" % model_type
        feature_filename = "%s.features.json" % model_type
        model_class = {"lr": LRRecModel, "fm": FMRecModel}[model_type]
        model_kwargs = (
            {"factor_dim": info.factor_dim} if model_type == "fm" else {}
        )
        if (dataset / "sample_users.jsonl").exists():
            sample_users = read_records(dataset / "sample_users.jsonl")
            sample_items = read_records(dataset / "sample_items.jsonl")
            sample_users, sample_items = _align_materialized_rows(
                events, sample_users, sample_items
            )
        else:
            feature_events = read_records(dataset / "feature_events.jsonl")
            items = read_records(dataset / "items.jsonl")
            users = read_records(dataset / "users.jsonl")
            events, sample_users, sample_items = (
                materialize_point_in_time_samples(
                    events, feature_events, users, items, info.target_type
                )
            )
        if events.empty:
            raise ValueError(
                "rank training has no entities active at their label times"
            )
        space = FeatureSpace.for_model(
            info.model_type, info.target_type, selection
        )
        for frame, columns in (
            (sample_users, space.user_columns),
            (sample_items, space.item_columns),
        ):
            for column in columns:
                if (
                    column.name not in frame
                    or not frame[column.name].notna().any()
                ):
                    raise ValueError(
                        "offline feature has no materialized values: %s"
                        % column.feature_id
                    )
        latest_users = _latest_feature_rows(events, sample_users)
        latest_items = _latest_feature_rows(events, sample_items)
        user_features = UserFeature(latest_users)
        item_features = (
            ItemFeature(latest_items)
            if info.target_type == "item"
            else UserFeature(latest_items)
        )
        rank_model = model_class(
            user_features,
            item_features,
            events,
            feature_space=space,
            scene=info.scene,
            model_file=staging / model_filename,
            feature_file=staging / feature_filename,
            target_type=info.target_type,
            sample_users=sample_users,
            sample_items=sample_items,
            validation_ratio=info.validation_ratio,
            **model_kwargs,
        )
        if not len(rank_model.dataset):
            raise ValueError(
                "rank training produced no labelled samples "
                "after entity filtering"
            )
        if rank_model.dataset.positive_rate in (0.0, 1.0):
            raise ValueError(
                "rank training requires both click and expose labels"
            )
        training, validation = rank_model._split(
            val_ratio=info.validation_ratio, seed=42
        )
        rank_model.train(
            epoch_num=info.epochs,
            batch_size=info.batch_size,
            val_ratio=info.validation_ratio,
        )
        auc = rank_model.evaluate(validation, batch_size=info.batch_size)
        if auc is None:
            raise ValueError("AUC is undefined for validation data")
        if auc is not None and auc < info.min_auc:
            raise ValueError("AUC %.6f is below %.6f" % (auc, info.min_auc))
        rank_model.save()
        # Keep the unencoded entity snapshots next to the checkpoint. They are
        # the portable
        # bootstrap representation for Redis; *.features.json remains the
        # model-specific encoding
        # contract and must not be confused with actual entity feature values.
        candidate_features = (
            item_features.items
            if info.target_type == "item"
            else item_features.users
        )
        for frame, filename in (
            (user_features.users, "user_feature.csv"),
            (candidate_features, "item_feature.csv"),
        ):
            exported = frame.copy()
            exported.insert(
                1,
                "as_of_time",
                info.feature_until_time or info.feature_cutoff_time,
            )
            exported.to_csv(staging / filename, index=False)
        feature_bytes = (staging / feature_filename).read_bytes()
        feature_sha256 = hashlib.sha256(feature_bytes).hexdigest()
        feature_space = rank_model.dataset.feature_space
        manifest = {
            "model_sha256": hashlib.sha256(
                (staging / model_filename).read_bytes()
            ).hexdigest(),
            "feature_selection": feature_space.selection,
            "feature_definitions": feature_space.feature_definitions,
            "training_config": {
                "epochs": info.epochs,
                "batch_size": info.batch_size,
                "validation_ratio": info.validation_ratio,
                "min_auc": info.min_auc,
                "model_type": info.model_type,
                "factor_dim": info.factor_dim,
                "scene": info.scene,
                "target_type": info.target_type,
            },
            "version": info.version,
            "scene": info.scene,
            "model_type": model_type,
            "target_type": info.target_type,
            "business_date": info.business_date,
            "revision": info.revision,
            "label_observation_cutoff": info.label_observation_cutoff,
            "feature_cutoff_time": info.feature_cutoff_time,
            "feature_until_time": info.feature_until_time
            or info.feature_cutoff_time,
            "feature_join": "per_sample_point_in_time",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "evaluated",
            "model": model_filename,
            "feature": feature_filename,
            "user_feature_snapshot": "user_feature.csv",
            "item_feature_snapshot": "item_feature.csv",
            "feature_set": feature_space.feature_set,
            "catalog_version": feature_space.catalog_version,
            "catalog_sha256": feature_space.catalog_sha256,
            "feature_sha256": feature_sha256,
            "input_dim": rank_model.model.dim,
            "metrics": {
                "auc": auc,
                "positive_rate": rank_model.dataset.positive_rate,
                "samples": len(rank_model.dataset),
                "input_labels": info.input_label_count,
                "constructed_labels": info.constructed_label_count,
                "spark_materialized_labels": info.materialized_label_count,
                "dropped_labels": (
                    (
                        info.constructed_label_count
                        if info.constructed_label_count is not None
                        else info.input_label_count
                    )
                    - len(rank_model.dataset)
                ),
                "history_rows": info.history_row_count,
                "materialization_seconds": info.materialization_seconds,
                "training_samples": len(training),
                "validation_samples": len(validation),
                "label_time_min": int(events["time"].min()),
                "label_time_max": int(events["time"].max()),
                "feature_dim": rank_model.model.dim,
                **(
                    {"factor_dim": rank_model.model.factor_dim}
                    if model_type == "fm"
                    else {}
                ),
            },
            "gate": {"min_auc": info.min_auc, "passed": True},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
        os.replace(staging, target)
        shutil.rmtree(dataset, ignore_errors=True)
        return response(manifest)
    except Exception:
        logging.exception("model training failed")
        shutil.rmtree(staging, ignore_errors=True)
        raise ReException(ErrorCode.LOAD_MODEL_FAILED)


@app.post("/model/refresh-features")
def refresh_features():
    try:
        for target in ("item", "user"):
            _refresh_features(target, force=True)
        return response(feature_service.stats())
    except Exception:
        logging.exception("feature refresh failed")
        raise ReException(ErrorCode.LOAD_MODEL_FAILED)


def _refresh_features(target, force=False):
    # Use the publication lock while preparing; inference retains its old
    # snapshot.
    with load_lock:
        with model_lock:
            active = user_model if target == "user" else model
            snapshot = getattr(active, "feature_snapshot", None)
        if snapshot is None or not snapshot.get("feature_file"):
            return
        age = time.monotonic() - snapshot.get("loaded_at", 0)
        seconds = Config.MODEL.FEATURE_REFRESH_SECONDS
        if not force and (seconds <= 0 or age < seconds):
            return
        prepared = feature_service.prepare_all_features(
            snapshot["feature_file"]
        )
        if prepared["dim"] != active.dim:
            raise ValueError("refreshed features do not match model dimension")
        prepared["loaded_at"] = time.monotonic()
        with model_lock:
            feature_service.activate(prepared)
            active.feature_snapshot = prepared


@app.post("/clean")
def clean():
    global model, model_info, user_model, user_model_info
    with model_lock:
        model = None
        model_info = None
        user_model = None
        user_model_info = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return response(message="model unloaded")


@app.post("/model/score")
def score(user_items: UserItems):
    with model_lock:
        current_model = (
            user_model if user_items.target_type == "user" else model
        )
        current_info = (
            user_model_info if user_items.target_type == "user" else model_info
        )
    if current_model is None:
        try:
            _restore_model(user_items.target_type)
            with model_lock:
                current_model = (
                    user_model if user_items.target_type == "user" else model
                )
                current_info = (
                    user_model_info
                    if user_items.target_type == "user"
                    else model_info
                )
        except Exception:
            logging.exception("lazy model load failed")
    if current_model is None:
        raise ReException(ErrorCode.MODEL_NOT_LOAD_YET)
    candidate_ids = user_items.candidate_ids or user_items.item_ids
    target_type = (current_info or {}).get("target_type", "item")
    if user_items.target_type != target_type:
        raise ReException(ErrorCode.INVALID_MODEL)
    if not candidate_ids:
        return response({})
    try:
        try:
            _refresh_features(target_type)
        except Exception:
            logging.exception(
                "feature refresh failed; retaining last usable snapshot"
            )
        snapshot = getattr(current_model, "feature_snapshot", None)
        user_features = (
            snapshot["users"].get(user_items.user_id)
            if snapshot is not None
            else feature_service.get_user_feature_by_id(
                user_items.user_id, namespace=target_type
            )
        )
        batch_features = []
        item_score_map = {}
        hit_items = []
        for item_id in candidate_ids:
            item_features = (
                snapshot["items"].get(item_id)
                if snapshot is not None
                else (
                    feature_service.get_item_feature_by_id(item_id)
                    if target_type == "item"
                    else feature_service.get_candidate_user_feature_by_id(
                        item_id
                    )
                )
            )
            if item_features is None:
                item_score_map[item_id] = 0.0
                continue
            effective_user = user_features
            if effective_user is None:
                effective_user = np.zeros(
                    current_model.dim - item_features.size
                )
            features = np.concatenate((effective_user, item_features))
            if features.size != current_model.dim:
                raise ValueError(
                    "feature dimension does not match loaded model"
                )
            batch_features.append(
                torch.tensor(
                    features,
                    dtype=torch.float32,
                    device=next(current_model.parameters()).device,
                )
            )
            hit_items.append(item_id)
        if batch_features:
            with torch.no_grad():
                scores = (
                    current_model(torch.stack(batch_features))
                    .reshape(-1)
                    .tolist()
                )
            item_score_map.update(zip(hit_items, scores))
        return response(item_score_map)
    except Exception:
        logging.exception("inference failed")
        raise ReException(ErrorCode.INFERENCE_FAILED)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=Config.SERVER.HOST, port=Config.SERVER.PORT)
