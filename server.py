import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
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
from proto import Model, ReResponse, UserItems
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


def _bootstrap_model_info(target_type):
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
            try:
                saved = json.loads(path.read_text())
                if (
                    saved["schema_version"] != 1
                    or saved["target_type"] != target_type
                ):
                    raise ValueError("invalid persisted model state")
                info = Model(**saved["load"])
                return _load_model(
                    info, persist=False, expected_target=target_type
                )
            except Exception:
                logging.exception(
                    "persisted %s model is unusable; falling back to bootstrap",
                    target_type,
                )
                info = _bootstrap_model_info(target_type)
                if info is None:
                    raise
                # Replace the stale state only after the bootstrap model has
                # loaded successfully. A transient failure keeps the published
                # state available for a later retry.
                return _load_model(
                    info, persist=True, expected_target=target_type
                )
        info = _bootstrap_model_info(target_type)
        if info is not None:
            return _load_model(info, persist=False, expected_target=target_type)


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
        kwargs = {}
        if model_type == "lightgbm":
            loaded_model = model_func_map[model_type].load(info.model)
            if loaded_model.booster.num_feature() != effective_dim:
                raise ValueError(
                    "LightGBM model features do not match the feature dimension"
                )
            loaded_model.dim = effective_dim
        else:
            state = torch.load(info.model, map_location=device)
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
        if model_type != "lightgbm":
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
            "device": "cpu" if model_type == "lightgbm" else str(device),
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
        space = snapshot.get("space") if snapshot is not None else None
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
        from algorithm.feature.context_feature import materialize_request_context
        dynamic_contexts = materialize_request_context(
            user_items.context, candidate_ids,
            user_items.context.get("request_time"), user_items.candidate_contexts)
        for position, item_id in enumerate(candidate_ids):
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
                width = space.user_width if space is not None else (
                    current_model.dim - item_features.size)
                effective_user = np.zeros(width)
            parts = [effective_user, item_features]
            if space is not None:
                session = snapshot.get("sessions", {}).get(user_items.session_id)
                if session is None:
                    session = np.zeros(space.session_width)
                parts.append(session)
                row = dynamic_contexts.iloc[[position]]
                parts.append(space.transform_contexts(row)[0])
                # Candidate-specific interaction values use the same envelope.
                parts.append(space.transform_interactions(row)[0])
            features = np.concatenate(parts)
            if features.size != current_model.dim:
                raise ValueError(
                    "feature dimension does not match loaded model"
                )
            batch_features.append(features.astype(np.float32, copy=False))
            hit_items.append(item_id)
        if batch_features:
            matrix = np.stack(batch_features)
            if (current_info or {}).get("type") == "lightgbm":
                scores = current_model.predict_proba(matrix).reshape(-1).tolist()
            else:
                device = next(current_model.parameters()).device
                with torch.no_grad():
                    scores = (
                        current_model(
                            torch.tensor(matrix, dtype=torch.float32, device=device)
                        )
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
