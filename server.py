import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Gauge, generate_latest

from algorithm.utils.file_util import resolve_feature_file
from config import Config
from error_code import ErrorCode, ReException
from model import model_func_map
from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.user_feature import UserFeature
from algorithm.rank.lr import LRRecModel
from algorithm.rank.fm import FMRecModel
from proto import Model, ReResponse, TrainModel, UserItems
from service.feature_service import FeatureService
from util.redis_util import get_redis_client

app = FastAPI(title="OpenRec Rank Engine", version="1.0")
model = None
model_info = None
model_lock = threading.RLock()
load_lock = threading.Lock()
feature_service = FeatureService()
request_count = Counter("openrec_rank_requests", "Rank Engine requests", ["method", "path", "status"])
request_latency = Histogram("openrec_rank_request_latency_seconds", "Rank Engine request latency", ["method", "path"])
model_loaded = Gauge("openrec_rank_model_loaded", "Whether a ranking model is loaded")


@app.middleware("http")
async def observe_request(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    with request_latency.labels(request.method, request.url.path).time():
        try:
            result = await call_next(request)
            request_count.labels(request.method, request.url.path, str(result.status_code)).inc()
            return result
        except Exception:
            request_count.labels(request.method, request.url.path, "500").inc()
            raise


@app.get("/metrics")
def metrics():
    with model_lock:
        model_loaded.set(1 if model is not None else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def model_device():
    configured = Config.MODEL.DEVICE
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MODEL_DEVICE requests CUDA but no CUDA device is available")
    return torch.device(configured)


def response(data=None, message=""):
    return ReResponse(code=0, status="success", data=data, message=message).to_dict()


@app.exception_handler(ReException)
async def re_exception_handler(request: Request, exception: ReException):
    error = exception.error_code
    return JSONResponse(status_code=200, content=ReResponse(
        code=error.code, status="fail", data=None, message=error.message).to_dict())


@app.exception_handler(Exception)
async def unknown_exception_handler(request: Request, exception: Exception):
    logging.exception("unhandled request error")
    return JSONResponse(status_code=200, content=ReResponse(
        code=ErrorCode.UNKNOWN_ERROR.code, status="fail", data=None,
        message=ErrorCode.UNKNOWN_ERROR.message).to_dict())


@app.on_event("startup")
def startup():
    for directory in (Path("/models/training"), Path("/models/releases")):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o777 if directory.name == "training" else 0o755)
    if not Config.MODEL.PATH:
        logging.warning("MODEL_PATH is not configured; score requests remain disabled")
        return
    try:
        _load_model(Model(type=Config.MODEL.TYPE, model=Config.MODEL.PATH,
                          feature=Config.MODEL.FEATURE_PATH, dim=Config.MODEL.DIM))
    except Exception:
        logging.exception("automatic model load failed")
        if Config.MODEL.REQUIRED:
            raise


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
    return response({"redis": redis_ok, "model_loaded": loaded,
                     "model": current, "features": feature_service.stats(),
                     "ready": redis_ok and loaded})


def _load_model(info):
    global model, model_info
    model_type = info.type.strip().lower()
    if model_type not in model_func_map:
        raise ReException(ErrorCode.INVALID_MODEL)
    with load_lock:
        feature_file = info.feature or resolve_feature_file(info.model)
        snapshot = feature_service.prepare_all_features(
            str(feature_file) if feature_file else None)
        declared_model_type = snapshot.get("model_type")
        if declared_model_type and declared_model_type != model_type:
            raise ValueError("feature space belongs to %s, not %s" % (
                declared_model_type, model_type))
        effective_dim = snapshot["dim"] if feature_file else info.dim
        device = model_device()
        state = torch.load(info.model, map_location=device)
        kwargs = {}
        if model_type == "fm":
            factors = state.get("factors")
            if factors is None or factors.ndim != 2 or factors.shape[0] != effective_dim:
                raise ValueError("FM checkpoint factors do not match the feature dimension")
            kwargs["factor_dim"] = info.factor_dim or factors.shape[1]
        loaded_model = model_func_map[model_type](effective_dim, **kwargs)
        loaded_model.load_state_dict(state)
        loaded_model.to(device)
        loaded_model.eval()
        with model_lock:
            feature_service.activate(snapshot)
            model = loaded_model
            model_info = {"type": model_type, "path": info.model,
                          "feature": str(feature_file) if feature_file else None,
                          "dim": effective_dim, "device": str(device),
                          "feature_set": snapshot.get("feature_set"),
                          "catalog_version": snapshot.get("catalog_version"),
                          **({"factor_dim": loaded_model.factor_dim} if model_type == "fm" else {})}
    return model_info


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
    """Train one immutable artifact version from a Spark-prepared local dataset."""
    dataset = Path(info.dataset_dir).resolve()
    training_root = Path("/models/training").resolve()
    artifact_root = Path("/models/releases").resolve()
    if training_root not in dataset.parents or not re.match(r"^[A-Za-z0-9_-]+$", info.scene):
        raise ReException(ErrorCode.INVALID_MODEL)
    target = artifact_root / info.scene / info.version
    if target.exists():
        raise ReException(ErrorCode.LOAD_MODEL_FAILED)
    scene_root = artifact_root / info.scene
    scene_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".%s-" % info.version,
                                   dir=str(scene_root)))
    try:
        events = pd.read_json(dataset / "events.jsonl", lines=True)
        try:
            feature_events = pd.read_json(dataset / "feature_events.jsonl", lines=True)
        except pd.errors.EmptyDataError:
            # A cold-start label window can legitimately have no strictly-prior behaviour.
            feature_events = pd.DataFrame()
        items = pd.read_json(dataset / "items.jsonl", lines=True)
        users = pd.read_json(dataset / "users.jsonl", lines=True)
        model_type = info.model_type.strip().lower()
        model_filename = "%s.pth" % model_type
        feature_filename = "%s.features.json" % model_type
        model_class = {"lr": LRRecModel, "fm": FMRecModel}[model_type]
        model_kwargs = {"factor_dim": info.factor_dim} if model_type == "fm" else {}
        user_features = UserFeature(users, feature_events, as_of_time=info.feature_cutoff_time)
        item_features = ItemFeature(items, feature_events, as_of_time=info.feature_cutoff_time)
        rank_model = model_class(
            user_features, item_features, events,
            scene=info.scene, model_file=staging / model_filename,
            feature_file=staging / feature_filename, **model_kwargs)
        if not len(rank_model.dataset):
            raise ValueError("rank training produced no labelled samples after entity filtering")
        if rank_model.dataset.positive_rate in (0.0, 1.0):
            raise ValueError("rank training requires both click and expose labels")
        rank_model.train(epoch_num=info.epochs, batch_size=info.batch_size,
                         val_ratio=info.validation_ratio)
        _, validation = rank_model._split(val_ratio=info.validation_ratio, seed=42)
        auc = rank_model.evaluate(validation, batch_size=info.batch_size)
        if auc is None:
            raise ValueError("AUC is undefined for validation data")
        if auc is not None and auc < info.min_auc:
            raise ValueError("AUC %.6f is below %.6f" % (auc, info.min_auc))
        rank_model.save()
        # Keep the unencoded, point-in-time entity snapshots next to the checkpoint. They are the
        # portable bootstrap representation for Redis; *.features.json remains the model-specific
        # encoding contract and must not be confused with actual entity feature values.
        for frame, filename in ((user_features.users, "user_feature.csv"),
                                (item_features.items, "item_feature.csv")):
            exported = frame.copy()
            exported.insert(1, "as_of_time", info.feature_cutoff_time)
            exported.to_csv(staging / filename, index=False)
        feature_bytes = (staging / feature_filename).read_bytes()
        feature_sha256 = hashlib.sha256(feature_bytes).hexdigest()
        feature_space = rank_model.dataset.feature_space
        manifest = {"version": info.version, "scene": info.scene, "model_type": model_type,
                    "business_date": info.business_date, "revision": info.revision,
                    "feature_cutoff_time": info.feature_cutoff_time,
                    "created_at": datetime.now(timezone.utc).isoformat(), "status": "evaluated",
                    "model": model_filename, "feature": feature_filename,
                    "user_feature_snapshot": "user_feature.csv",
                    "item_feature_snapshot": "item_feature.csv",
                    "feature_set": feature_space.feature_set,
                    "catalog_version": feature_space.catalog_version,
                    "feature_sha256": feature_sha256,
                    "input_dim": rank_model.model.dim,
                    "metrics": {"auc": auc, "positive_rate": rank_model.dataset.positive_rate,
                                "samples": len(rank_model.dataset),
                                "feature_dim": rank_model.model.dim,
                                **({"factor_dim": rank_model.model.factor_dim}
                                   if model_type == "fm" else {})},
                    "gate": {"min_auc": info.min_auc, "passed": True}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
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
        feature_service.load_all_features()
        return response(feature_service.stats())
    except Exception:
        logging.exception("feature refresh failed")
        raise ReException(ErrorCode.LOAD_MODEL_FAILED)


@app.post("/clean")
def clean():
    global model, model_info
    with model_lock:
        model = None
        model_info = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return response(message="model unloaded")


@app.post("/model/score")
def score(user_items: UserItems):
    with model_lock:
        current_model = model
    if current_model is None and Config.MODEL.PATH:
        try:
            _load_model(Model(type=Config.MODEL.TYPE, model=Config.MODEL.PATH,
                              feature=Config.MODEL.FEATURE_PATH, dim=Config.MODEL.DIM))
            with model_lock:
                current_model = model
        except Exception:
            logging.exception("lazy model load failed")
    if current_model is None:
        raise ReException(ErrorCode.MODEL_NOT_LOAD_YET)
    if not user_items.item_ids:
        return response({})
    try:
        feature_service.refresh_if_stale(Config.MODEL.FEATURE_REFRESH_SECONDS)
        user_features = feature_service.get_user_feature_by_id(user_items.user_id)
        batch_features = []
        item_score_map = {}
        hit_items = []
        for item_id in user_items.item_ids:
            item_features = feature_service.get_item_feature_by_id(item_id)
            if item_features is None:
                item_score_map[item_id] = 0.0
                continue
            effective_user = user_features
            if effective_user is None:
                effective_user = np.zeros(current_model.dim - item_features.size)
            features = np.concatenate((effective_user, item_features))
            if features.size != current_model.dim:
                raise ValueError("feature dimension does not match loaded model")
            batch_features.append(torch.tensor(features, dtype=torch.float32,
                                                device=next(current_model.parameters()).device))
            hit_items.append(item_id)
        if batch_features:
            with torch.no_grad():
                scores = current_model(torch.stack(batch_features)).reshape(-1).tolist()
            item_score_map.update(zip(hit_items, scores))
        return response(item_score_map)
    except Exception:
        logging.exception("inference failed")
        raise ReException(ErrorCode.INFERENCE_FAILED)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.SERVER.HOST, port=Config.SERVER.PORT)
