import logging
import os
import threading

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

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
model_lock = threading.RLock()
load_lock = threading.Lock()
feature_service = FeatureService()


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
        feature_service.load_all_features(str(feature_file) if feature_file else None)
        effective_dim = feature_service.feature_dim if feature_file else info.dim
        device = model_device()
        loaded_model = model_func_map[model_type](effective_dim)
        loaded_model.load_state_dict(torch.load(info.model, map_location=device))
        loaded_model.to(device)
        loaded_model.eval()
        with model_lock:
            model = loaded_model
            model_info = {"type": model_type, "path": info.model,
                          "feature": str(feature_file) if feature_file else None,
                          "dim": effective_dim, "device": str(device)}
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
