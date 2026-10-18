import logging
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
)

from config import Config
from error_code import ErrorCode, ReException
from proto import Model, ReResponse, UserItems
from service.feature_service import FeatureService
from service.model_service import ModelService
from service.inference_service import InferenceService
from util.redis_util import get_redis_client

app = FastAPI(title="OpenRec Rank Engine", version="0.2.0")
feature_service = FeatureService()
model_service = ModelService(feature_service)
inference_service = InferenceService(model_service, feature_service)
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
    with model_service.model_lock:
        model_loaded.set(int(model_service.current("item") is not None))
        for target in ("item", "user"):
            models_loaded.labels(target).set(int(model_service.current(target) is not None))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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
    model_service.startup()


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


@app.get("/health")
def health():
    try:
        redis_ok = bool(get_redis_client().ping())
    except Exception:
        redis_ok = False
    with model_service.model_lock:
        item_runtime = model_service.current("item")
        user_runtime = model_service.current("user")
        loaded = item_runtime is not None
        current = item_runtime.info if loaded else None
        user_loaded = user_runtime is not None
        user_current = user_runtime.info if user_loaded else None
        features = feature_service.stats()
    required = {"item"}
    if Config.MODEL.USER_PATH or model_service.state_path("user").exists() or user_loaded:
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


@app.post("/model/load")
def load_model(info: Model):
    try:
        return response(model_service.load(info))
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
            model_service.refresh_features(target, force=True)
        return response(feature_service.stats())
    except Exception:
        logging.exception("feature refresh failed")
        raise ReException(ErrorCode.LOAD_MODEL_FAILED)


@app.post("/clean")
def clean():
    model_service.clean()
    return response(message="model unloaded")


@app.post("/model/score")
def score(user_items: UserItems):
    return response(inference_service.score(user_items))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=Config.SERVER.HOST, port=Config.SERVER.PORT)
