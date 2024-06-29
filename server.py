import json

import torch
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import FileResponse
from starlette.middleware.base import _StreamingResponse
from torch import nn

from error_code import ReException, ErrorCode
from model import model_func_map
from proto import Model, UserItems, ReResponse

app = FastAPI()
model: nn.Module = None

HEALTH_PATH = "/health"
CLEAN_PATH = "/clean"
MODEL_PATH = "/model"
MODEL_LOAD_PATH = MODEL_PATH + "/load"
MODEL_SCORE_PATH = MODEL_PATH + "/score"

PROXY_PATH_SET = {
    HEALTH_PATH,
    CLEAN_PATH,
    MODEL_LOAD_PATH,
    MODEL_SCORE_PATH,
}


@app.exception_handler(HTTPException)
async def exception_handler(request, exception: ReException):
    return Response(
        content=ReResponse(code=exception.status_code, status="fail", data=None, message=exception.detail).to_json(),
        status_code=exception.status_code
    )


@app.middleware("http")
async def response_format(request: Request, call_next):
    response = await call_next(request)

    if request.url.path not in PROXY_PATH_SET:
        return response

    if type(response) is ReResponse:
        return Response(content=response.to_json())
    elif type(response) is _StreamingResponse:
        if response.status_code >= ErrorCode.UNKNOWN_ERROR.code:
            response.status_code = 200
            return response

        async def build_content(stream_response):
            content = b''
            async for chunk in stream_response.body_iterator:
                content += chunk
            return content

        content = await build_content(response)
        content = content.decode("utf-8")
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            content = eval(content)
        return Response(content=ReResponse(code=0, status="success", data=content).to_json())
    else:
        return Response(content=response.body())


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return "ok"


@app.post("/model/load")
async def load_model(model_info: Model):
    global model
    model_type = model_info.type.strip().lower()
    if model_type not in model_func_map:
        raise ReException(ErrorCode.INVALID_MODEL)

    model = model_func_map[model_type](model_info.dim)
    try:
        model.load_state_dict(torch.load(model_info.model))
    except FileNotFoundError as fnfe:
        raise ReException(ErrorCode.MODEL_NOT_FOUND)
    except Exception as e:
        raise ReException(ErrorCode.LOAD_MODEL_FAILED)


@app.post("/clean")
def clean():
    torch.cuda.empty_cache()


@app.post("/model/score")
def score(user_items: UserItems):
    if not model:
        raise ReException(ErrorCode.MODEL_NOT_LOAD_YET)

    try:
        with torch.no_grad():
            # todo, get user feature from redis or es
            user_features = None
            batch_features = []
            for item_id in user_items.item_ids:
                # todo, get item feature from redis or es
                item_features = None
                batch_features.append(torch.cat(
                    (
                        torch.tensor(user_features, dtype=torch.float32),
                        torch.tensor(item_features, dtype=torch.float32)
                    ),
                    dim=0
                ))
            score = model(torch.stack(batch_features))
            return score.squeeze().tolist()
    except Exception as e:
        raise (ReException(ErrorCode.INFERENCE_FAILED))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
