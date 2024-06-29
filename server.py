import torch
import asyncio
from torch import nn
from pydantic import BaseModel
from starlette.middleware.base import _StreamingResponse
from fastapi import FastAPI, Request, Response, HTTPException

from proto import Model, UserItems, ReResponse
from model import model_func_map
from error_code import ReException, ErrorCode


app = FastAPI()
model: nn.Module = None


@app.exception_handler(HTTPException)
async def exception_handler(request, exception: ReException):
    return ReResponse(code=exception.status_code, status="fail", data=None, message=exception.detail)


@app.middleware("http")
async def response_format(request: Request, call_next):
    response = await call_next(request)
    if type(response) is ReResponse:
        return response
    elif type(response) is _StreamingResponse:
        async def build_content(stream_response):
            content = b''
            async for chunk in stream_response.body_iterator:
                content += chunk
            return content
        content = await build_content(response)
        return ReResponse(code=0, status="success", data=content, headers=dict(response.headers))
    else:
        return ReResponse(code=0, status="success", data=response.body(), headers=dict(response.headers))


@app.get("/")
def index():
    return "rank engine works"


@app.get("/health")
def health():
    return "ok"


@app.post("/model/load")
async def load_model(model_info: Model):
    global model
    if model_info.type in model_func_map:
        model = model_func_map[model_info.type](model_info.dim)
        model.load_state_dict(torch.load(model_info.model))


@app.post("/clean")
def clean():
    torch.cuda.empty_cache()


@app.post("/score")
def score(user_items: UserItems):
    if not model:
        raise ReException(ErrorCode.MODEL_NOT_LOAD_YET)
    with torch.no_grad():
        #todo, get user feature by id
        user_features = None
        batch_features = []
        for item_id in user_items.item_ids:
            #todo, get item feature by id
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

