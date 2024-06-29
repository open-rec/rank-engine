from pydantic import BaseModel, Field
from typing import Any, Optional


class Model(BaseModel):
    model: str = Field(default="lr.pth")
    dim: int = Field(default=1024)
    type: str = Field(default="lr")


class UserItems(BaseModel):
    user_id: str = Field(default="", description="user id")
    item_ids: list = Field(default=[], description="user score items")


class ReResponse(BaseModel):
    code: int = Field(default=0)
    status: str = Field(default="success")
    data: Any = Field(default=None, description="data returned by api")
    message: str = Field(default="")
