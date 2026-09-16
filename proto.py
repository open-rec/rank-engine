import json
from typing import List, Optional

from pydantic import BaseModel, Field


class Model(BaseModel):
    model: str = Field(default="lr.pth")
    dim: int = Field(default=1024)
    type: str = Field(default="lr")
    feature: Optional[str] = Field(
        default=None, description="optional FeatureSpace JSON sidecar"
    )
    factor_dim: Optional[int] = Field(default=None, ge=1, le=256)


class UserItems(BaseModel):
    user_id: str = Field(default="", description="user id")
    item_ids: List[str] = Field(
        default_factory=list, description="user score items"
    )
    candidate_ids: List[str] = Field(
        default_factory=list, description="generic score candidates"
    )
    target_type: str = Field(default="item", pattern="^(item|user)$")


class ReResponse:
    def __init__(self, code=0, status="success", data=None, message=""):
        self.code = code
        self.status = status
        self.data = data
        self.message = message

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.to_dict())
