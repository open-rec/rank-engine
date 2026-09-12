import json
from typing import List, Optional

from pydantic import BaseModel, Field


class Model(BaseModel):
    model: str = Field(default="lr.pth")
    dim: int = Field(default=1024)
    type: str = Field(default="lr")
    feature: Optional[str] = Field(default=None, description="optional FeatureSpace JSON sidecar")
    factor_dim: Optional[int] = Field(default=None, ge=1, le=256)


class TrainModel(BaseModel):
    scene: str
    version: str
    business_date: str
    revision: str
    dataset_dir: str
    epochs: int = Field(default=5, ge=1, le=100)
    batch_size: int = Field(default=256, ge=1)
    validation_ratio: float = Field(default=.2, gt=0, lt=1)
    min_auc: float = Field(default=0.0, ge=0, le=1)
    model_type: str = Field(default="lr", pattern="^(lr|fm)$")
    factor_dim: int = Field(default=8, ge=1, le=256)
    label_observation_cutoff: int = Field(ge=0)
    input_label_count: int = Field(ge=0)
    materialized_label_count: Optional[int] = Field(default=None, ge=0)
    history_row_count: Optional[int] = Field(default=None, ge=0)
    materialization_seconds: Optional[float] = Field(default=None, ge=0)
    feature_cutoff_time: int = Field(ge=0)
    feature_until_time: Optional[int] = Field(default=None, ge=0)
    target_type: str = Field(default="item", pattern="^(item|user)$")


class UserItems(BaseModel):
    user_id: str = Field(default="", description="user id")
    item_ids: List[str] = Field(default_factory=list, description="user score items")
    candidate_ids: List[str] = Field(default_factory=list, description="generic score candidates")
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
