"""A published model, feature snapshot and executor form one serving version."""

from dataclasses import dataclass
from typing import Any

from models.base import LoadedModel, ModelDefinition


class BatchExecutor:
    def __init__(self, predictor, projected=None):
        self.predictor = predictor
        self.projected = projected

    def score(self, batch):
        if self.projected is not None:
            return self.projected.score(
                batch.request.user_id, batch.candidate_ids,
                batch.request.session_id, batch.dynamic_parts,
            )
        return self.predictor.predict(batch.matrix())


@dataclass(frozen=True)
class ModelRuntime:
    loaded: LoadedModel
    definition: ModelDefinition
    snapshot: Any
    executor: Any
    info: dict

    @property
    def model(self):
        return self.loaded.model

    @property
    def dim(self):
        return self.info["dim"]
