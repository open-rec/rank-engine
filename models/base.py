"""Serving contracts; training definitions remain in rec-algorithm."""

from dataclasses import dataclass
from typing import Any, Protocol


class Predictor(Protocol):
    def predict(self, matrix) -> list[float]: ...


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    predictor: Predictor
    metadata: dict


class ModelDefinition(Protocol):
    def load(self, info, dim, device) -> LoadedModel: ...

    def prepare(self, loaded: LoadedModel, snapshot):
        """Return a snapshot-specific executor, ready for publication."""
        ...
