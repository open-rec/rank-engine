"""Explicit supported-model registry; no model branching in the services."""

from models.fm.model import FMDefinition
from models.lightgbm.model import LightGBMDefinition
from models.lr.model import LRDefinition


MODEL_REGISTRY = {
    "lr": LRDefinition(),
    "fm": FMDefinition(),
    "lightgbm": LightGBMDefinition(),
}
