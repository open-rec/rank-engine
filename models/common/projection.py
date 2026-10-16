"""Opt in to projections only for a known, unchanged model implementation."""

from inference.runtime import BatchExecutor
from models.common.projected_scorer import ProjectedScorer


def prepare_torch(loaded, snapshot, model_class):
    model = loaded.model
    space = snapshot.get("space") if snapshot else None
    projected = None
    if (type(model) is model_class
            and next(model.parameters()).device.type == "cpu"
            and space is not None and snapshot.get("dim") == model.dim):
        if space.dim != model.dim:
            raise ValueError("feature dimension does not match loaded model")
        projected = ProjectedScorer(model, snapshot)
    return BatchExecutor(loaded.predictor, projected)
