"""Extending the registry must not require framework branches in services."""

from types import SimpleNamespace

import numpy as np
import pytest

from error_code import ErrorCode, ReException
from inference.batch import InferenceBatch, prepare_batch
from inference.runtime import BatchExecutor
from models.base import LoadedModel
from proto import Model, UserItems
from service.inference_service import InferenceService
from service.model_service import ModelService
from test_inference import serving_model
from helpers import install_model
import server


class SumBackend:
    def predict(self, matrix):
        return matrix.sum(axis=1).tolist()


class SumDefinition:
    def load(self, info, dim, device):
        return LoadedModel(SimpleNamespace(dim=dim), SumBackend(), {"implementation": "sum"})

    def prepare(self, loaded, snapshot):
        return BatchExecutor(loaded.predictor)


def test_registered_model_load_score_refresh_and_restore(monkeypatch):
    snapshot = {"dim": 2, "model_type": "sum", "target_type": "item",
                "feature_file": "features.json", "users": {"u": np.array([1.])},
                "items": {"i": np.array([2.])}}
    monkeypatch.setattr(server.feature_service, "prepare_all_features", lambda path: dict(snapshot))
    models = ModelService(server.feature_service, registry={"sum": SumDefinition()})
    inference = InferenceService(models, server.feature_service)
    info = models.load(Model(type=" SUM ", model="sum.model", feature="features.json"))
    assert info["implementation"] == "sum"
    original = models.current("item")
    assert not hasattr(original.model, "feature_snapshot")
    assert "scorer" not in original.snapshot
    request = UserItems(user_id="u", item_ids=["i", "absent"])
    assert inference.score(request) == {"i": 3., "absent": 0.}

    snapshot["items"] = {"i": np.array([4.])}
    models.refresh_features("item", force=True)
    assert models.current("item") is not original
    assert original.executor.score(  # The retired version remains independently usable.
        prepare_batch(
            request, original, server.feature_service
        )
    ) == [3.]
    assert inference.score(request)["i"] == 5.
    models.clean()
    models.restore("item")
    assert inference.score(request)["i"] == 5.


def test_unknown_model_does_not_prepare_features(monkeypatch):
    monkeypatch.setattr(server.feature_service, "prepare_all_features",
                        lambda path: pytest.fail("unsupported model must fail before IO"))
    models = ModelService(server.feature_service, registry={})
    with pytest.raises(ReException) as error:
        models.load(Model(type="lr", model="unused.pth"))
    assert error.value.error_code is ErrorCode.INVALID_MODEL


def test_projection_never_assembles_unused_dense_matrix(monkeypatch):
    install_model(serving_model("fm"), projected=True)
    monkeypatch.setattr(InferenceBatch, "matrix",
                        lambda self: pytest.fail("projection must not assemble a dense batch"))
    assert server.score(UserItems(user_id="u", item_ids=["a"]))["code"] == 0
