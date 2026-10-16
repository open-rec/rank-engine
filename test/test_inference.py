import copy

import numpy as np
import pandas as pd
import pytest
import torch

from algorithm.feature.context_feature import materialize_request_context
from algorithm.feature.feature_space import (
    BOOL, HASH, ID, MULTI, NUM, ColumnSpec, FeatureSpace,
)
from algorithm.rank.fm import FMModel
from algorithm.rank.lr import LRModel
from proto import UserItems
from models.common.projection import prepare_torch
from models.base import LoadedModel
from backends.torch_backend import TorchBackend
from helpers import install_model
from helpers import runtime_for
import server


def serving_model(kind, dynamic=True):
    users = pd.DataFrame([{"id": "u", "age": 20}, {"id": "v", "age": 40}])
    items = pd.DataFrame([{"id": "a", "weight": 1}, {"id": "b", "weight": 3}])
    sessions = pd.DataFrame([{"id": "s", "count": 2}, {"id": "t", "count": 5}])
    contexts = pd.DataFrame([
        {"position_ratio": 0, "device": "phone", "subscribed": True,
         "tags": "a/b", "text": "first candidate", "clicks": 0},
        {"position_ratio": 1, "device": "desktop", "subscribed": False,
         "tags": "b/c", "text": "second candidate", "clicks": 4},
    ])
    space = FeatureSpace(
        user_columns=[ColumnSpec("age", NUM)],
        item_columns=[ColumnSpec("weight", NUM)],
        session_columns=[ColumnSpec("count", NUM)] if dynamic else [],
        context_columns=[ColumnSpec("position_ratio", NUM),
                         ColumnSpec("device", ID), ColumnSpec("subscribed", BOOL),
                         ColumnSpec("tags", MULTI), ColumnSpec("text", HASH, hash_dim=16)]
        if dynamic else [],
        interaction_columns=[ColumnSpec("clicks", NUM)] if dynamic else [],
    ).fit(users, items, sessions, contexts, contexts)
    user_map, item_map = space.build_maps(users, items)
    model = (LRModel(space.dim) if kind == "lr"
             else FMModel(space.dim, factor_dim=6)).eval()
    generator = torch.Generator().manual_seed(71)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .2)
    model.feature_snapshot = {
        "space": space, "dim": space.dim, "users": user_map, "items": item_map,
        "sessions": space._build_map(sessions, space.transform_sessions(sessions)),
        "feature_file": "features.json", "loaded_at": float("inf"),
    }
    return model


def reference_scores(model, request):
    """The original row-wise encoder and trained model are the oracle."""
    snapshot = model.feature_snapshot
    space = snapshot["space"]
    ids = request.candidate_ids or request.item_ids
    contexts = materialize_request_context(
        request.context, ids, request.context.get("request_time"), request.candidate_contexts
    )
    result = {}
    with torch.no_grad():
        for position, identity in enumerate(ids):
            candidate = snapshot["items"].get(identity)
            if candidate is None:
                result[identity] = 0.0
                continue
            row = contexts.iloc[[position]]
            vector = np.concatenate((
                snapshot["users"].get(request.user_id, np.zeros(space.user_width)),
                candidate,
                snapshot["sessions"].get(request.session_id, np.zeros(space.session_width)),
                space.transform_contexts(row)[0], space.transform_interactions(row)[0],
            )).astype(np.float32)
            result[identity] = model(torch.from_numpy(vector[None, :])).item()
    return result


@pytest.mark.parametrize("kind", ["lr", "fm"])
@pytest.mark.parametrize("target", ["item", "user"])
@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("user,session", [("u", "s"), ("unknown", "s"), ("u", "")])
def test_batched_and_projected_scores_preserve_feature_semantics(
    kind, target, projected, dynamic, user, session, monkeypatch
):
    model = serving_model(kind, dynamic)
    install_model(model, target=target, kind=kind, projected=projected)
    request = UserItems(
        user_id=user, session_id=session, target_type=target,
        candidate_ids=["missing", "a", "b", "absent", "a"],
        context={"request_time": 1700000000, "device": "phone", "subscribed": "yes"},
        candidate_contexts={"b": {"device": "unknown", "clicks": 3,
                                  "tags": ["c", "a"], "text": "new text",
                                  "subscribed": False}},
    )
    expected = reference_scores(model, request)
    actual = server.score(request)["data"]
    assert actual == pytest.approx(expected, abs=1e-6, rel=1e-5)
    assert actual["missing"] == actual["absent"] == 0.0


@pytest.mark.parametrize("kind", ["lr", "fm"])
def test_refresh_atomically_replaces_projections(kind, monkeypatch):
    model = serving_model(kind)
    original = model.feature_snapshot
    original_runtime = install_model(model, projected=True)
    original_values = original_runtime.executor.projected.blocks["items"][1].copy()
    request = UserItems(user_id="u", candidate_ids=["a", "b"], session_id="s")
    before = server.score(request)["data"]
    replacement = {**original, "items": {key: row + 1 for key, row in original["items"].items()}}
    monkeypatch.setattr(server.feature_service, "prepare_all_features", lambda path: replacement)
    server.model_service.refresh_features("item", force=True)
    replacement_runtime = server.model_service.current("item")
    assert replacement_runtime.snapshot is replacement
    assert replacement_runtime.executor.projected is not original_runtime.executor.projected
    model.feature_snapshot = replacement  # Reference oracle uses the new feature rows.
    assert server.score(request)["data"] == pytest.approx(reference_scores(model, request), abs=1e-6)
    assert server.score(request)["data"] != before
    np.testing.assert_array_equal(original_runtime.executor.projected.blocks["items"][1], original_values)


def test_failed_projection_refresh_retains_previous_snapshot(monkeypatch):
    model = serving_model("fm")
    original = model.feature_snapshot
    original_runtime = install_model(model, projected=True)
    invalid = {**original, "items": {"a": np.zeros(3)}}
    monkeypatch.setattr(server.feature_service, "prepare_all_features", lambda path: invalid)
    with pytest.raises(ValueError, match="feature block"):
        server.model_service.refresh_features("item", force=True)
    assert server.model_service.current("item") is original_runtime


def test_refresh_during_encoding_keeps_request_projection_snapshot(monkeypatch):
    model = serving_model("fm")
    original = model.feature_snapshot
    original_runtime = install_model(model, projected=True)
    request = UserItems(user_id="u", candidate_ids=["a", "b"])
    expected = reference_scores(model, request)
    replacement = copy.deepcopy(original)
    replacement["items"]["a"] += 5
    replacement_runtime = runtime_for(model, snapshot=replacement, projected=True)
    transform = original["space"].transform_contexts

    def refresh_while_encoding(frame):
        server.model_service.runtimes["item"] = replacement_runtime
        return transform(frame)

    monkeypatch.setattr(original["space"], "transform_contexts", refresh_while_encoding)
    assert server.score(request)["data"] == pytest.approx(expected, abs=1e-6)


def test_custom_forward_is_not_replaced_by_projection():
    class CustomLR(LRModel):
        def forward(self, x):
            return super().forward(x) * .5

    snapshot = serving_model("lr").feature_snapshot
    custom = CustomLR(snapshot["dim"])
    loaded = LoadedModel(custom, TorchBackend(custom), {})
    assert prepare_torch(loaded, snapshot, LRModel).projected is None


@pytest.mark.parametrize("factor_dim", [0, 1, 8, 64])
def test_projection_chunk_boundaries_match_dense_model(factor_dim):
    rng = np.random.default_rng(17)
    space = FeatureSpace(
        user_columns=[ColumnSpec("user", HASH, hash_dim=200)],
        item_columns=[ColumnSpec("candidate", HASH, hash_dim=223)],
        session_columns=[ColumnSpec("session", HASH, hash_dim=3)],
        context_columns=[ColumnSpec("context", HASH, hash_dim=4)],
    )
    model = (FMModel(space.dim, factor_dim) if factor_dim else LRModel(space.dim)).eval()
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .01)
    user = rng.uniform(-3, 3, space.user_width)
    items = rng.uniform(-3, 3, (1100, space.item_width))
    session = rng.uniform(-3, 3, space.session_width)
    contexts = rng.uniform(-3, 3, (1100, space.context_width))
    snapshot = {
        "space": space, "dim": space.dim, "users": {"u": user},
        "items": {str(i): row for i, row in enumerate(items)}, "sessions": {"s": session},
    }
    runtime = runtime_for(model, snapshot=snapshot, projected=True)
    actual = runtime.executor.projected.score("u", list(snapshot["items"]), "s", [contexts])
    matrix = np.concatenate((np.broadcast_to(user, (len(items), len(user))), items,
                             np.broadcast_to(session, (len(items), len(session))), contexts), axis=1)
    with torch.no_grad():
        expected = model(torch.tensor(matrix, dtype=torch.float32)).reshape(-1).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_all_missing_candidates_do_not_encode_request(monkeypatch):
    model = serving_model("lr")
    install_model(model, projected=True)
    monkeypatch.setattr(model.feature_snapshot["space"], "transform_contexts",
                        lambda frame: pytest.fail("no known candidates to encode"))
    assert server.score(UserItems(candidate_ids=["missing"]))["data"] == {"missing": 0.0}
