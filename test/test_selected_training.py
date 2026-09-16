import json

import numpy as np
import pandas as pd
import pytest
import torch

from algorithm.feature.feature_space import FeatureSpace
from proto import Model, UserItems
from algorithm.rank.training import TrainingRequest, train_release
import server


@pytest.mark.parametrize("kind", ["lr", "fm"])
@pytest.mark.parametrize("target", ["item", "user"])
def test_subset_requires_publish_and_preserves_encoding(
    tmp_path, monkeypatch, kind, target
):
    directory = tmp_path / "training" / "run"
    directory.mkdir(parents=True)
    source = {"id": "u", "age": 28, "gender": True}
    candidate = (
        {"id": "i", "weight": 2, "category": "film"}
        if target == "item"
        else {"id": "i", "age": 35, "gender": False}
    )
    events, users, items = [], [], []
    for index in range(40):
        identity = str(index)
        events.append(
            {
                "_sample_id": identity,
                "user_id": "u",
                "item_id": "i",
                "type": "click" if index % 2 else "expose",
                "time": 1000 + index,
                "trace_id": identity,
            }
        )
        users.append(dict(source, _sample_id=identity))
        items.append(dict(candidate, _sample_id=identity))
    for name, rows in (
        ("events", events),
        ("sample_users", users),
        ("sample_items", items),
    ):
        (directory / (name + ".jsonl")).write_text(
            "\n".join(json.dumps(row) for row in rows)
        )
    selection = {
        "user": ["user.age"],
        "candidate": ["item.weight"]
        if target == "item"
        else ["user.age", "user.gender"],
    }
    request = TrainingRequest(
        scene="global",
        version="test-v1",
        business_date="2026-09-16",
        revision="r001",
        dataset_dir=str(directory),
        epochs=1,
        batch_size=1,
        model_type=kind,
        target_type=target,
        feature_selection=selection,
        label_observation_cutoff=2000,
        feature_cutoff_time=1000,
        input_label_count=40,
    )
    manifest = train_release(request, tmp_path / "releases")
    assert server.model is None and server.user_model is None
    assert manifest["feature_selection"] == selection
    release = tmp_path / "releases" / target / "global" / "test-v1"
    sidecar = release / manifest["feature"]
    space = FeatureSpace.load(sidecar)
    assert space.selection == selection
    users_frame = pd.DataFrame(
        [source] + ([candidate] if target == "user" else [])
    )
    values = {
        "user:*": dict(enumerate(users_frame.to_dict("records"))),
        "item:*": {0: candidate},
        "feature:user:*": {},
        "feature:item:*": {},
    }
    monkeypatch.setattr(
        server.feature_service,
        "_batch_load",
        lambda pattern, batch_size=500: values[pattern],
    )
    server.load_model(
        Model(
            type=kind,
            model=str(release / manifest["model"]),
            feature=str(sidecar),
        )
    )
    current = server.model if target == "item" else server.user_model
    vector = np.concatenate(
        (
            space.transform_users(pd.DataFrame([source])),
            space.transform_items(pd.DataFrame([candidate])),
        ),
        axis=1,
    )
    expected = current(torch.tensor(vector, dtype=torch.float32)).item()
    actual = server.score(
        UserItems(user_id="u", candidate_ids=["i"], target_type=target)
    )["data"]["i"]
    assert actual == pytest.approx(expected)
    assert not directory.exists()


def test_load_rejects_unmaterialized_feature(tmp_path, monkeypatch):
    users = pd.DataFrame([{"id": "u", "age": 20}])
    items = pd.DataFrame([{"id": "i", "weight": 2}])
    space = FeatureSpace.for_model(
        "lr",
        selection={
            "user": ["user.event_count_7d"],
            "candidate": ["item.weight"],
        },
    ).fit(users, items)
    sidecar = tmp_path / "features.json"
    space.save(sidecar)
    values = {
        "user:*": {0: {"id": "u", "age": 20}},
        "item:*": {0: {"id": "i", "weight": 2}},
        "feature:user:*": {},
        "feature:item:*": {},
    }
    monkeypatch.setattr(
        server.feature_service,
        "_batch_load",
        lambda pattern, batch_size=500: values[pattern],
    )
    with pytest.raises(ValueError, match="no materialized values"):
        server.feature_service.prepare_all_features(str(sidecar))


def test_publication_during_request_keeps_original_model_and_features(
    monkeypatch,
):
    from algorithm.rank.lr import LRModel

    old = LRModel(2)
    with torch.no_grad():
        old.linear.weight.fill_(1)
        old.linear.bias.zero_()
    old.feature_snapshot = {
        "users": {"u": np.array([1.0])},
        "items": {"i": np.array([2.0])},
    }
    server.model = old
    server.model_info = {"target_type": "item"}

    def publish_during_refresh(target):
        server.model = LRModel(8)
        server.model.feature_snapshot = {
            "users": {"u": np.ones(4)},
            "items": {"i": np.ones(4)},
        }

    monkeypatch.setattr(server, "_refresh_features", publish_during_refresh)
    result = server.score(UserItems(user_id="u", item_ids=["i"]))
    assert result["data"]["i"] == pytest.approx(
        torch.sigmoid(torch.tensor(3.0)).item()
    )


def test_failed_refresh_retains_usable_snapshot(monkeypatch):
    from algorithm.rank.lr import LRModel

    server.model = LRModel(2)
    server.model.feature_snapshot = {
        "users": {"u": np.array([1.0])},
        "items": {"i": np.array([2.0])},
    }

    def fail(target):
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr(server, "_refresh_features", fail)
    assert server.score(UserItems(user_id="u", item_ids=["i"]))["code"] == 0
