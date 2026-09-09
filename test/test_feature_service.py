import numpy as np
import pandas as pd

from service.feature_service import FeatureService


def fresh_service():
    # FeatureService is decorated as a singleton; reset only the mutable fields relevant here.
    service = FeatureService()
    service.namespaces = {
        name: service._empty_snapshot(name) for name in service.NAMESPACES
    }
    return service


def test_activate_swaps_complete_snapshot():
    service = fresh_service()
    value = {"users": {"u": np.array([1.])}, "items": {"i": np.array([2.])},
             "dim": 2, "feature_file": "space.json"}
    service.activate(value)
    assert service.get_user_feature_by_id("u").tolist() == [1.]
    assert service.get_item_feature_by_id("i").tolist() == [2.]
    assert service.stats()["item"] == {
        "users": 1, "candidates": 1, "dim": 2, "feature_file": "space.json"}


def test_item_and_user_namespaces_do_not_overwrite_each_other():
    service = fresh_service()
    service.activate({"users": {"u": np.array([1.])}, "items": {"i": np.array([2.])},
                      "dim": 2, "feature_file": "item.json", "target_type": "item"})
    service.activate({"users": {"u": np.array([3., 4.])},
                      "items": {"v": np.array([5., 6.])}, "dim": 4,
                      "feature_file": "user.json", "target_type": "user"})

    assert service.get_user_feature_by_id("u", namespace="item").tolist() == [1.]
    assert service.get_item_feature_by_id("i").tolist() == [2.]
    assert service.get_user_feature_by_id("u", namespace="user").tolist() == [3., 4.]
    assert service.get_candidate_user_feature_by_id("v").tolist() == [5., 6.]
    assert service.stats()["item"]["dim"] == 2
    assert service.stats()["user"]["dim"] == 4


def test_refresh_only_reloads_when_snapshot_is_stale(monkeypatch):
    service = fresh_service()
    calls = []
    monkeypatch.setattr(service, "load_all_features",
                        lambda feature_file=None, namespace=None: calls.append(namespace))
    monkeypatch.setattr("service.feature_service.time.monotonic", lambda: 100.)

    service.namespaces["user"]["loaded_at"] = 95.
    service.refresh_if_stale(10)
    assert calls == []

    service.namespaces["user"]["loaded_at"] = 80.
    service.refresh_if_stale(10, namespace="user")
    assert calls == ["user"]

    service.refresh_if_stale(0, namespace="user")
    assert calls == ["user"]


def test_merge_event_features_overlays_snapshot_without_recreating_entities():
    entities = pd.DataFrame([{"id": "u1", "country": "CN", "event_count": 1}])
    snapshots = {
        0: {"entityId": "u1", "features": {"event_count": 7, "event_click_count": 3}},
        1: {"entityId": "deleted", "features": {"event_count": 99}},
    }

    merged = fresh_service()._merge_event_features(entities, snapshots)

    assert merged.to_dict("records") == [{
        "id": "u1", "country": "CN", "event_count": 7, "event_click_count": 3,
    }]


def test_load_user_feature_reads_realtime_snapshot(monkeypatch):
    service = fresh_service()
    values = {
        "user:*": {0: {"id": "u1", "country": "CN"}},
        "feature:user:*": {0: {
            "entityId": "u1", "features": {"event_count": 4, "event_click_count": 2},
        }},
    }
    monkeypatch.setattr(service, "_batch_load",
                        lambda pattern, batch_size=500: values[pattern])

    users = service.load_user_feature().users.set_index("id")

    assert users.loc["u1", "event_count"] == 4
    assert users.loc["u1", "event_click_count"] == 2
