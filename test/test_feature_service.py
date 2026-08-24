import numpy as np

from service.feature_service import FeatureService


def fresh_service():
    # FeatureService is decorated as a singleton; reset only the mutable fields relevant here.
    service = FeatureService()
    service.user_feature_map = {}
    service.item_feature_map = {}
    service.feature_dim = 0
    service.feature_file = None
    service.loaded_at = 0
    return service


def test_activate_swaps_complete_snapshot():
    service = fresh_service()
    value = {"users": {"u": np.array([1.])}, "items": {"i": np.array([2.])},
             "dim": 2, "feature_file": "space.json"}
    service.activate(value)
    assert service.get_user_feature_by_id("u").tolist() == [1.]
    assert service.get_item_feature_by_id("i").tolist() == [2.]
    assert service.stats() == {"users": 1, "items": 1, "dim": 2}
    assert service.feature_file == "space.json"


def test_refresh_only_reloads_when_snapshot_is_stale(monkeypatch):
    service = fresh_service()
    calls = []
    monkeypatch.setattr(service, "load_all_features", lambda: calls.append(True))
    monkeypatch.setattr("service.feature_service.time.monotonic", lambda: 100.)

    service.loaded_at = 95.
    service.refresh_if_stale(10)
    assert calls == []

    service.loaded_at = 80.
    service.refresh_if_stale(10)
    assert calls == [True]

    service.refresh_if_stale(0)
    assert calls == [True]
