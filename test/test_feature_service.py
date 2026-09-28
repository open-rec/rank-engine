import numpy as np
import pandas as pd
import pytest

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


def test_merge_rematerializes_time_dependent_features(monkeypatch):
    monkeypatch.setattr("service.feature_service.time.time", lambda: 200000.)
    entities = pd.DataFrame([{"id": "u1"}])
    snapshots = {0: {"entityId": "u1", "features": {
        "event_last_time": 100000, "event_recency_seconds": 0,
        "event_count_1d": 2, "event_count_7d": 2,
    }, "recentEventTimeCounts": {"100000": 1, "190000": 1}}}

    merged = fresh_service()._merge_event_features(entities, snapshots).iloc[0]

    assert merged.event_recency_seconds == 100000
    assert merged.event_count_1d == 1
    assert merged.event_count_7d == 2


def test_merge_ignores_realtime_snapshot_from_another_catalog():
    entities = pd.DataFrame([{"id": "u1", "country": "CN"}])
    snapshots = {0: {"entityId": "u1", "catalogVersion": 1,
                     "catalogSha256": "different", "features": {"event_count": 1}}}

    merged = fresh_service()._merge_event_features(entities, snapshots)

    assert merged.to_dict("records") == [{"id": "u1", "country": "CN"}]


def test_merge_ignores_stale_catalog_snapshot_for_deleted_entity():
    entities = pd.DataFrame([{"id": "u1"}])
    snapshots = {
        0: {"entityId": "deleted", "catalogVersion": 1,
            "catalogSha256": "different", "features": {"event_count": 99}},
        1: {"entityId": "u1", "features": {"event_count": 4}},
    }

    merged = fresh_service()._merge_event_features(entities, snapshots)

    assert merged.to_dict("records") == [{"id": "u1", "event_count": 4}]


def test_load_user_feature_reads_realtime_snapshot(monkeypatch):
    service = fresh_service()
    values = {
        "user:*": {0: {"id": "u1", "country": "CN"}},
        "feature:user:*": {0: {
            "entityId": "u1", "features": {"event_count": 4, "event_click_count": 2},
            "stringFeatures": {"preferred_categories": "books,music"},
        }},
    }
    monkeypatch.setattr(service, "_batch_load",
                        lambda pattern, batch_size=500: values[pattern])

    users = service.load_user_feature().users.set_index("id")

    assert users.loc["u1", "event_count"] == 4
    assert users.loc["u1", "event_click_count"] == 2
    assert users.loc["u1", "preferred_categories"] == "books,music"


def test_load_item_feature_materializes_content_for_online_encoding(monkeypatch):
    service = fresh_service()
    published = int(pd.Timestamp.now(tz="UTC").timestamp()) - 7200
    values = {
        "item:*": {0: {
            "id": "i1", "title": "Cold Start News", "category": "news",
            "subcategory": "local", "tags": "breaking", "pubTime": published,
        }},
        "feature:item:*": {},
    }
    monkeypatch.setattr(
        service, "_batch_load", lambda pattern, batch_size=500: values[pattern]
    )

    item_feature = service.load_item_feature()
    row = item_feature.items.iloc[0]

    assert row.pub_time == published
    assert 1.9 <= row.content_age_hours <= 2.1
    assert "content_age_hours" in item_feature.materialized_columns


def test_sessions_validate_catalog_and_refresh_recency(monkeypatch):
    from algorithm.feature.feature_catalog import FeatureCatalog
    catalog = FeatureCatalog.load()
    service = fresh_service()
    monkeypatch.setattr("service.feature_service.time.time", lambda: 200.)
    monkeypatch.setattr(service, "_batch_load", lambda *args, **kwargs: {
        0: {"entityId": "good", "catalogVersion": catalog.version,
            "catalogSha256": catalog.sha256, "features": {
                "event_count": 2, "event_last_time": 150, "event_recency_seconds": 0}},
        1: {"entityId": "bad", "catalogVersion": 1, "catalogSha256": "bad",
            "features": {"event_count": 100}},
    })
    rows = service._load_session_features().set_index("id")
    assert rows.loc["good", "event_recency_seconds"] == 50
    assert pd.isna(rows.loc["bad", "event_count"])


def test_sessions_accept_pinned_producer_catalog(monkeypatch):
    service = fresh_service()
    monkeypatch.setattr(service, "_batch_load", lambda *args, **kwargs: {
        0: {"entityId": "s", "catalogVersion": 17, "catalogSha256": "pinned",
            "features": {"event_count": 2}},
    })
    assert service._load_session_features().empty
    assert service._load_session_features((17, "pinned")).iloc[0].event_count == 2


def test_refresh_decays_short_windows_rates_and_commerce_means(monkeypatch):
    monkeypatch.setattr("service.feature_service.time.time", lambda: 200000.)
    features = {
        "event_expose_count_5m": 1, "event_value_sum_5m": 5,
        "event_expose_count_1h": 2, "event_value_sum_1h": 7,
        "event_count_1d": 3, "event_ctr_1d": 1,
        "event_click_price_mean_1d": 100, "event_click_price_mean_30d": 100,
        "event_buy_price_mean_1d": 90, "event_recent_to_long_click_price_ratio": 1,
    }
    stats = {
        "100000": {"count": 1, "count:click": 1, "value_sum": 2,
                   "price_count:click": 1, "price_sum:click": 100},
        "199000": {"count": 1, "count:expose": 1, "value_sum": 5},
        "199999": {"count": 1, "count:click": 1, "value_sum": 3,
                   "price_count:click": 1, "price_sum:click": 20},
    }
    rows = fresh_service()._merge_event_features(pd.DataFrame([{"id": "u"}]), {
        0: {"entityId": "u", "features": features, "recentEventStats": stats}})
    row = rows.iloc[0]
    assert row.event_expose_count_5m == 0
    assert row.event_value_sum_5m == 3
    assert row.event_expose_count_1h == 1
    assert row.event_value_sum_1h == 8
    assert row.event_count_1d == 2
    assert row.event_ctr_1d == 1
    assert row.event_click_price_mean_1d == 20
    assert row.event_click_price_mean_30d == 60
    assert row.event_buy_price_mean_1d == 0
    assert row.event_recent_to_long_click_price_ratio == pytest.approx(1 / 3)
    from algorithm.feature.event_feature import enrich_entity_features
    events = pd.DataFrame([
        {"user_id": "u", "item_id": "i", "scene": "s", "type": kind,
         "time": stamp, "value": value}
        for kind, stamp, value in (("click", 100000, 2), ("expose", 199000, 5),
                                   ("click", 199999, 3))
    ])
    offline = enrich_entity_features(pd.DataFrame([{"id": "u"}]), events,
                                     "user", 200000).iloc[0]
    for name in ("event_expose_count_5m", "event_value_sum_5m",
                 "event_expose_count_1h", "event_value_sum_1h",
                 "event_count_1d", "event_ctr_1d"):
        assert row[name] == pytest.approx(offline[name])


def test_empty_window_histogram_clears_old_values():
    features = {"event_expose_count_5m": 5, "event_ctr_7d": 2,
                "event_click_price_mean_1d": 80}
    fresh_service()._refresh_window_stats(features, {}, 100)
    assert all(value == 0 for value in features.values())
