import threading
from types import SimpleNamespace

import numpy as np

import server
from algorithm.rank.lr import LRModel
from proto import UserItems


def test_slow_refresh_does_not_block_scoring_or_duplicate_workers(monkeypatch):
    active = LRModel(2)
    active.feature_snapshot = {
        'users': {'u': np.array([1.0])},
        'items': {'i': np.array([2.0])},
        'feature_file': 'features.json', 'loaded_at': 0,
    }
    monkeypatch.setattr(server, 'model', active)
    monkeypatch.setattr(server.Config.MODEL, 'FEATURE_REFRESH_SECONDS', 10)
    started, release, scored = threading.Event(), threading.Event(), threading.Event()
    calls, results = [], []

    def refresh(target):
        calls.append(target)
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(server, '_refresh_features', refresh)

    def score():
        results.append(server.score(UserItems(user_id='u', item_ids=['i'])))
        scored.set()

    request = threading.Thread(target=score)
    request.start()
    worker = None
    try:
        assert started.wait(2)
        with server.refresh_lock:
            worker = server.refresh_workers['item']
        assert scored.wait(2), 'inference waited for feature refresh'
        for _ in range(10):
            server._schedule_feature_refresh('item')
        assert calls == ['item']
        assert results[0]['code'] == 0
    finally:
        release.set()
        request.join(5)
        if worker:
            worker.join(5)


def test_failed_refresh_keeps_snapshot_and_throttles_retries(monkeypatch):
    snapshot = {'feature_file': 'features.json', 'loaded_at': 0}
    active = SimpleNamespace(feature_snapshot=snapshot)
    monkeypatch.setattr(server, 'model', active)
    monkeypatch.setattr(server.Config.MODEL, 'FEATURE_REFRESH_SECONDS', 60)
    calls = []

    def fail(target):
        calls.append(target)
        raise ConnectionError('offline')

    monkeypatch.setattr(server, '_refresh_features', fail)
    server._schedule_feature_refresh('item')
    with server.refresh_lock:
        worker = server.refresh_workers.get('item')
    if worker:
        worker.join(5)
    server._schedule_feature_refresh('item')
    assert calls == ['item']
    assert active.feature_snapshot is snapshot


def test_clean_during_refresh_does_not_reactivate_removed_model(monkeypatch):
    active = SimpleNamespace(dim=2, feature_snapshot={'feature_file': 'features.json'})
    monkeypatch.setattr(server, 'model', active)
    activations = []

    def prepare(path):
        monkeypatch.setattr(server, 'model', None)
        return {'dim': 2}

    monkeypatch.setattr(server.feature_service, 'prepare_all_features', prepare)
    monkeypatch.setattr(server.feature_service, 'activate', activations.append)
    server._refresh_features('item', force=True)
    assert activations == []
