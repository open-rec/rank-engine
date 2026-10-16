import threading
from types import SimpleNamespace

import numpy as np

from helpers import install_model
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
    install_model(active, target="item")
    monkeypatch.setattr(server.Config.MODEL, 'FEATURE_REFRESH_SECONDS', 10)
    started, release, scored = threading.Event(), threading.Event(), threading.Event()
    calls, results = [], []

    def refresh(target):
        calls.append(target)
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(server.model_service, 'refresh_features', refresh)

    def score():
        results.append(server.score(UserItems(user_id='u', item_ids=['i'])))
        scored.set()

    request = threading.Thread(target=score)
    request.start()
    worker = None
    try:
        assert started.wait(2)
        with server.model_service.refresh_lock:
            worker = server.model_service.refresh_workers['item']
        assert scored.wait(2), 'inference waited for feature refresh'
        for _ in range(10):
            server.model_service.schedule_feature_refresh('item')
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
    install_model(active, target="item")
    monkeypatch.setattr(server.Config.MODEL, 'FEATURE_REFRESH_SECONDS', 60)
    calls = []

    def fail(target):
        calls.append(target)
        raise ConnectionError('offline')

    monkeypatch.setattr(server.model_service, 'refresh_features', fail)
    server.model_service.schedule_feature_refresh('item')
    with server.model_service.refresh_lock:
        worker = server.model_service.refresh_workers.get('item')
    if worker:
        worker.join(5)
    server.model_service.schedule_feature_refresh('item')
    assert calls == ['item']
    assert server.model_service.current("item").snapshot is snapshot


def test_clean_during_refresh_does_not_reactivate_removed_model(monkeypatch):
    active = SimpleNamespace(dim=2, feature_snapshot={'feature_file': 'features.json'})
    install_model(active, target="item")
    activations = []

    def prepare(path):
        install_model(None, target="item")
        return {'dim': 2}

    monkeypatch.setattr(server.feature_service, 'prepare_all_features', prepare)
    monkeypatch.setattr(server.feature_service, 'activate', activations.append)
    server.model_service.refresh_features('item', force=True)
    assert activations == []
