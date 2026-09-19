import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch
import uvicorn

from algorithm.rank.fm import FMModel
from algorithm.rank.lr import LRModel
from proto import Model, UserItems
import server


@pytest.fixture
def client():
    """Exercise HTTP with Uvicorn and a standard-library client."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    runtime = uvicorn.Server(
        uvicorn.Config(server.app, lifespan="off", log_level="critical")
    )
    thread = threading.Thread(
        target=runtime.run, kwargs={"sockets": [sock]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not runtime.started:
            assert thread.is_alive() and time.monotonic() < deadline, (
                "HTTP server failed to start"
            )
            time.sleep(0.01)

        def request(path, body=None):
            data = json.dumps(body).encode() if body is not None else None
            req = Request(
                "http://127.0.0.1:%s%s" % (sock.getsockname()[1], path),
                data=data,
                headers={"Content-Type": "application/json"},
            )
            try:
                response = urlopen(req, timeout=5)
            except HTTPError as error:
                response = error
            with response:
                text = response.read().decode()
                return SimpleNamespace(
                    status_code=response.status,
                    text=text,
                    json=lambda: json.loads(text),
                )

        yield SimpleNamespace(
            get=request, post=lambda path, json: request(path, json)
        )
    finally:
        runtime.should_exit = True
        thread.join(timeout=10)
        sock.close()


def checkpoint(
    tmp_path, monkeypatch, target="item", kind="lr", name="published"
):
    path = tmp_path / (name + ".pth")
    model = FMModel(4, factor_dim=3) if kind == "fm" else LRModel(4)
    torch.save(model.state_dict(), path)
    monkeypatch.setattr(
        server.feature_service,
        "prepare_all_features",
        lambda _: {
            "target_type": target,
            "model_type": kind,
            "dim": 4,
            "users": {"u": np.ones(2)},
            "items": {"c": np.ones(2)},
            "feature_file": str(tmp_path / "features.json"),
        },
    )
    return Model(
        model=str(path), type=kind, feature=str(tmp_path / "features.json")
    )


@pytest.mark.parametrize("target", ["item", "user"])
@pytest.mark.parametrize("kind", ["lr", "fm"])
def test_restart_restores_last_publication_and_rollback(
    tmp_path, monkeypatch, target, kind
):
    info = checkpoint(tmp_path, monkeypatch, target, kind)
    server.load_model(info)
    config_path = "USER_PATH" if target == "user" else "PATH"
    monkeypatch.setattr(
        server.Config.MODEL, config_path, "/missing/bootstrap.pth"
    )
    server.clean()
    server.startup()
    active = server.user_model_info if target == "user" else server.model_info
    assert active["path"] == info.model
    assert active["type"] == kind
    rolled_back = checkpoint(tmp_path, monkeypatch, target, kind, "rollback")
    server.load_model(rolled_back)
    server.clean()
    server.startup()
    active = server.user_model_info if target == "user" else server.model_info
    assert active["path"] == rolled_back.model


def test_failed_load_preserves_persisted_release(tmp_path, monkeypatch):
    info = checkpoint(tmp_path, monkeypatch)
    server.load_model(info)
    previous = server._state_path("item").read_bytes()
    with pytest.raises(Exception):
        server.load_model(Model(model="missing.pth", feature=info.feature))
    assert server._state_path("item").read_bytes() == previous
    assert server.model_info["path"] == info.model


def test_failed_state_write_keeps_old_model(tmp_path, monkeypatch):
    info = checkpoint(tmp_path, monkeypatch)
    server.load_model(info)
    previous = server.model

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(server, "_persist_model", fail)
    with pytest.raises(Exception):
        server.load_model(info)
    assert server.model is previous


def test_corrupt_state_falls_back_to_bootstrap(tmp_path, monkeypatch):
    info = checkpoint(tmp_path, monkeypatch)
    monkeypatch.setattr(server.Config.MODEL, "PATH", info.model)
    monkeypatch.setattr(server.Config.MODEL, "FEATURE_PATH", info.feature)
    path = server._state_path("item")
    path.parent.mkdir()
    path.write_text("broken json")
    server.startup()
    assert server.model is not None
    assert server.model_info["path"] == info.model
    saved = json.loads(path.read_text())
    assert saved["load"]["model"] == str(Path(info.model).resolve())


def test_lazy_user_load_works_without_item_configuration(
    tmp_path, monkeypatch
):
    info = checkpoint(tmp_path, monkeypatch, "user")
    monkeypatch.setattr(server.Config.MODEL, "USER_PATH", info.model)
    monkeypatch.setattr(server.Config.MODEL, "USER_FEATURE_PATH", info.feature)
    result = server.score(
        UserItems(user_id="u", candidate_ids=["c"], target_type="user")
    )
    assert result["status"] == "success"
    assert 0 < result["data"]["c"] < 1
    assert server.model is None
    assert server.user_model_info["target_type"] == "user"


def test_startup_item_failure_does_not_skip_user(tmp_path, monkeypatch):
    info = checkpoint(tmp_path, monkeypatch, "user")
    monkeypatch.setattr(server.Config.MODEL, "PATH", "missing.pth")
    monkeypatch.setattr(server.Config.MODEL, "USER_PATH", info.model)
    monkeypatch.setattr(server.Config.MODEL, "USER_FEATURE_PATH", info.feature)
    server.startup()
    assert server.model is None
    assert server.user_model is not None


def test_lazy_restore_retries_published_model_after_transient_failure(
    tmp_path, monkeypatch
):
    info = checkpoint(tmp_path, monkeypatch)
    server.load_model(info)
    server.clean()
    original = server.feature_service.prepare_all_features

    def fail(_):
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr(server.feature_service, "prepare_all_features", fail)
    server.startup()
    assert server.model is None
    monkeypatch.setattr(
        server.feature_service, "prepare_all_features", original
    )
    assert (
        server.score(UserItems(user_id="u", item_ids=["c"]))["status"]
        == "success"
    )
    assert server.model_info["path"] == info.model


def test_health_readiness_and_both_targets(tmp_path, monkeypatch, client):
    monkeypatch.setattr(
        server, "get_redis_client", lambda: SimpleNamespace(ping=lambda: True)
    )
    assert client.get("/health").status_code == 503
    server.load_model(checkpoint(tmp_path, monkeypatch))
    assert client.get("/health").status_code == 200
    monkeypatch.setattr(server.Config.MODEL, "USER_PATH", "user.pth")
    assert client.get("/health").status_code == 503
    server.load_model(checkpoint(tmp_path, monkeypatch, "user", name="user"))
    assert client.get("/health").status_code == 200
    server.feature_service.namespaces["user"]["items"] = {}
    assert client.get("/health").status_code == 503
    monkeypatch.setattr(
        server, "get_redis_client", lambda: SimpleNamespace(ping=lambda: False)
    )
    assert client.get("/health").json()["data"]["ready"] is False


def test_http_200_business_failures_are_counted(monkeypatch, client):
    counter = server.business_errors.labels("POST", "/model/score", "1002")
    before = counter._value.get()
    result = client.post(
        "/model/score", json={"user_id": "u", "item_ids": ["c"]}
    )
    assert result.status_code == 200
    assert result.json()["code"] == 1002
    assert counter._value.get() == before + 1
    # Unexpected failures also retain the envelope and have a distinct business
    # code.
    monkeypatch.setattr(server.feature_service, "stats", lambda: 1 / 0)
    counter = server.business_errors.labels("GET", "/health", "1000")
    before = counter._value.get()
    result = client.get("/health")
    assert result.json()["code"] == 1000
    assert counter._value.get() == before + 1
    assert client.get("/health").status_code == 503


def test_metrics_report_each_target(client):
    server.user_model = LRModel(4)
    body = client.get("/metrics").text
    assert 'openrec_rank_models_loaded{target_type="user"} 1.0' in body
    assert 'openrec_rank_models_loaded{target_type="item"} 0.0' in body


def test_online_service_does_not_expose_training(client):
    assert client.post("/model/train", json={}).status_code == 404
    assert client.get("/features").status_code == 404
    assert client.post("/features/validate", json={}).status_code == 404
