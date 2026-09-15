import pytest

import server


@pytest.fixture(autouse=True)
def reset_rank_engine_state(monkeypatch, tmp_path):
    """Isolate serving state and configuration between tests."""
    server.model = None
    server.model_info = None
    server.user_model = None
    server.user_model_info = None
    monkeypatch.setattr(server.Config.MODEL, "PATH", None)
    monkeypatch.setattr(server.Config.MODEL, "USER_PATH", None)
    monkeypatch.setattr(server.Config.MODEL, "ROOT", str(tmp_path))
    monkeypatch.setattr(
        server.Config.MODEL, "STATE_DIR", str(tmp_path / "active")
    )
    monkeypatch.setattr(server.Config.MODEL, "REQUIRED", False)
    server.feature_service.namespaces = {
        target: server.feature_service._empty_snapshot(target)
        for target in ("item", "user")
    }
    monkeypatch.setattr(server.Config.MODEL, "DEVICE", "cpu")
    yield
    server.model = None
    server.model_info = None
    server.user_model = None
    server.user_model_info = None
