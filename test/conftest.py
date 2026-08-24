import pytest

import server


@pytest.fixture(autouse=True)
def reset_rank_engine_state(monkeypatch):
    """Keep module-level serving state and configuration isolated between tests."""
    server.model = None
    server.model_info = None
    monkeypatch.setattr(server.Config.MODEL, "PATH", None)
    monkeypatch.setattr(server.Config.MODEL, "DEVICE", "cpu")
    yield
    server.model = None
    server.model_info = None
