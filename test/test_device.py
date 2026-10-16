import numpy as np
import pytest
import torch

from algorithm.feature.feature_space import ColumnSpec, FeatureSpace, NUM
from algorithm.rank.fm import FMModel
from algorithm.rank.lr import LRModel
from proto import Model, UserItems
import server


@pytest.mark.parametrize("available,expected", [(False, "cpu"), (True, "cuda")])
@pytest.mark.parametrize("kind", ["lr", "fm"])
def test_auto_selects_available_device(monkeypatch, available, expected, kind):
    monkeypatch.setattr(server.Config.MODEL, "DEVICE", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert server.model_device(kind) == torch.device(expected)


def test_explicit_cpu_overrides_available_gpu(monkeypatch):
    monkeypatch.setattr(server.Config.MODEL, "DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert server.model_device("fm") == torch.device("cpu")


def test_explicit_cuda_still_reports_missing_gpu(monkeypatch):
    monkeypatch.setattr(server.Config.MODEL, "DEVICE", "cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CUDA device"):
        server.model_device("lr")


@pytest.mark.parametrize("configured", ["auto", "cpu", "cuda:0"])
def test_lightgbm_does_not_require_cuda(monkeypatch, configured):
    monkeypatch.setattr(server.Config.MODEL, "DEVICE", configured)
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: pytest.fail("CPU tree inference must not probe CUDA"))
    assert server.model_device("lightgbm") == torch.device("cpu")


@pytest.mark.parametrize("kind,model_class", [("lr", LRModel), ("fm", FMModel)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_auto_load_and_score_on_selected_device(
    monkeypatch, tmp_path, kind, model_class, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a visible CUDA GPU")
    monkeypatch.setattr(server.Config.MODEL, "DEVICE", "auto")
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    space = FeatureSpace(
        user_columns=[ColumnSpec("age", NUM)],
        item_columns=[ColumnSpec("weight", NUM)],
    )
    space.fitted = True
    snapshot = {
        "space": space, "dim": 2, "users": {"u": np.array([1.0])},
        "items": {"i": np.array([2.0])}, "sessions": {},
    }
    monkeypatch.setattr(server.feature_service, "prepare_all_features", lambda path: snapshot)
    checkpoint = model_class(2).eval()
    path = tmp_path / (kind + ".pth")
    torch.save(checkpoint.state_dict(), path)
    info = server._load_model(Model(type=kind, model=str(path), feature="features.json"))
    assert info["device"] == device
    assert next(server.model.parameters()).device.type == device
    assert ("scorer" in snapshot) == (device == "cpu")
    with torch.inference_mode():
        expected = checkpoint(torch.tensor([[1.0, 2.0]])).item()
    actual = server.score(UserItems(user_id="u", item_ids=["i"]))["data"]["i"]
    assert actual == pytest.approx(expected, abs=1e-6)
