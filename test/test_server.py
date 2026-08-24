import numpy as np
import pytest
import torch

from algorithm.rank.fm import FMModel
from algorithm.rank.lr import LRModel
from error_code import ErrorCode, ReException
from proto import Model, UserItems
import server


def snapshot(dim):
    return {"users": {"u1": np.array([1., 2.], dtype=np.float32)},
            "items": {"i1": np.array([3., 4.], dtype=np.float32)},
            "dim": dim, "feature_file": "features.json"}


def stub_feature_snapshot(monkeypatch, value):
    activated = []
    monkeypatch.setattr(server.feature_service, "prepare_all_features",
                        lambda feature_file: value)
    monkeypatch.setattr(server.feature_service, "activate", activated.append)
    return activated


@pytest.mark.parametrize("model_type, checkpoint", [
    ("lr", LRModel(dim=4)),
    ("fm", FMModel(dim=4, factor_dim=6)),
])
def test_load_model_activates_lr_and_fm_atomically(tmp_path, monkeypatch,
                                                   model_type, checkpoint):
    model_file = tmp_path / (model_type + ".pth")
    torch.save(checkpoint.state_dict(), model_file)
    prepared = snapshot(4)
    activated = stub_feature_snapshot(monkeypatch, prepared)

    result = server._load_model(Model(type=model_type, model=str(model_file),
                                      feature="features.json"))

    assert result["type"] == model_type
    assert result["dim"] == 4
    assert result["device"] == "cpu"
    assert activated == [prepared]
    assert server.model is not checkpoint
    assert not server.model.training
    if model_type == "fm":
        assert result["factor_dim"] == 6
        assert server.model.factor_dim == 6


def test_failed_fm_load_keeps_previous_model_and_feature_snapshot(tmp_path, monkeypatch):
    previous = LRModel(dim=4)
    server.model = previous
    server.model_info = {"type": "lr", "dim": 4}
    invalid = FMModel(dim=5, factor_dim=3)
    model_file = tmp_path / "fm.pth"
    torch.save(invalid.state_dict(), model_file)
    activated = stub_feature_snapshot(monkeypatch, snapshot(4))

    with pytest.raises(ValueError, match="factors do not match"):
        server._load_model(Model(type="fm", model=str(model_file),
                                 feature="features.json"))

    assert server.model is previous
    assert server.model_info == {"type": "lr", "dim": 4}
    assert activated == []


def test_load_model_rejects_explicit_wrong_fm_factor_dim(tmp_path, monkeypatch):
    checkpoint = FMModel(dim=4, factor_dim=6)
    model_file = tmp_path / "fm.pth"
    torch.save(checkpoint.state_dict(), model_file)
    activated = stub_feature_snapshot(monkeypatch, snapshot(4))

    with pytest.raises(RuntimeError, match="size mismatch"):
        server._load_model(Model(type="fm", model=str(model_file),
                                 feature="features.json", factor_dim=2))

    assert activated == []
    assert server.model is None


def test_load_endpoint_rejects_unknown_model_type():
    with pytest.raises(ReException) as error:
        server.load_model(Model(type="deepfm", model="unused.pth"))
    assert error.value.error_code is ErrorCode.INVALID_MODEL


def test_load_endpoint_maps_missing_checkpoint_to_model_not_found(monkeypatch):
    stub_feature_snapshot(monkeypatch, snapshot(4))
    with pytest.raises(ReException) as error:
        server.load_model(Model(type="lr", model="missing.pth", feature="features.json"))
    assert error.value.error_code is ErrorCode.MODEL_NOT_FOUND


def configure_scoring_features(monkeypatch, user, items):
    monkeypatch.setattr(server.feature_service, "refresh_if_stale", lambda seconds: None)
    monkeypatch.setattr(server.feature_service, "get_user_feature_by_id", lambda user_id: user)
    monkeypatch.setattr(server.feature_service, "get_item_feature_by_id", items.get)


def test_score_batches_known_items_and_degrades_missing_item(monkeypatch):
    scoring_model = LRModel(dim=4)
    with torch.no_grad():
        scoring_model.linear.weight.fill_(.1)
        scoring_model.linear.bias.zero_()
    scoring_model.eval()
    server.model = scoring_model
    configure_scoring_features(
        monkeypatch, np.array([1., 2.], dtype=np.float32),
        {"i1": np.array([3., 4.], dtype=np.float32)})

    result = server.score(UserItems(user_id="u1", item_ids=["i1", "missing"]))

    assert result["status"] == "success"
    assert result["data"]["i1"] == pytest.approx(torch.sigmoid(torch.tensor(1.)).item())
    assert result["data"]["missing"] == 0.0


def test_score_unknown_user_uses_zero_user_vector(monkeypatch):
    scoring_model = LRModel(dim=4)
    with torch.no_grad():
        scoring_model.linear.weight.copy_(torch.tensor([[10., 10., 1., 1.]]))
        scoring_model.linear.bias.zero_()
    server.model = scoring_model
    configure_scoring_features(
        monkeypatch, None, {"i1": np.array([1., 2.], dtype=np.float32)})

    result = server.score(UserItems(user_id="unknown", item_ids=["i1"]))

    assert result["data"]["i1"] == pytest.approx(torch.sigmoid(torch.tensor(3.)).item())


def test_score_dimension_mismatch_is_reported_as_inference_failure(monkeypatch):
    server.model = LRModel(dim=4)
    configure_scoring_features(
        monkeypatch, np.array([1.], dtype=np.float32),
        {"i1": np.array([2.], dtype=np.float32)})

    with pytest.raises(ReException) as error:
        server.score(UserItems(user_id="u1", item_ids=["i1"]))

    assert error.value.error_code is ErrorCode.INFERENCE_FAILED


def test_score_without_loaded_model_is_rejected():
    with pytest.raises(ReException) as error:
        server.score(UserItems(user_id="u1", item_ids=["i1"]))
    assert error.value.error_code is ErrorCode.MODEL_NOT_LOAD_YET


def test_empty_item_list_does_not_touch_feature_store(monkeypatch):
    server.model = LRModel(dim=4)
    monkeypatch.setattr(
        server.feature_service, "refresh_if_stale",
        lambda seconds: pytest.fail("feature store should not be touched"))
    assert server.score(UserItems(user_id="u1", item_ids=[]))["data"] == {}
