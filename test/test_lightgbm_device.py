from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from proto import Model, UserItems
from service.lightgbm_inference import LightGBMGPUModel
import server


def test_gpu_initialization_failure_does_not_publish(monkeypatch, tmp_path):
    from algorithm.rank.lightgbm import LightGBMBinaryModel
    model = LightGBMBinaryModel(n_estimators=2, min_child_samples=1).fit(
        np.array([[0, 0], [1, 1]], dtype=np.float32), np.array([0, 1])
    )
    path = tmp_path / "model.txt"
    model.save(path)
    monkeypatch.setattr(server.feature_service, "prepare_all_features", lambda path: {
        "dim": 2, "users": {"u": np.zeros(1)}, "items": {"i": np.zeros(1)},
    })
    previous = object()
    monkeypatch.setattr(server, "model", previous)
    monkeypatch.setattr(server, "model_device", lambda kind: torch.device("cuda"))
    def fail(*args):
        raise RuntimeError("GPU backend unavailable")
    monkeypatch.setattr(server, "LightGBMGPUModel", fail)
    with pytest.raises(RuntimeError, match="GPU backend unavailable"):
        server._load_model(Model(type="lightgbm", model=str(path), feature="features.json"))
    assert server.model is previous
    assert not server._state_path("item").exists()


@pytest.mark.parametrize("boosting", ["gbdt", "rf"])
@pytest.mark.parametrize("zero_as_missing", [False, True])
@pytest.mark.parametrize("categorical", [False, True])
def test_real_gpu_lightgbm_matches_cpu(monkeypatch, tmp_path, zero_as_missing, categorical, boosting):
    if not torch.cuda.is_available():
        pytest.skip("requires a visible CUDA GPU and requirements-cuda.txt")
    import lightgbm as lgb
    # Do not skip a missing nvForest installation on GPU: the image must include it.
    import nvforest  # noqa: F401
    rng = np.random.default_rng(23)
    features = rng.normal(size=(256, 4)).astype(np.float32)
    if categorical:
        features[:, 3] = rng.integers(0, 5, size=len(features))
    labels = (features[:, 0] + features[:, 1] + (features[:, 3] == 2) > 0).astype(int)
    features[::7, 0] = np.nan
    features[::5, 1] = 0
    data = lgb.Dataset(features, label=labels, categorical_feature=[3] if categorical else [])
    booster = lgb.train({"objective": "binary", "sigmoid": 1.7, "num_leaves": 7,
                         "min_data_in_leaf": 2, "num_threads": 1, "verbosity": -1,
                         "zero_as_missing": zero_as_missing, "boosting": boosting,
                         "bagging_fraction": .8, "bagging_freq": 1}, data, num_boost_round=12)
    path = tmp_path / "model.txt"
    booster.save_model(path)
    monkeypatch.setattr(server.Config.MODEL, "DEVICE", "auto")
    snapshot = {"dim": 4, "users": {"u": np.zeros(0)},
                "items": {str(i): row for i, row in enumerate(features)}, "sessions": {}}
    monkeypatch.setattr(server.feature_service, "prepare_all_features", lambda path: snapshot)
    info = server._load_model(Model(type="lightgbm", model=str(path), feature="features.json"))
    assert info["device"] == "cuda"
    assert isinstance(server.model, LightGBMGPUModel)
    actual = server.model.predict_proba(features)
    expected = booster.predict(features)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    scores = server.score(UserItems(user_id="u", item_ids=["0", "1", "missing"]))["data"]
    assert scores == pytest.approx({"0": expected[0], "1": expected[1], "missing": 0}, abs=1e-12)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(server.model.predict_proba, [features] * 8))
    for result in results:
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
    # A persisted release must choose the device again when restored.
    server.model = None
    server._restore_model("item")
    assert isinstance(server.model, LightGBMGPUModel)
    np.testing.assert_allclose(server.model.predict_proba(features), expected,
                               rtol=0, atol=1e-12)

    probes = rng.normal(size=(300, 4)).astype(np.float64)
    boundary = float(np.float32(1e-35))
    probes[:100, 0] = np.resize([0, -0.0, np.nan, boundary, -boundary,
                               np.nextafter(boundary, np.inf),
                               np.nextafter(-boundary, -np.inf)], 100)
    probes[100:200, 1] = np.nan
    probes[200:, 2] = np.nan  # Missing values unseen in this training feature.
    if categorical:
        probes[:, 3] = np.resize([0, 1, 2, 3, 4, 99, -1, np.nan, -.5, 1.9], 300)
    np.testing.assert_allclose(server.model.predict_proba(probes), booster.predict(probes),
                               rtol=0, atol=1e-12)
