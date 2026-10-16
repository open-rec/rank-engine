"""LightGBM binary inference using nvForest and the original split semantics."""

import threading

import numpy as np

from models.lightgbm.converter import convert_model


_ZERO_THRESHOLD = float(np.float32(1e-35))


class LightGBMGPUModel:
    def __init__(self, model, path, device):
        # CPU deployments do not import or initialize CUDA libraries.
        try:
            import cupy
            import nvforest
        except ImportError as error:
            raise RuntimeError(
                "LightGBM GPU inference requires requirements-cuda.txt"
            ) from error
        self.booster = model.booster
        self.dim = model.dim
        self.device = device
        self._cupy = cupy
        self._device_id = device.index if device.index is not None else 0
        self._lock = threading.Lock()
        converted, self._zero_features, self._sigmoid = convert_model(self.booster)
        with cupy.cuda.Device(self._device_id):
            self._forest = nvforest.load_from_treelite_model(
                converted, device="gpu", device_id=self._device_id, precision="double",
            )
        # Initialize the backend before publishing the new model.
        self.predict_proba(np.zeros((1, self.dim), dtype=np.float32))

    def predict_proba(self, features):
        with self._lock, self._cupy.cuda.Device(self._device_id):
            cp = self._cupy
            matrix = cp.asarray(features, dtype=cp.float64)
            if self._zero_features:
                extra = matrix[:, self._zero_features]
                # LightGBM IsZero uses an inclusive +/- 1e-35 interval.
                extra = cp.where(cp.abs(extra) <= _ZERO_THRESHOLD, cp.nan, extra)
                matrix = cp.concatenate((matrix, extra), axis=1)
            margins = cp.asnumpy(self._forest.predict(matrix))
        if margins.shape not in ((len(features),), (len(features), 1)):
            raise ValueError(f"Invalid LightGBM GPU margin shape: {margins.shape}")
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + np.exp(-self._sigmoid * margins.reshape(-1)))


class LightGBMPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, matrix):
        return self.model.predict_proba(matrix).reshape(-1).tolist()
