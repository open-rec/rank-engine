from algorithm.rank.lightgbm import LightGBMBinaryModel

from models.lightgbm.predictor import LightGBMPredictor, LightGBMGPUModel
from inference.runtime import BatchExecutor
from models.base import LoadedModel


class LightGBMDefinition:
    def load(self, info, dim, device):
        model = LightGBMBinaryModel.load(info.model)
        if model.booster.num_feature() != dim:
            raise ValueError("LightGBM model features do not match the feature dimension")
        model.dim = dim
        if device.type == "cuda":
            model = LightGBMGPUModel(model, info.model, device)
        return LoadedModel(model, LightGBMPredictor(model), {})

    def prepare(self, loaded, snapshot):
        return BatchExecutor(loaded.predictor)
