from algorithm.rank.fm import FMModel

from backends.torch_backend import TorchBackend, load_state, restore_model
from models.base import LoadedModel
from models.common.projection import prepare_torch
from models.fm.forward import forward


class FMDefinition:
    def load(self, info, dim, device):
        state = load_state(info.model, device)
        factors = state.get("factors")
        if factors is None or factors.ndim != 2 or factors.shape[0] != dim:
            raise ValueError("FM checkpoint factors do not match the feature dimension")
        model = restore_model(
            FMModel(dim, factor_dim=info.factor_dim or factors.shape[1]), state, device,
        )
        return LoadedModel(model, TorchBackend(model, forward),
                           {"factor_dim": model.factor_dim})

    def prepare(self, loaded, snapshot):
        return prepare_torch(loaded, snapshot, FMModel)
