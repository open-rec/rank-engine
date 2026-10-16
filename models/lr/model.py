from algorithm.rank.lr import LRModel

from backends.torch_backend import TorchBackend, load_state, restore_model
from models.base import LoadedModel
from models.common.projection import prepare_torch
from models.lr.forward import forward


class LRDefinition:
    def load(self, info, dim, device):
        model = restore_model(LRModel(dim), load_state(info.model, device), device)
        return LoadedModel(model, TorchBackend(model, forward), {})

    def prepare(self, loaded, snapshot):
        return prepare_torch(loaded, snapshot, LRModel)
