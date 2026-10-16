"""Torch checkpoint IO, device selection and batched execution."""

import torch

from config import Config


def select_device(model_type=None):
    configured = Config.MODEL.DEVICE
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MODEL_DEVICE requests CUDA but no CUDA device is available")
    return torch.device(configured)


def load_state(path, device):
    return torch.load(path, map_location=device)


def restore_model(model, state, device):
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


class TorchBackend:
    def __init__(self, model, forward=None):
        self.model = model
        self.forward = forward or (lambda model, matrix: model(matrix))

    def predict(self, matrix):
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            return self.forward(
                self.model, torch.from_numpy(matrix).to(device)
            ).reshape(-1).tolist()
