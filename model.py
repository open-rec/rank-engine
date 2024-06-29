import torch
from torch import nn


class LRModel(nn.Module):
    def __init__(self, dim=10):
        super().__init__()
        self.linear = nn.Linear(in_features=dim, out_features=1)

    def forward(self, x):
        pred = torch.sigmoid(self.linear(x))
        return pred


model_func_map = {
    "lr": LRModel
}