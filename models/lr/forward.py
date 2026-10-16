"""LR serving forward; retain the training implementation as the default."""


def forward(model, features):
    return model(features)
