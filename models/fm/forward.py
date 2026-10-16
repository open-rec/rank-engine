"""FM serving forward; an entry point for future fused interaction operators."""


def forward(model, features):
    return model(features)
