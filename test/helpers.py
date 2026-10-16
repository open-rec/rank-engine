"""Build isolated serving runtimes for tests without loading checkpoint files."""

from dataclasses import replace

import server
from algorithm.rank.fm import FMModel
from backends.torch_backend import TorchBackend
from inference.runtime import BatchExecutor, ModelRuntime
from models.base import LoadedModel
from models.registry import MODEL_REGISTRY


def active_model(target="item"):
    runtime = server.model_service.current(target)
    return runtime.model if runtime is not None else None


def active_info(target="item"):
    runtime = server.model_service.current(target)
    return runtime.info if runtime is not None else None


def runtime_for(model, snapshot=None, kind=None, projected=False, target="item"):
    kind = kind or ("fm" if isinstance(model, FMModel) else "lr")
    snapshot = snapshot if snapshot is not None else getattr(model, "feature_snapshot", None)
    definition = MODEL_REGISTRY[kind]
    loaded = LoadedModel(model, TorchBackend(model), {})
    executor = (definition.prepare(loaded, snapshot) if projected
                else BatchExecutor(loaded.predictor))
    return ModelRuntime(loaded, definition, snapshot, executor,
                        {"type": kind, "target_type": target, "dim": getattr(model, "dim", 0)})


def install_model(model, target="item", **kwargs):
    if model is None:
        server.model_service.runtimes.pop(target, None)
        return None
    runtime = runtime_for(model, target=target, **kwargs)
    server.model_service.runtimes[target] = runtime
    return runtime


def set_info(info, target="item"):
    runtime = server.model_service.current(target)
    if runtime is not None:
        server.model_service.runtimes[target] = replace(runtime, info={**runtime.info, **(info or {})})
