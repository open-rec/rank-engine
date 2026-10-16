# Serving architecture and model extensions

Rank-engine owns online inference. Training definitions, feature encoders and checkpoint
contracts remain in rec-algorithm. The serving wrappers reuse those definitions rather
than maintaining a second copy of the training models.

## Code ownership

| Location | Responsibility |
|---|---|
| `server.py` | HTTP routes, response envelopes, exception mapping, health and metrics |
| `service/feature_service.py` | Redis feature loading, encoding and feature snapshots |
| `service/model_service.py` | Loading, validation, publication, recovery, refresh and unloading |
| `service/inference_service.py` | Pin a runtime for a request, prepare candidates and return scores |
| `models/registry.py` | Explicit registry of supported serving model types |
| `models/base.py` | `ModelDefinition`, `LoadedModel` and `Predictor` contracts |
| `models/lr/`, `models/fm/` | Model construction, checkpoint validation and serving forward entry points |
| `models/common/` | Shared LR/FM CPU projection preparation and scoring |
| `models/lightgbm/model.py` | LightGBM loading, validation and CPU/GPU predictor selection |
| `models/lightgbm/predictor.py` | Native CPU prediction and nvForest GPU execution |
| `models/lightgbm/converter.py` | GPU tree conversion preserving missing-value and sigmoid semantics |
| `backends/torch_backend.py` | Reusable Torch checkpoint helpers, device selection and batched execution |
| `inference/batch.py` | Candidate feature encoding and lazy dense-matrix assembly |
| `inference/runtime.py` | `ModelRuntime` and the batch/projection executor |

`backends/` holds execution frameworks that can serve multiple model types. Torch is
shared by LR and FM. The current LightGBM predictor is model-specific and belongs under
`models/lightgbm/`; there is no separate LightGBM backend directory or service.
Both implement the `Predictor` interface where dense prediction is needed.

## Model lifecycle and request flow

1. `ModelService.load` resolves a registered definition and prepares the feature snapshot.
   It validates the target and feature-space model type, selects the device, and asks the
   definition to load the checkpoint and prepare an executor.
2. The service persists activation state and publishes a `ModelRuntime` containing the
   loaded model, definition, feature snapshot, executor and public metadata. Preparation
   must finish before the new runtime replaces the active one.
3. `InferenceService.score` captures one runtime for the request. It retains that runtime
   even if a background refresh or another load publishes a replacement.
4. Feature batching preserves the original candidate positions when encoding request
   context. Unknown candidates receive `0.0`. The executor scores known candidates and
   the service maps the results back to candidate IDs.
5. Feature refresh prepares a new snapshot and executor, then replaces the runtime only
   if the active version still matches the one it started with. Existing requests can
   finish using their captured version. A failed preparation leaves that version usable.

Treat published snapshots and metadata as read-only. `ModelRuntime` is a frozen dataclass,
but its nested dictionaries and arrays are not deeply frozen. Do not mutate them after
publication. Projection caches belong to the executor; they are not stored in the feature
snapshot or attached to the training model.

## Adding a model

Create `models/<name>/model.py` with a definition implementing:

- `load(info, dim, device) -> LoadedModel`: load and validate weights, create a predictor,
  and return model-specific metadata. Preserve the artifact contract used by the producer.
- `prepare(loaded, snapshot)`: return an executor implementing `score(batch)`, ready for
  publication. A dense predictor can use `BatchExecutor(loaded.predictor)`.
- `Predictor.predict(matrix)`: return one score per input row in the same order. Existing
  models return positive-class probabilities, which callers use directly as ranking scores.

Register a definition instance in `MODEL_REGISTRY`. Generic model loading and scoring do
not need another model-type branch in `server.py` or the service classes. Registration
only enables online serving: a new model may also need training, release validation and
control-plane support in other repositories.

Use `TorchBackend` for compatible Torch models. Keep framework-independent or
model-specific predictors with their models. Introduce a new shared backend when adding
an execution framework that needs reusable infrastructure.

The current recovery record preserves the existing `Model` request fields, including FM's
`factor_dim`. If a new model needs additional load parameters, deliberately extend and
validate the request and recovery contracts; arbitrary metadata is not automatically
persisted as load configuration.

## Forward and operator optimization

LR and FM expose serving forward entry points in their respective `forward.py` files.
They currently delegate to the rec-algorithm model's forward method. These are places to
introduce model-specific execution changes without changing the training checkpoint.
Reusable custom or fused operators can be introduced under `operators/` when implemented;
that directory is not currently required.

CPU LR/FM may use static projections instead of calling the dense forward. Changing a
forward therefore also requires validating or disabling its projection optimization.
Exact model-type checks avoid replacing an arbitrary subclass's custom forward with the
standard LR/FM projection formula.

`InferenceBatch.matrix()` assembles dense input only when requested. Preserve this lazy
behavior: the projection executor consumes candidate IDs and dynamic feature blocks
without allocating the full dense batch. Rebuild projections when publishing new weights
or refreshing features.

LightGBM's GPU path preserves per-split missing-value semantics, including
`zero_as_missing`, and applies the original double-precision sigmoid to GPU-computed
margins. Keep these conversions with the model. GPU initialization errors fail activation;
they do not silently change the selected device to CPU. `MODEL_DEVICE=auto` selects CPU
when CUDA is unavailable. CPU deployments do not import CuPy or nvForest at startup.

## Compatibility and verification

This refactor preserves HTTP routes, request fields and defaults, response envelopes,
business error codes, health status behavior and checkpoint/recovery formats. In
particular, `item_ids` remains supported, nonempty `candidate_ids` takes precedence, and
`session_id: null` still normalizes to an empty string. Consumers use HTTP, not the old
internal `model.py` or `service/inference.py` module paths.

Run the full suite from rank-engine:

```shell
pip install -e ../rec-algorithm
pip install -r requirements.txt -r requirements-test.txt
python -m pytest -q test
```

For real GPU validation, install `requirements-cuda.txt` in the supported environment,
expose the GPU and run the same suite. GPU-specific tests skip without CUDA; when CUDA is
visible, missing GPU inference dependencies are a failure rather than a skipped test.

The suite covers model registration, dense/projection equivalence, publication and
refresh isolation, recovery, CPU/GPU device selection, LightGBM GPU precision and HTTP
load/score/refresh/clean behavior for both item and user targets. Add coverage for new
models at these boundaries, especially feature ordering and checkpoint compatibility.
Performance experiments should remain outside the repository, as separate local artifacts.
