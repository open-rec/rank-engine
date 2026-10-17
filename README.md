# OpenRec Rank Engine

[Release v0.1.0](RELEASE_NOTES.md) · [Changelog](CHANGELOG.md)

[![CI](https://github.com/open-rec/rank-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/open-rec/rank-engine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?logo=pytorch&logoColor=white)

Online ranking service for OpenRec. `rec-server`'s rank DAG nodes POST a source user plus candidate
items or users and receive a score per candidate, which is fused with recall scores.

FastAPI serving for PyTorch LR/FM and LightGBM, listening on port 8123.

## how it fits in

```
rec-server ──POST /model/score──> rank-engine ──> registered LR / FM / LightGBM predictor
                                       │
                                       └──reads user:* / item:* features──> Redis
```

Standalone deliberately bypasses ranking. In cluster, a rank-call failure may be caught by
rec-server and leave `rankScore` unset while recall/operation processing still returns candidates.
The cluster startup gate therefore checks rank-engine health and requires rank scores in its
recommendation smoke. `/health` on rec-server is liveness; `/ready` controls recommendation
admission. Check these gates and the returned scores rather than assuming nonempty results prove
ranking ran. Rank-engine remains a Python service; its callers and Spark jobs use Java 21.

## install

Training model classes (LR, FM and LightGBM) and feature encoders come from the sibling
`rec-algorithm` project; rank-engine owns their serving adapters.
Install it before rank-engine when running directly on the host:

```shell
cd rec-algorithm
pip install -r requirements.txt
pip install -e .
```

```shell
cd rank-engine
pip install -r requirements.txt
```

## start

```shell
bash start.sh            # uvicorn server:app; host, port, and workers are configurable
```

All settings can be supplied through environment variables; local defaults still use Redis on
`localhost:6379` and listen on `0.0.0.0:8123`. Interactive docs: http://127.0.0.1:8123/docs

Features are loaded when a model is loaded, not while the Python module is imported. Redis is read
with incremental `SCAN` calls rather than the blocking `KEYS` command. After `FEATURE_REFRESH_SECONDS` (300 by default), a scoring request schedules a background
refresh while continuing to use the last usable snapshot. Each target has at most one refresh
worker, and failed attempts are retried no more than once per interval.
`/model/refresh-features` remains synchronous and can force an immediate refresh. Item and User rank use independent `item` and `user` feature namespaces; each namespace
owns its source-user vectors, candidate vectors, FeatureSpace, dimension, and refresh timestamp, so
loading or refreshing one model cannot change the other's encoding. `/health` reports both under
`features.item` and `features.user`. If automatic loading starts before Redis has data, the first
score request retries it.

## cluster mode

Start `bigdata-platform` first so its external Docker network and Redis service exist, then:

```shell
docker compose -f docker-compose.cluster.yml up -d --build
curl http://127.0.0.1:8123/health
```

The compose build reuses the official PyTorch 2.10.0 CUDA 12.8 base image and upgrades
its Python runtime to `torch==2.10.0+cu129` with CUDA 12.9 libraries. nvForest and
PyTorch share this environment; no separate inference process is needed. The first
build downloads the replacement Torch/CUDA wheels as well as the tree inference libraries.
The image uses Ubuntu system Python; pip installation is explicitly enabled inside
the container with `PIP_BREAK_SYSTEM_PACKAGES=1`. It
uses the sibling `rec-algorithm` directory as a BuildKit additional context, joins
`openrec-bigdata`, reads Redis at `redis:6379`, mounts the sibling `model` directory read-only at
`/bootstrap-models`, and automatically loads the default LR checkpoint. The separate
`openrec-model-artifacts` volume is mounted at `/models` for retained releases and active state. The default deployment does not
require an NVIDIA runtime. To explicitly reserve all visible GPUs, add the repository-owned
override:

```shell
docker compose -f docker-compose.cluster.yml -f docker-compose.gpu.yml up -d --build
```

`MODEL_DEVICE=auto` selects CUDA when available and otherwise falls back to CPU. Keep one worker
unless each worker having its own model and feature cache is intentional.
LR/FM use batched CUDA inference when a GPU is visible to PyTorch, or CPU snapshot
projections otherwise. LightGBM uses nvForest GPU inference when CUDA is available,
and its native CPU predictor otherwise. GPU LightGBM uses double precision to preserve
tree thresholds and probability accuracy. The adapter preserves per-split missing-value
semantics (including `zero_as_missing`) and applies the original double-precision sigmoid
to GPU-computed margins, avoiding the importer's float32 sigmoid parameter. This does not guarantee bitwise identical
scores or a speedup for every batch size. `/model/load` and `/health` report the selected
device. GPU model initialization must succeed before activation; a failure retains the
previous model instead of silently running the new model on CPU.
An explicit `MODEL_DEVICE=cuda` still fails if CUDA is unavailable; use `auto` for fallback.
For GPU inference outside Docker, install `pip install -r requirements-cuda.txt` in a
Linux/Python 3.11+ environment with a CUDA 12.9 compatible driver. CPU-only installations
can continue using `requirements.txt`. GPU containers require the GPU Compose override
above to expose the host GPU. Standalone does not deploy rank-engine.

Regional registries and Python package mirrors can be selected without editing repository files:

```shell
RANK_BASE_IMAGE=registry.example.com/pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime \
RANK_PIP_INDEX_URL=https://pypi.example.com/simple \
docker compose -f docker-compose.cluster.yml build rank-engine
```

For a build using a complete local wheel cache, pass the Docker build arguments
`PIP_FIND_LINKS` (a wheel index reachable from the build) and `PIP_NO_INDEX=1`.
The build validates the base Torch 2.10.0 / CUDA 12.8 runtime before installing
`requirements-cuda.txt`, then runs `pip check` and verifies Torch 2.10.0+cu129,
LightGBM and nvForest. CUDA toolkit and Python binding versions are pinned to match
the Torch wheel and prevent transitive CUDA upgrades. The base image's
unused torchvision and torchaudio packages are removed to avoid unrelated dependency constraints.

For a host-side `rec-server`, use:

```properties
rank.open=true
rank.host=127.0.0.1
rank.port=8123
```

If `rec-server` also runs in the `openrec-bigdata` Docker network, use `rank.host=rank-engine`.

## api

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | readiness; HTTP 503 until Redis and required model feature snapshots are usable |
| GET | `/metrics` | HTTP and business-error counters, latency, and per-target model state |
| POST | `/model/load` | load a checkpoint into memory |
| POST | `/model/score` | score candidate items or users for a source user |
| POST | `/model/refresh-features` | rebuild the Redis-backed `item` and `user` feature caches |
| POST | `/clean` | drop the loaded model and free CUDA cache |
| GET | `/` | static `index.html` |

Rank-engine is an online inference service. Training belongs to rec-algorithm and runs
in the offline Spark job triggered by rec-console through Airflow. The former
`/model/train`, `/features`, and `/features/validate` endpoints have been removed;
feature discovery and selection validation use the rec-algorithm runner instead.
Deploy the companion rec-algorithm and rec-console changes together. Training can
complete while rank-engine is stopped and never activates a model automatically.

Loading a release builds both its model and feature snapshot before changing the
live scorer, so a failed load leaves the previously active version usable.

Every successful `/model/load` also atomically saves its load configuration to
`MODEL_STATE_DIR/{item,user}.json` before activating the model. The cluster default is
`/models/active` on the existing persistent model-artifact volume. Startup and lazy recovery restore
these records before considering bootstrap environment paths, including FM model type and factor
dimension. If a saved release is missing, corrupt, or cannot load, that target remains unavailable
and retries on its next score request; it never silently falls back to the bootstrap checkpoint.
Item and user recovery run independently. `/clean` unloads memory but retains the saved release,
which is restored on the next request or restart. Keep one worker for model lifecycle operations.

When upgrading an existing installation, re-publish its intended active versions through
rec-console once after deploying this change to create the recovery records. Earlier versions did
not write these records; the first startup without them still uses the configured bootstrap paths.
Standalone does not deploy rank-engine. Existing load/score response envelopes remain compatible.

`/health` reports diagnostic data and returns HTTP 200 when ready, or HTTP 503
when Redis is unavailable, an expected model is missing, or a feature snapshot is empty or has the
wrong dimension. Item ranking is required; user ranking is also required when configured, saved,
or loaded. Docker checks `/health`. This endpoint does not load a model or refresh features.
Callers that previously treated any HTTP 200 as liveness should now handle 503 during recovery.
The existing `openrec_rank_requests_total` counts HTTP status; business failures that retain HTTP
200 additionally increment `openrec_rank_business_errors_total{method,path,code}`. Monitor this
counter for nonzero rates, and `openrec_rank_models_loaded{target_type}` for item/user load state.
The legacy `openrec_rank_model_loaded` gauge continues to describe item ranking.

The global catalog describes shared business features. Offline training saves each
model's selected subset, fitted encoders, feature-definition fingerprints, training
parameters and checksums in an immutable release. Publishing loads that selection;
changing it requires retraining in rec-algorithm.

New sidecars validate only their selected definitions against the installed catalog,
so unrelated catalog additions do not invalidate them. Older sidecars retain their
original provenance validation. Deployment rejects selected columns with no
materialized online values. This is a presence check, not a freshness or coverage SLA.
Each scoring request uses a consistent model and encoded-feature snapshot, including
during concurrent publication or refresh. Failed refresh keeps the previous snapshot.

CPU LR/FM models with a fitted sidecar precompute static user, candidate and session
projections when loading or refreshing a snapshot. Requests combine these projections
with batch-encoded context/interaction features; FM preserves interactions across all
feature roles. The additional numeric cache uses one float32 per entity for LR and
`factor_dim + 2` for FM, plus entity lookup indexes. Publication and refresh rebuild the
cache before activation. CUDA, LightGBM and legacy models use batched feature matrices;
CPU PyTorch inputs share the assembled NumPy buffer. These CPU optimizations require
no additional dependency or checkpoint conversion. Floating-point reassociation can produce small
score differences from the original dense evaluation.

Offline sample validation, point-in-time joins, FeatureSpace fitting and evaluation
gates are documented in [rec-algorithm](../rec-algorithm/README.md). Rank-engine only
loads the resulting checkpoint and fitted sidecar for online scoring.

### load a model first

`/model/score` returns `MODEL_NOT_LOAD_YET` until a checkpoint is loaded — this is the step most
easily missed. From the OpenRec workspace root, use:

```shell
curl -X POST http://127.0.0.1:8123/model/load \
  -H 'Content-Type: application/json' \
  -d '{"type": "lr", "model": "model/rank/item/lr/lr.pth", "feature": "model/rank/item/lr/lr.features.json"}'
```

When running from the `rank-engine` directory, prefix both host paths with `../`. The cluster
Compose does not use these relative paths: it mounts the model repository at `/bootstrap-models`
and configures `/bootstrap-models/rank/item/lr/lr.pth` plus the matching feature sidecar. User ranking
uses `/bootstrap-models/rank/user/lr/lr.pth`; requests set `target_type=user` and use `candidate_ids`.

| Field | Default | Meaning |
|---|---|---|
| `type` | `lr` | registered model type: `lr`, `fm`, or `lightgbm` |
| `model` | `lr.pth` | path to the PyTorch `state_dict` or LightGBM text model |
| `dim` | `1024` | input feature width — **must** match what the checkpoint was trained with |
| `feature` | `null` | persisted feature-space JSON; inferred from the model path when possible |
| `factor_dim` | inferred | optional FM latent width; normally inferred from the checkpoint |

When `feature` is available, rank-engine encodes the already-materialized Redis user/item rows with
the exact training vocabulary and derives `dim` from it. `dim` remains only as a compatibility
fallback for legacy checkpoints without a sidecar.

For a legacy model without a feature sidecar, `dim` constructs `LRModel(dim)` before
`load_state_dict`, so a mismatch fails to load. The current generated bootstrap checkpoint in
[model](https://github.com/open-rec/model) includes a fitted sidecar and manifest; use their
recorded input dimension rather than a historical hard-coded width. The checked-in item LR
manifest currently records 423, which may change when the bundle is regenerated.

### score

```shell
curl -X POST http://127.0.0.1:8123/model/score \
  -H 'Content-Type: application/json' \
  -d '{"user_id": "test", "item_ids": ["5105858", "3785327", "123"]}'
```

```json
{
    "code": 0,
    "status": "success",
    "data": {
        "123": 0.0,
        "5105858": 0.6175433993339539,
        "3785327": 0.510399341583252
    },
    "message": ""
}
```

Items with no cached features score `0.0` rather than being dropped (`123` above). An unknown
`user_id` falls back to a zero user-feature vector, so the response is still well-formed.

## models

LR, FM, and LightGBM are implemented. All consume the same persisted `FeatureSpace` vector; FM adds
second-order feature interactions and LightGBM uses tree inference without changing Redis materialization or `/model/score`.
Content-enabled sidecars can select item title, subcategory, tags and content age. Raw item
profiles are normalized through rec-algorithm's shared content materializer during every feature
refresh: title-like fields use the fitted sidecar's fixed-width feature hash, while content age is
recomputed from `pubTime` at refresh time. New item IDs therefore remain scoreable from content
without rebuilding a vocabulary.
See [serving architecture and model extensions](docs/architecture.md) for lifecycle,
interface contracts and operator optimization guidance. Serving code is organized by responsibility:

```text
service/       feature_service, model_service, inference_service
models/        registry and per-model load/validation/forward implementations
  lr/          LR definition and forward entry point
  fm/          FM definition and forward entry point
  lightgbm/    LightGBM definition, CPU/GPU predictor and GPU converter
  common/      shared LR/FM static projection optimization
backends/      reusable execution backends (currently Torch for LR/FM)
inference/     lazy feature batches and versioned ModelRuntime
```

`models/registry.py` declares the supported model definitions. A definition implements
`load(info, dim, device)` and `prepare(loaded, snapshot)`. Loading returns the model,
predictor and model-specific metadata; preparation returns an executor with `score(batch)`.
To add a model, implement and register a definition without adding branches to the HTTP
or service layers. Existing LR/FM definitions reuse rec-algorithm's training classes and
checkpoint format; their `forward.py` files are the serving entry points for future
operator optimizations. A changed forward must also validate or disable static projections.
Reusable custom operators can be added under `operators/` when needed.

ModelService publishes the model, feature snapshot and prepared executor together as a
`ModelRuntime`. Refresh builds a replacement runtime without mutating an in-flight
request's version. Projection caches belong to the executor, not the feature snapshot.
InferenceService preserves candidate ordering and missing-candidate behavior; dense matrices
are assembled lazily so projected LR/FM inference does not pay for unused dense inputs.
LightGBM-specific inference and nvForest integration live entirely under `models/lightgbm/`.
`backends/` is reserved for execution frameworks shared by multiple model types.
CUDA imports remain lazy, keeping CPU installations usable.

FM's `factor_dim` defaults to 8 during training and is recorded in the manifest. Loading also
derives it from the `factors` tensor, so activation and rollback remain self-contained.

Train a checkpoint with `rec-algorithm`, or use the generated example-fixture bootstrap bundle:

| Source | Type | Dim | Path |
|---|---|---|---|
| [model](https://github.com/open-rec/model) | LR | from `lr.features.json` / `lr.manifest.json` (currently 423) | `model/rank/item/lr/lr.pth` from the workspace root |
| `rec-algorithm` `test_lr.py::test_train` | LR | depends on the fitted feature space | `{model_home}/rank/default/lr.pth` |

`model_home` uses `OPENREC_MODEL_HOME` first, then the sibling model repository when it has a
`rank/` directory, otherwise `rec-algorithm/model`. Every checkpoint must travel with its matching
`*.features.json`; the current default bundle is generated from `example/data/test`, not Douban.

## configuration

| Variable | Default | Meaning |
|---|---|---|
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` | `localhost`, `6379`, `0` | feature store |
| `REDIS_PASSWORD` | empty | optional Redis password |
| `REDIS_SOCKET_TIMEOUT` | `2` | connect/read timeout in seconds |
| `RANK_HOST`, `RANK_PORT` | `0.0.0.0`, `8123` | HTTP bind address |
| `RANK_WORKERS` | `1` | Uvicorn worker processes |
| `MODEL_TYPE` | `lr` | registered model type |
| `MODEL_PATH` | empty | checkpoint loaded at startup and retried lazily |
| `MODEL_FEATURE_PATH` | inferred | training feature-space sidecar |
| `MODEL_DIM` | `1024` | legacy checkpoint fallback dimension |
| `MODEL_REQUIRED` | `false` | fail startup when automatic loading fails |
| `MODEL_ROOT` | `/models` | model artifact root |
| `MODEL_STATE_DIR` | `${MODEL_ROOT}/active` | durable per-target recovery records; must be writable and survive restarts |
| `MODEL_DEVICE` | `auto` | inference device: `auto`, `cuda`, `cuda:0`, or `cpu` |
| `FEATURE_REFRESH_SECONDS` | `300` | Redis feature cache refresh interval; `0` disables |

`rec-server` reaches this service via `rank.host` / `rank.port` in its
`application-cluster.properties`, defaulting to `127.0.0.1:8123`.

## test

Install the sibling algorithm package and test dependencies, then run the focused unit suite:

```shell
pip install -e ../rec-algorithm
pip install -r requirements.txt -r requirements-test.txt
pytest -q test
```

The tests use in-memory feature snapshots and temporary checkpoints; Redis and a running
rank-engine service are not required. HTTP tests start a temporary local server.
GPU tests skip when CUDA is unavailable; to run them, install `requirements-cuda.txt`
and expose the GPU, then run the same suite. Coverage includes model registration,
versioned runtime refresh, CPU/GPU precision and the existing HTTP contracts.


### Session and request features

Fitted spaces can include `session`, `context`, and `interaction` roles in addition
to user/candidate vectors. `/model/score` accepts `session_id`, `context`, and
`candidate_contexts`; candidate-specific values override request values. Redis
`feature:session:*` snapshots use the same catalog compatibility check as user/item
snapshots, and session recency is recomputed on feature refresh. Missing sessions
use the existing cold-start zero vector.
An omitted, empty, or JSON `null` `session_id` means no session. The API normalizes
`null` to an empty string for compatibility with Java callers.

This serving capability does not imply that every feature is produced by
rec-server or available in cluster training. In particular, generic user-item
interaction values need an explicit candidate-context producer, and device or
subscription values must actually be supplied by the caller. The algorithm
runner advertises the narrower `training_models` capability set to rec-console.
Transformer experiment checkpoints are not supported by the serving lifecycle.


New producer snapshots also provide `recentEventStats` for refresh-time decay of
5m/1h/24h aggregates, 1/7/30-day conversion rates and commerce means. Legacy
snapshots retain their earlier limited refresh behavior until regenerated by
data-processor. Deploy and replay the producer alongside this consumer.
The LightGBM binary loader rejects non-binary objectives, including experiment
LambdaRank models whose raw scores are not click probabilities.
