# OpenRec Rank Engine

[![CI](https://github.com/open-rec/rank-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/open-rec/rank-engine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C?logo=pytorch&logoColor=white)

Online ranking service for OpenRec. `rec-server`'s `rank` DAG node POSTs a user plus a candidate
item list here and gets a score per item back, which it adds to the recall scores.

FastAPI + PyTorch, listening on port 8123.

## how it fits in

```
rec-server ──POST /model/score──> rank-engine ──> LRModel (from rec-algorithm)
                                       │
                                       └──reads user:* / item:* features──> Redis
```

Ranking is optional. If this service is down, `rec-server` logs
`rank score failed with exception` and returns the recall order unranked — the request still
succeeds, so check the logs rather than assuming ranking is live.

## install

The model class (`LRModel`) and feature encoders come from the sibling `rec-algorithm` project.
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
bash start.sh            # uvicorn server:app --reload
```

All settings can be supplied through environment variables; local defaults still use Redis on
`localhost:6379` and listen on `0.0.0.0:8123`. Interactive docs: http://127.0.0.1:8123/docs

Features are loaded when a model is loaded, not while the Python module is imported. Redis is read
with incremental `SCAN` calls rather than the blocking `KEYS` command. The cache refreshes every
`FEATURE_REFRESH_SECONDS` (300 by default), and `/model/refresh-features` can force an immediate
refresh. If automatic loading starts before Redis has data, the first score request retries it.

## cluster mode

Start `bigdata-platform` first so its external Docker network and Redis service exist, then:

```shell
docker compose -f docker-compose.cluster.yml up -d --build
curl http://127.0.0.1:8123/health
```

The compose build starts from the official PyTorch 2.8 CUDA 12.9 runtime image, then installs the
repository requirements, including `torch==2.10.0`; 2.10.0 is therefore the application runtime
version in both the container and a direct host install. The image
uses the sibling `rec-algorithm` directory as a BuildKit additional context, joins
`openrec-bigdata`, reads Redis at `redis:6379`, mounts the sibling `model` repository read-only at
`/models`, and automatically loads the default LR checkpoint. The default deployment does not
require an NVIDIA runtime. To explicitly reserve all visible GPUs, add the repository-owned
override:

```shell
docker compose -f docker-compose.cluster.yml -f docker-compose.gpu.yml up -d --build
```

`MODEL_DEVICE=auto` selects CUDA when available and otherwise falls back to CPU. Keep one worker
unless each worker having its own model and feature cache is intentional.

Regional registries and Python package mirrors can be selected without editing repository files:

```shell
RANK_BASE_IMAGE=registry.example.com/pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime \
RANK_PIP_INDEX_URL=https://pypi.example.com/simple \
docker compose -f docker-compose.cluster.yml build rank-engine
```

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
| GET | `/health` | liveness |
| POST | `/model/load` | load a checkpoint into memory |
| POST | `/model/train` | train and evaluate one immutable release from Spark-prepared JSONL |
| POST | `/model/score` | score items for a user |
| POST | `/model/refresh-features` | rebuild the Redis-backed feature cache |
| POST | `/clean` | drop the loaded model and free CUDA cache |
| GET | `/` | static `index.html` |

In cluster mode `/model/train` is internal. It accepts a dataset below `/models/training`, writes
the checkpoint, FeatureSpace sidecar, metrics, and evaluation gate to
`/models/releases/{scene}/{version}`, then atomically exposes that immutable directory. Loading a
new release builds both its model and feature snapshot before changing the live scorer, so a failed
load leaves the previously active version usable.

The global catalog and LR/FM feature-set declarations are used only while training. A deployed
release is self-contained: rank-engine loads its own `lr.features.json` or `fm.features.json` and
does not consult those declarations. New manifests carry the fitted sidecar's SHA-256, input
dimension, catalog version and feature-set name; rec-console verifies the immutable file before
asking rank-engine to activate it. Legacy sidecars without this provenance remain loadable.
Training refuses to create a release when entity filtering leaves no labelled samples, when labels
contain only clicks or only exposures, or when held-out AUC is undefined; a zero threshold no longer
allows an untrained random checkpoint through the evaluation gate.

### load a model first

`/model/score` returns `MODEL_NOT_LOAD_YET` until a checkpoint is loaded — this is the step most
easily missed. From the OpenRec workspace root, use:

```shell
curl -X POST http://127.0.0.1:8123/model/load \
  -H 'Content-Type: application/json' \
  -d '{"type": "lr", "model": "model/rank/default/lr.pth", "feature": "model/feature/default/lr.features.json"}'
```

When running from the `rank-engine` directory, prefix both host paths with `../`. The cluster
Compose does not use these relative paths: it mounts the model repository at `/bootstrap-models`
and configures `/bootstrap-models/rank/default/lr.pth` plus the matching feature sidecar.

| Field | Default | Meaning |
|---|---|---|
| `type` | `lr` | registered model type: `lr` or `fm` |
| `model` | `lr.pth` | path to the `state_dict`, relative to the working directory |
| `dim` | `1024` | input feature width — **must** match what the checkpoint was trained with |
| `feature` | `null` | persisted feature-space JSON; inferred from the model path when possible |
| `factor_dim` | inferred | optional FM latent width; normally inferred from the checkpoint |

When `feature` is available, rank-engine encodes the already-materialized Redis user/item rows with
the exact training vocabulary and derives `dim` from it. `dim` remains only as a compatibility
fallback for legacy checkpoints without a sidecar.

For a legacy model without a feature sidecar, `dim` constructs `LRModel(dim)` before
`load_state_dict`, so a mismatch fails to load. The pre-trained Douban checkpoint in
[model](https://github.com/open-rec/model) uses 63.

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

LR and FM are implemented. Both consume the same persisted `FeatureSpace` vector; FM adds
second-order feature interactions without changing Redis materialization or `/model/score`.
`model_func_map` maps a `type` string to a class from `rec-algorithm`:

```python
model_func_map = {
    "lr": LRModel,
    "fm": FMModel,
}
```

FM's `factor_dim` defaults to 8 during training and is recorded in the manifest. Loading also
derives it from the `factors` tensor, so activation and rollback remain self-contained.

Train a checkpoint with `rec-algorithm`, or download the Douban one:

| Source | Type | Dim | Path |
|---|---|---|---|
| [model](https://github.com/open-rec/model) | LR | 63 | `model/rank/default/lr.pth` from the workspace root |
| `rec-algorithm` `test_lr.py::test_train` | LR | depends on the dataset | `rec-algorithm/model/lr.pth` |

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
| `MODEL_DEVICE` | `auto` | inference device: `auto`, `cuda`, `cuda:0`, or `cpu` |
| `FEATURE_REFRESH_SECONDS` | `300` | Redis feature cache refresh interval; `0` disables |

`rec-server` reaches this service via `rank.host` / `rank.port` in its
`application-cluster.properties`, defaulting to `127.0.0.1:8123`.

## test

Install the sibling algorithm package and test dependencies, then run the focused unit suite:

```shell
pip install -e ../rec-algorithm
pip install -r requirements-test.txt
pytest -q test
```

The tests use in-memory feature snapshots and temporary checkpoints; Redis and a running
rank-engine service are not required.
