# rank-engine

Online ranking service for open-rec. `rec-server`'s `rank` DAG node POSTs a user plus a candidate
item list here and gets a score per item back, which it adds to the recall scores.

FastAPI + PyTorch, listening on port 8000.

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
`localhost:6379` and listen on `0.0.0.0:8000`. Interactive docs: http://127.0.0.1:8000/docs

Features are loaded when a model is loaded, not while the Python module is imported. Redis is read
with incremental `SCAN` calls rather than the blocking `KEYS` command. The cache refreshes every
`FEATURE_REFRESH_SECONDS` (300 by default), and `/model/refresh-features` can force an immediate
refresh. If automatic loading starts before Redis has data, the first score request retries it.

## cluster mode

Start `bigdata-platform` first so its external Docker network and Redis service exist, then:

```shell
docker compose -f docker-compose.cluster.yml up -d --build
curl http://127.0.0.1:8000/health
```

The compose build uses the PyTorch 2.8 GPU base image with CUDA 12.9 and installs
all remaining pip packages from the Alibaba Cloud mirror. A direct host install uses the same mirror
and installs the CUDA-enabled `torch==2.10.0` package. The image
uses the sibling `rec-algorithm` directory as a BuildKit additional context, joins
`openrec-bigdata`, reads Redis at `redis:6379`, mounts the sibling `model` repository read-only at
`/models`, requests all visible NVIDIA GPUs, and automatically loads the default LR checkpoint.
`MODEL_DEVICE=auto` selects CUDA when available and otherwise falls back to CPU. Keep one worker
unless each worker having its own model and feature cache is intentional.

For a host-side `rec-server`, use:

```properties
rank.open=true
rank.host=127.0.0.1
rank.port=8000
```

If `rec-server` also runs in the `openrec-bigdata` Docker network, use `rank.host=rank-engine`.

## api

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| POST | `/model/load` | load a checkpoint into memory |
| POST | `/model/score` | score items for a user |
| POST | `/model/refresh-features` | rebuild the Redis-backed feature cache |
| POST | `/clean` | drop the loaded model and free CUDA cache |
| GET | `/` | static `index.html` |

### load a model first

`/model/score` returns `MODEL_NOT_LOAD_YET` until a checkpoint is loaded — this is the step most
easily missed:

```shell
curl -X POST http://127.0.0.1:8000/model/load \
  -H 'Content-Type: application/json' \
  -d '{"type": "lr", "model": "model/rank/default/lr.pth", "feature": "model/feature/default/lr.features.json"}'
```

| Field | Default | Meaning |
|---|---|---|
| `type` | `lr` | key into `model_func_map`; only `lr` is implemented |
| `model` | `lr.pth` | path to the `state_dict`, relative to the working directory |
| `dim` | `1024` | input feature width — **must** match what the checkpoint was trained with |
| `feature` | `null` | persisted feature-space JSON; inferred from the model path when possible |

When `feature` is available, rank-engine encodes the already-materialized Redis user/item rows with
the exact training vocabulary and derives `dim` from it. `dim` remains only as a compatibility
fallback for legacy checkpoints without a sidecar.

For a legacy model without a feature sidecar, `dim` constructs `LRModel(dim)` before
`load_state_dict`, so a mismatch fails to load. The pre-trained Douban checkpoint in
[model](https://github.com/open-rec/model) uses 63.

### score

```shell
curl -X POST http://127.0.0.1:8000/model/score \
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

Only LR is implemented. `model_func_map` in `model.py` maps a `type` string to a class from
`rec-algorithm`:

```python
model_func_map = {
    "lr": LRModel,
}
```

To add one, implement it in `rec-algorithm` (`algorithm/rank/`) and register it here.

Train a checkpoint with `rec-algorithm`, or download the Douban one:

| Source | Type | Dim | Path |
|---|---|---|---|
| [model](https://github.com/open-rec/model) | LR | 63 | `rank/lr.pth` |
| `rec-algorithm` `test_lr.py::test_train` | LR | depends on the dataset | `rec-algorithm/model/lr.pth` |

## configuration

| Variable | Default | Meaning |
|---|---|---|
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` | `localhost`, `6379`, `0` | feature store |
| `REDIS_PASSWORD` | empty | optional Redis password |
| `REDIS_SOCKET_TIMEOUT` | `2` | connect/read timeout in seconds |
| `RANK_HOST`, `RANK_PORT` | `0.0.0.0`, `8000` | HTTP bind address |
| `RANK_WORKERS` | `1` | Uvicorn worker processes |
| `MODEL_TYPE` | `lr` | registered model type |
| `MODEL_PATH` | empty | checkpoint loaded at startup and retried lazily |
| `MODEL_FEATURE_PATH` | inferred | training feature-space sidecar |
| `MODEL_DIM` | `1024` | legacy checkpoint fallback dimension |
| `MODEL_REQUIRED` | `false` | fail startup when automatic loading fails |
| `MODEL_DEVICE` | `auto` | inference device: `auto`, `cuda`, `cuda:0`, or `cpu` |
| `FEATURE_REFRESH_SECONDS` | `300` | Redis feature cache refresh interval; `0` disables |

`rec-server` reaches this service via `rank.host` / `rank.port` in its
`application-cluster.properties`, defaulting to `127.0.0.1:8000`.
