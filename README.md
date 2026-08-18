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

`requirements.txt` pins `rec-algorithm==0.0.1`; the model class (`LRModel`) and the feature encoders
come from that package, so build its wheel first:

```shell
git clone https://github.com/open-rec/rec-algorithm.git
cd rec-algorithm
pip install -r requirements.txt
bash package.sh
pip install dist/rec_algorithm-0.0.1-*.whl
```

```shell
cd rank-engine
pip install -r requirements.txt
```

## start

```shell
bash start.sh            # uvicorn server:app --reload
```

Host and port come from `config.py` (`ServerConfig`), Redis from `RedisConfig` — both default to
localhost. Interactive docs: http://127.0.0.1:8000/docs

**Redis must be populated before startup.** `FeatureService` is a singleton that runs at import time:
it scans every `user:*` and `item:*` key, builds one-hot / scaled features with `rec-algorithm`'s
`UserFeature` and `ItemFeature`, and caches them in memory. Starting against an empty Redis yields
empty feature maps, and new data pushed later is not picked up until restart.

## api

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| POST | `/model/load` | load a checkpoint into memory |
| POST | `/model/score` | score items for a user |
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

`config.py`, edited in place — there is no env-var override:

```python
class RedisConfig:  HOST = "localhost"; PORT = 6379; DB = 0
class ServerConfig: HOST = "0.0.0.0";   PORT = 8000
```

`rec-server` reaches this service via `rank.host` / `rank.port` in its
`application-{dev,prod}.properties`, defaulting to `127.0.0.1:8000`.
