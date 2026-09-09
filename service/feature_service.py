import json
import logging
import threading
import time

import numpy as np
import pandas as pd
from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.user_feature import UserFeature
from algorithm.feature.feature_space import FeatureSpace

from sugar import singleton
from util.redis_util import get_redis_client


@singleton
class FeatureService(object):

    NAMESPACES = ("item", "user")

    def __init__(self):
        self.namespaces = {name: self._empty_snapshot(name) for name in self.NAMESPACES}
        self.lock = threading.RLock()

    @staticmethod
    def _empty_snapshot(namespace):
        return {"users": {}, "items": {}, "dim": 0, "feature_file": None,
                "feature_set": None, "catalog_version": None, "model_type": None,
                "target_type": namespace, "loaded_at": 0}

    def load_all_features(self, feature_file=None, namespace=None):
        if feature_file:
            snapshot = self.prepare_all_features(feature_file)
            self.activate(snapshot)
            return
        namespaces = [namespace] if namespace else list(self.NAMESPACES)
        for name in namespaces:
            with self.lock:
                active_file = self.namespaces[name]["feature_file"]
            if active_file:
                self.activate(self.prepare_all_features(active_file))

    def prepare_all_features(self, feature_file=None):
        """Build a feature snapshot without changing the live scorer."""
        user_feature = self.load_user_feature()
        item_feature = self.load_item_feature()

        space = FeatureSpace.load(feature_file) if feature_file else None
        target_type = space.target_type if space else "item"
        if user_feature.users.empty or (target_type == "item" and item_feature.items.empty):
            return {"users": {}, "items": {}, "dim": 0,
                    "feature_file": feature_file, "feature_set": None,
                    "catalog_version": None, "model_type": None,
                    "target_type": target_type}

        if space:
            candidates = item_feature.items if target_type == "item" else user_feature.users
            user_map, item_map = space.build_maps(user_feature.users, candidates)
            return {"users": user_map, "items": item_map, "dim": space.dim,
                    "feature_file": feature_file, "feature_set": space.feature_set,
                    "catalog_version": space.catalog_version, "model_type": space.model_type,
                    "target_type": target_type}

        user_features = np.hstack([
            user_feature.country,
            user_feature.city,
            user_feature.gender,
            user_feature.age,
            user_feature.tags
        ])

        item_features = np.hstack([
            item_feature.category,
            item_feature.scene,
            item_feature.weight,
        ])

        user_map = {
            user_id: user_features[i]
            for i, user_id in enumerate(user_feature.raw_id)
        }

        item_map = {
            item_id: item_features[i]
            for i, item_id in enumerate(item_feature.raw_id)
        }
        return {"users": user_map, "items": item_map,
                "dim": user_features.shape[1] + item_features.shape[1],
                "feature_file": feature_file, "feature_set": None,
                "catalog_version": None, "model_type": None, "target_type": "item"}

    def activate(self, snapshot):
        namespace = snapshot.get("target_type", "item")
        if namespace not in self.NAMESPACES:
            raise ValueError("unsupported feature namespace: %s" % namespace)
        with self.lock:
            activated = dict(snapshot)
            activated["loaded_at"] = time.monotonic()
            self.namespaces[namespace] = activated

    @staticmethod
    def _batch_load(key_pattern="*", batch_size=500):
        redis_client = get_redis_client()
        key_values = {}
        batch_keys = []

        def update_key_values(keys):
            values = redis_client.batch_get_values(keys)
            for key, value in zip(keys, values):
                try:
                    key_values[key] = json.loads(value.decode("utf-8"))
                except Exception as e:
                    logging.warning(f"load key:{key}, value:{value} failed")
                    continue
        for key in redis_client.scan_iter(key_pattern, count=batch_size):
            batch_keys.append(key.decode("utf-8"))
            if len(batch_keys) >= batch_size:
                update_key_values(batch_keys)
                batch_keys = []
        if batch_keys:
            update_key_values(batch_keys)

        filter_values = {key: value for key, value in key_values.items() if value}
        return {i: value for i, value in enumerate(filter_values.values())}

    def load_user_feature(self):
        user_data = self._batch_load("user:*", batch_size=500)
        users = pd.DataFrame.from_dict(user_data, orient="index")
        users = self._merge_event_features(
            users, self._batch_load("feature:user:*", batch_size=500))
        user_feature = UserFeature(users=users)
        return user_feature

    def load_item_feature(self, ):
        item_data = self._batch_load("item:*", batch_size=500)
        items = pd.DataFrame.from_dict(item_data, orient="index")
        items = self._merge_event_features(
            items, self._batch_load("feature:item:*", batch_size=500))
        item_feature = ItemFeature(items=items)
        return item_feature

    @staticmethod
    def _merge_event_features(entities, snapshots):
        """Overlay data-processor snapshots onto raw entity rows by entity id.

        A snapshot is stored as ``{entityId, features: {...}}`` rather than as a flat entity.
        Keep the raw profile as the left-hand side: deleted/stale snapshot keys must not recreate
        an entity that is absent from the serving entity table.
        """
        if entities.empty or not snapshots:
            return entities
        rows = []
        for snapshot in snapshots.values():
            entity_id = snapshot.get("entityId")
            features = snapshot.get("features")
            if entity_id is None or not isinstance(features, dict):
                continue
            rows.append(dict(features, id=entity_id))
        if not rows:
            return entities
        feature_frame = pd.DataFrame(rows).drop_duplicates("id", keep="last")
        feature_columns = [name for name in feature_frame.columns if name != "id"]
        # A refreshed realtime snapshot is authoritative for behavioural columns.
        entities = entities.drop(columns=[c for c in feature_columns if c in entities.columns])
        return entities.merge(feature_frame, how="left", on="id")

    def get_item_feature_by_id(self, id=""):
        with self.lock:
            return self.namespaces["item"]["items"].get(id)

    def get_user_feature_by_id(self, id="", namespace="item"):
        with self.lock:
            return self.namespaces[namespace]["users"].get(id)

    def get_candidate_user_feature_by_id(self, id=""):
        with self.lock:
            return self.namespaces["user"]["items"].get(id)

    def refresh_if_stale(self, seconds, namespace="item"):
        with self.lock:
            loaded_at = self.namespaces[namespace]["loaded_at"]
        if seconds > 0 and loaded_at and time.monotonic() - loaded_at >= seconds:
            self.load_all_features(namespace=namespace)

    def stats(self):
        with self.lock:
            return {name: {"users": len(snapshot["users"]),
                           "candidates": len(snapshot["items"]),
                           "dim": snapshot["dim"],
                           "feature_file": snapshot["feature_file"]}
                    for name, snapshot in self.namespaces.items()}
