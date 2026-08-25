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

    def __init__(self):
        self.user_feature_map = {}
        self.item_feature_map = {}
        self.feature_dim = 0
        self.feature_file = None
        self.loaded_at = 0
        self.lock = threading.RLock()

    def load_all_features(self, feature_file=None):
        snapshot = self.prepare_all_features(feature_file)
        self.activate(snapshot)

    def prepare_all_features(self, feature_file=None):
        """Build a feature snapshot without changing the live scorer."""
        feature_file = feature_file or self.feature_file
        user_feature = self.load_user_feature()
        item_feature = self.load_item_feature()

        if user_feature.users.empty or item_feature.items.empty:
            return {"users": {}, "items": {}, "dim": 0,
                    "feature_file": feature_file, "feature_set": None,
                    "catalog_version": None, "model_type": None}

        if feature_file:
            space = FeatureSpace.load(feature_file)
            user_map, item_map = space.build_maps(user_feature.users, item_feature.items)
            return {"users": user_map, "items": item_map, "dim": space.dim,
                    "feature_file": feature_file, "feature_set": space.feature_set,
                    "catalog_version": space.catalog_version, "model_type": space.model_type}

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
                "catalog_version": None, "model_type": None}

    def activate(self, snapshot):
        with self.lock:
            self.user_feature_map = snapshot["users"]
            self.item_feature_map = snapshot["items"]
            self.feature_dim = snapshot["dim"]
            self.feature_file = snapshot["feature_file"]
            self.loaded_at = time.monotonic()

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
            return self.item_feature_map.get(id)

    def get_user_feature_by_id(self, id=""):
        with self.lock:
            return self.user_feature_map.get(id)

    def refresh_if_stale(self, seconds):
        if seconds > 0 and time.monotonic() - self.loaded_at >= seconds:
            self.load_all_features()

    def stats(self):
        with self.lock:
            return {"users": len(self.user_feature_map), "items": len(self.item_feature_map),
                    "dim": self.feature_dim}
