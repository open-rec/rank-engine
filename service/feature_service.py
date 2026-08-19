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
        feature_file = feature_file or self.feature_file
        user_feature = self.load_user_feature()
        item_feature = self.load_item_feature()

        if user_feature.users.empty or item_feature.items.empty:
            with self.lock:
                self.user_feature_map = {}
                self.item_feature_map = {}
                self.loaded_at = time.monotonic()
            return

        if feature_file:
            space = FeatureSpace.load(feature_file)
            user_map, item_map = space.build_maps(user_feature.users, item_feature.items)
            with self.lock:
                self.user_feature_map, self.item_feature_map = user_map, item_map
                self.feature_dim = space.dim
                self.feature_file = feature_file
                self.loaded_at = time.monotonic()
            return

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
        with self.lock:
            self.user_feature_map = user_map
            self.item_feature_map = item_map
            self.feature_dim = user_features.shape[1] + item_features.shape[1]
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
        user_feature = UserFeature(users=users)
        return user_feature

    def load_item_feature(self, ):
        item_data = self._batch_load("item:*", batch_size=500)
        items = pd.DataFrame.from_dict(item_data, orient="index")
        item_feature = ItemFeature(items=items)
        return item_feature

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
