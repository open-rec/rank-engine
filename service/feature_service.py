import json
import logging

import numpy as np
import pandas as pd
from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.user_feature import UserFeature

from sugar import singleton
from util.redis_util import get_redis_client


@singleton
class FeatureService(object):

    def __init__(self):
        self.user_feature_map = {}
        self.item_feature_map = {}
        self.load_all_features()

    def load_all_features(self):
        user_feature = self.load_user_feature()
        item_feature = self.load_item_feature()

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

        self.user_feature_map = {
            user_id: user_features[i]
            for i, user_id in enumerate(user_feature.raw_id)
        }

        self.item_feature_map = {
            item_id: item_features[i]
            for i, item_id in enumerate(item_feature.raw_id)
        }

    @staticmethod
    def _batch_load(key_pattern="*", batch_size=500):
        redis_client = get_redis_client()
        keys = redis_client.keys(key_pattern)
        length = len(keys)
        tick = 0
        key_values = {}
        batch_keys = []

        def update_key_values(batch_keys=[]):
            values = redis_client.batch_get_values(batch_keys)
            for key, value in zip(batch_keys, values):
                try:
                    key_values[key] = json.loads(value.decode("utf-8"))
                except Exception as e:
                    logging.warn(f"load key:{key}, value:{value} failed")
                    continue
            batch_keys.clear()

        while tick < length:
            batch_keys.append(keys[tick].decode("utf-8"))
            tick += 1
            if tick % batch_size == 0:
                update_key_values(batch_keys=batch_keys)
        if batch_keys:
            update_key_values(batch_keys=batch_keys)

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
        return self.item_feature_map.get(id)

    def get_user_feature_by_id(self, id=""):
        return self.user_feature_map.get(id)
