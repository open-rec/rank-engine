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
            self.user_feature.country,
            self.user_feature.city,
            self.user_feature.gender,
            self.user_feature.age,
            self.user_feature.tags,
        ])

        item_features = np.hstack([
            self.item_feature.category,
            self.item_feature.scene,
            self.item_feature.weight,
        ])

        self.user_feature_map = {
            user_id: user_features[i]
            for i, user_id in enumerate(user_feature.raw_id)
        }

        self.item_feature_map = {
            item_id: item_features[i]
            for i, item_id in enumerate(item_feature.raw_id)
        }

    def load_user_feature(self):
        redis_client = get_redis_client()
        user_data = {}
        for key in redis_client.keys("user:*"):
            try:
                key = key.decode("utf-8")
                value = redis_client.get_value(key).decode("utf-8", errors='ignore')
                user_data[key] = value
            except Exception as e:
                logging.warn(f"load user feature key:{key} failed, {e}")
                continue

        users = pd.DataFrame.from_dict(user_data, orient="index")
        user_feature = UserFeature(users=users)
        return user_feature

    def load_item_feature(self, ):
        redis_client = get_redis_client()
        item_data = {}
        for key in redis_client.keys("user:*"):
            try:
                key = key.decode("utf-8")
                value = redis_client.get_value(key).decode("utf-8", errors='ignore')
                item_data[key] = value
            except Exception as e:
                logging.warn(f"load item feature key:{key} failed, {e}")
                continue
        items = pd.DataFrame.from_dict(item_data, orient="index")
        item_feature = ItemFeature(items=items)
        return item_feature

    def get_item_feature_by_id(self, id=""):
        return self.item_feature_map.get(id)

    def get_user_feature_by_id(self, id=""):
        return self.user_feature_map.get(id)
