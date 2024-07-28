import logging

import redis

common_redis_client = None


class RedisClient(object):

    def __init__(self, host="localhost", port=6379, db=0):
        try:
            self.pool = redis.ConnectionPool(host=host, port=port, db=db)
            self.client = redis.Redis(connection_pool=self.pool)
        except Exception as e:
            logging.error(f"redis client init failed: {e}")
            raise e

    def exists_key(self, key=""):
        return self.client.exists(key) == 1

    def get_value(self, key=""):
        return self.client.get(key)

    def mget_values(self, keys=[]):
        return self.client.mget(keys)

    def keys(self, pattern="*"):
        return self.client.keys(pattern)


def get_redis_client():
    global common_redis_client
    if not common_redis_client:
        common_redis_client = RedisClient(host="localhost", port=6379, db=0)
    return common_redis_client
