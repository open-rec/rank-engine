import logging

import redis

from config import Config

common_redis_client = None


class RedisClient(object):

    def __init__(self, host="localhost", port=6379, db=0, password=None, socket_timeout=2):
        try:
            self.pool = redis.ConnectionPool(host=host, port=port, db=db,
                                             password=password, socket_timeout=socket_timeout,
                                             socket_connect_timeout=socket_timeout,
                                             decode_responses=False)
            self.client = redis.Redis(connection_pool=self.pool)
        except Exception as e:
            logging.error(f"redis client init failed: {e}")
            raise e

    def exists_key(self, key=""):
        return self.client.exists(key) == 1

    def get_value(self, key=""):
        return self.client.get(key)

    def mget_values(self, keys=None):
        keys = keys or []
        return self.client.mget(keys)

    def batch_get_values(self, keys=None):
        keys = keys or []
        pipeline = self.client.pipeline()
        for key in keys:
            pipeline.get(key)
        return pipeline.execute()

    def delete_keys(self, keys):
        self.client.delete(keys)

    def keys(self, pattern="*"):
        return self.client.keys(pattern)

    def scan_iter(self, pattern="*", count=500):
        return self.client.scan_iter(match=pattern, count=count)

    def ping(self):
        return self.client.ping()


def get_redis_client():
    global common_redis_client
    if not common_redis_client:
        common_redis_client = RedisClient(host=Config.REDIS.HOST, port=Config.REDIS.PORT,
                                          db=Config.REDIS.DB, password=Config.REDIS.PASSWORD,
                                          socket_timeout=Config.REDIS.SOCKET_TIMEOUT)
    return common_redis_client
