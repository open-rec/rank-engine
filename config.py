
class RedisConfig(object):
    HOST = "localhost"
    PORT = 6379
    DB = 0


class ServerConfig(object):
    HOST = "0.0.0.0"
    PORT = 8000


class Config(object):
    REDIS = RedisConfig
    SERVER = ServerConfig


