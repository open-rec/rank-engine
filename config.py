import os


def _int(name, default):
    return int(os.getenv(name, str(default)))


class RedisConfig(object):
    HOST = os.getenv("REDIS_HOST", "localhost")
    PORT = _int("REDIS_PORT", 6379)
    DB = _int("REDIS_DB", 0)
    PASSWORD = os.getenv("REDIS_PASSWORD") or None
    SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2"))


class ServerConfig(object):
    HOST = os.getenv("RANK_HOST", "0.0.0.0")
    PORT = _int("RANK_PORT", 8123)
    WORKERS = _int("RANK_WORKERS", 1)


class ModelConfig(object):
    TYPE = os.getenv("MODEL_TYPE", "lr")
    PATH = os.getenv("MODEL_PATH")
    FEATURE_PATH = os.getenv("MODEL_FEATURE_PATH")
    DIM = _int("MODEL_DIM", 1024)
    REQUIRED = os.getenv("MODEL_REQUIRED", "false").lower() == "true"
    DEVICE = os.getenv("MODEL_DEVICE", "auto").lower()
    FEATURE_REFRESH_SECONDS = _int("FEATURE_REFRESH_SECONDS", 300)


class Config(object):
    REDIS = RedisConfig
    SERVER = ServerConfig
    MODEL = ModelConfig
