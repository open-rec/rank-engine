from fastapi import HTTPException


class ErrorCode:

    SUCCESS = (0, "success")
    UNKNOWN_ERROR = (1000, "unknown error")
    INVALID_MODEL = (1001, "invalid model")
    MODEL_NOT_LOAD_YET = (1002, "model not load yet")

    map = {}

    @classmethod
    def message(cls, code):
        if not cls.map:
            cls.map = {v[0]: v[1] for k, v in cls.__dict__.items() if k.isupper()}
        return cls.map.get(code, 1000)


class ReException(HTTPException):

    def __init__(self, code: int = 0):
        code, msg = code, ErrorCode.message(code)
        super().__init__(status_code=code, detail=msg)
        self.error_code = code
