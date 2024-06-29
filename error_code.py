from enum import Enum

from fastapi import HTTPException


class ErrorCode(Enum):
    SUCCESS = 0
    UNKNOWN_ERROR = 1000
    INVALID_MODEL = 1001
    MODEL_NOT_LOAD_YET = 1002
    LOAD_MODEL_FAILED = 1003
    MODEL_NOT_FOUND = 1004
    INFERENCE_FAILED = 1005

    def __init__(self, code=0):
        self._code = code

    @property
    def code(self):
        return self._code

    @property
    def message(self):
        return self.name


class ReException(HTTPException):

    def __init__(self, error_code: ErrorCode):
        super().__init__(status_code=error_code.code, detail=error_code.message)
