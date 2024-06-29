import json

from pydantic import BaseModel, Field


class Model(BaseModel):
    model: str = Field(default="lr.pth")
    dim: int = Field(default=1024)
    type: str = Field(default="lr")


class UserItems(BaseModel):
    user_id: str = Field(default="", description="user id")
    item_ids: list = Field(default=[], description="user score items")


class ReResponse:

    def __init__(self, code=0, status="success", data=None, message=""):
        self.code = code
        self.status = status
        self.data = data
        self.message = message

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.to_dict())


if __name__ == "__main__":
    print(type(ReResponse().to_json()))
