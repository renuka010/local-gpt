from __future__ import annotations
from typing import List, Optional, Any
from pydantic.generics import GenericModel, Generic, TypeVar
from pydantic import BaseModel

T = TypeVar('T')

# Response Schema(No pagination)
class ResponseModel(GenericModel, Generic[T]):
    result: Optional[T]
    code: str
    status: str
    message: str


class FetchRequestModel(BaseModel):
    query: str