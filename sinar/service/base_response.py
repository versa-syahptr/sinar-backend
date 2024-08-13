import json
from typing import TypeVar, Generic
from fastapi.responses import JSONResponse
from bson import json_util

T = TypeVar("T")


class ApiResponse(Generic[T]):

    @staticmethod
    def success(data: T, message: str = "") -> JSONResponse:
        return JSONResponse(status_code=200, content={"success": True, "message": message, "data": data})

    @staticmethod
    def failed(message: str) -> JSONResponse:
        return JSONResponse(status_code=400, content={"success": False, "message": message, "data": None})
