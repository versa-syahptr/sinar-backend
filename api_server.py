import json
from typing import List

import motor.motor_asyncio
from bson import json_util, ObjectId
from fastapi import FastAPI, Body, Path
from fastapi.encoders import jsonable_encoder

from base_response import ApiResponse
from model.cctv import CCTVModel

app = FastAPI()
client = motor.motor_asyncio.AsyncIOMotorClient("mongodb+srv://public:20031015@test-crud.utmjs38.mongodb.net/")
db = client.sinar


@app.get("/ping")
async def ping():
    return ApiResponse.success("pong")


@app.get("/list-cctv")
async def get_list_cctv():
    cctv = await db["cctv"].find().to_list(1000)
    cctv: List[CCTVModel] = json.loads(json_util.dumps(cctv))
    if len(cctv) > 0:
        for item in cctv:
            item["_id"] = item["_id"]["$oid"]
    return ApiResponse.success(cctv)


@app.get("/cctv/{cctv_id}")
async def get_one_cctv(cctv_id: str):
    cctv = await db["cctv"].find_one({"_id": ObjectId(cctv_id)})

    if cctv:
        cctv: CCTVModel = json.loads(json_util.dumps(cctv))
        cctv["_id"] = cctv["_id"]["$oid"]
        return ApiResponse.success(cctv)
    else:
        return ApiResponse.failed(f"CCTV {cctv_id} not found")


@app.post("/insert-cctv")
async def insert_cctv(body=Body(...)):
    cctv:dict = jsonable_encoder(body)
    if "lat" not in cctv.keys() or "lng" not in cctv.keys():
        return ApiResponse.failed("Lat, Lng is Null")
    cctv["status"] = "SAFE"
    new_cctv = await db["cctv"].insert_one(cctv)
    created_cctv = await db["cctv"].find_one({"_id": ObjectId(new_cctv.inserted_id)})
    created_cctv: CCTVModel = json.loads(json_util.dumps(created_cctv))
    created_cctv["_id"] = created_cctv["_id"]["$oid"]
    return ApiResponse.success(created_cctv)


@app.post("/update-cctv/{cctv_id}")
async def update_cctv(cctv_id:str, body=Body(...)):
    cctv: dict = jsonable_encoder(body)
    if len(cctv) == 0:
        return ApiResponse.failed("Body length is Null")
    get_cctv = {k: v for k, v in cctv.items() if v is not None}
    if len(get_cctv) >= 1:
        update_result = await db["cctv"].update_one({"_id": cctv_id}, {"$set": get_cctv})
        if update_result.modified_count == 1:
            if (updated_cctv := await db["cctv"].find_one({"_id": ObjectId(cctv_id)})) is not None:
                updated_cctv: CCTVModel = json.loads(json_util.dumps(updated_cctv))
                updated_cctv["_id"] = updated_cctv["_id"]["$oid"]
                return ApiResponse.success(updated_cctv)
    if (existing_cctv := await db["cctv"].find_one({"_id": ObjectId(cctv_id)})) is not None:
        existing_cctv: CCTVModel = json.loads(json_util.dumps(existing_cctv))
        existing_cctv["_id"] = existing_cctv["_id"]["$oid"]
        return ApiResponse.success(existing_cctv)
    return ApiResponse.failed(f"CCTV {cctv_id} not found")


@app.get("/start-tracker/{cctv_id}")
async def start_tracker_cctv(cctv_id: str):
    if len(cctv_id) > 0:
        # Panggil tracker
        if True:
            return ApiResponse.success(True)
        else:
            return ApiResponse.failed(f"Gagal menjalankan tracker {cctv_id}")

