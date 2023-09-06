import json
from typing import List

import motor.motor_asyncio
from bson import json_util, ObjectId
from fastapi import FastAPI, Body, Path
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from geopy import GoogleV3


from base_response import ApiResponse
from model.cctv import CCTVModel

import sinar

app = FastAPI()
client = motor.motor_asyncio.AsyncIOMotorClient("mongodb+srv://public:20031015@test-crud.utmjs38.mongodb.net/")
db = client.sinar

geolocator = GoogleV3(api_key="") # apikey geolocation api

@app.get("/")
async def root():
    return FileResponse("/var/www/html/index.nginx-debian.html")


@app.get("/ping")
async def ping():
    # geolocator = Nominatim(user_agent="com.pkmkc.backend")
    location = geolocator.reverse("-6.967863, 107.634587")
    # location = geolocator.reverse("-18.252100, 96.706631")
    print(location.address)
    return ApiResponse.success(str(location), message="pong")


@app.get("/total-crime")
async def get_total_cctv():
    cctv = [doc async for doc in db["cctv"].find({"status": "DANGER"})]
    return ApiResponse.success(len(cctv))


@app.get("/list-cctv")
async def get_list_cctv():
    cctv = [doc async for doc in db["cctv"].find()]
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
    location = geolocator.reverse(f"{cctv['lat']}, {cctv['lng']}")
    cctv["address"] = str(location)
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
        if "lat" in get_cctv or "lng" in get_cctv:
            if (temp_cctv := await db["cctv"].find_one({"_id": ObjectId(cctv_id)})) is not None:
                location = geolocator.reverse(f"{get_cctv.get('lat', temp_cctv['lat'])}, {get_cctv.get('lng', temp_cctv['lng'])}")
                get_cctv["address"] = str(location)

        update_result = await db["cctv"].update_one({"_id": ObjectId(cctv_id)}, {"$set": get_cctv})

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
        ret = sinar.new_sinar_process("best.pt", "anbev-cnn-fix.keras", 
            f"rtmp://localhost/input/{cctv_id}", f"rtmp://localhost/output/{cctv_id}", 
            process_name=f"{cctv_id}")
        if ret:
            return ApiResponse.success(True)
        else:
            return ApiResponse.failed(f"Gagal menjalankan tracker {cctv_id}")

@app.get("/stop-tracker/{cctv_id}")
async def stop_tracker_cctv(cctv_id: str):
    if len(cctv_id) > 0:
        # Panggil tracker
        ret = sinar.stop_process(cctv_id)
        if ret:
            return ApiResponse.success(True)
        else:
            return ApiResponse.failed(f"Gagal menghentikan tracker {cctv_id}")
