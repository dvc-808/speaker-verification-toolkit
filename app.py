from typing import Union, List
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Body
from contextlib import asynccontextmanager
import json

from Controller import *
from SASV.SASVNet import *
from Database.Database import database_connect
from Database.db_models import Speaker
from Schemas import ResponseModel
from Database.Database import database_connect 
from Secrete import SecreteManager
from google.cloud import speech


#init NN model
s = SASVNet(model="MFA_Conformer")
s = WrappedModel(s)
SASV_Model = Inference(s)
SASV_Model.loadParameters("weights/MFA_11spk_VSASV_1.model")

#AWS secman
secman = SecreteManager()


#init stt client
try:
    gcp = json.loads(secman.get_secret(secname="doan/backend/gcp"))
    stt_client = speech.SpeechClient.from_service_account_info(gcp)
    stt_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="vi-VN",
    )
    print ("gcp stt init successfully")
except Exception as e:
    print (f"failed to init gcp stt{e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database_connect(secman)
    yield


app = FastAPI(lifespan=lifespan)


# app = FastAPI()


@app.post("/enroll-speaker", response_model=ResponseModel)
async def enroll(ssid: str ,fullname: str, files: List[UploadFile] = File(...)):
    return await enroll_controller(ssid ,fullname, SASV_Model, files)

@app.post("/verify", response_model=ResponseModel)
async def verify(ssid: str , file: UploadFile = File(...)):
    return await verify_controller(ssid, file, SASV_Model, stt_client, stt_config)

@app.get("/speakers", response_model=ResponseModel)
async def get_spk():
    return await get_speakers()
