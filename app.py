from typing import Union, List
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Body
from contextlib import asynccontextmanager

from Controller import *
from SASV.SASVNet import *
from Database.Database import database_connect
from Database.db_models import Speaker
from Schemas import ResponseModel
from Database.Database import database_connect 

#init NN model
s = SASVNet(model="MFA_Conformer")
s = WrappedModel(s)
SASV_Model = Inference(s)
SASV_Model.loadParameters("weights/MFA_11spk_VSASV_1.model")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    await database_connect()
    yield
    # Clean up the ML models and release the resources


app = FastAPI(lifespan=lifespan)


# app = FastAPI()


@app.post("/enroll-speaker", response_model=ResponseModel)
async def enroll(ssid: str ,fullname: str, files: List[UploadFile] = File(...)):
    return await enroll_controller(ssid ,fullname, SASV_Model, files)

@app.post("/verify", response_model=ResponseModel)
async def verify(ssid: str , file: UploadFile = File(...)):
    return await verify_controller(ssid, file, SASV_Model)
