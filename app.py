from typing import Union
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Body
from Controller import *
from SASV.SASVNet import *
from Database.Database import database_connect
from Database.db_models import Speaker

#init NN model
s = SASVNet(model="MFA_Conformer")
s = WrappedModel(s)
model = Inference(s)
model.loadParameters("weights/MFA_11spk_VSASV_1.model")

app = FastAPI()

@app.get("/")
async def dummy():
    await database_connect()

@app.post("/enroll-user")
async def enroll(speaker:Speaker=Body(...), files: list[UploadFile] = File(...) ):
    #the ellipsis ... means "this is required"
    return enroll_controller(model, files)

@app.get("/enroll-user")
async def check_availability(ssid: str ,fullname: str, age :int):
    #if
    return 

@app.get("/verify")
def verify():
    return verify_controller()
