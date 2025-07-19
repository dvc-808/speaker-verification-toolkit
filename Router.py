from typing import Union
from fastapi import FastAPI, UploadFile, File
from Controller import *
from SASVNet import *

s = SASVNet(model="MFA_Conformer")
s = WrappedModel(s)
Infer = Inference(s)
Infer.loadParameters("weights/MFA_11spk_VSASV_1.model")
model = Infer
app = FastAPI()

@app.post("/enroll-user")
#the ellipsis ... means "this is required"
async def enroll(fullname: str, age :int, files: list[UploadFile] = File(...) ):
    return enroll_controller(model, files)

@app.get("/verify")
def verify():
    return verify_controller()
