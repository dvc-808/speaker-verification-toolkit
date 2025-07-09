from typing import Union
from fastapi import FastAPI
from contextlib import asynccontextmanager
from Controller import *
from os import name
from SASVNet import *

# model = object()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
# Load the ML model
s = SASVNet(model="MFA_Conformer")
s = WrappedModel(s)
Infer = Inference(s)
Infer.loadParameters("weights/MFA_11spk_VSASV_1.model")
model = Infer
    # yield
    # Clean up the ML models and release the resources

# app = FastAPI(lifespan=lifespan)
app = FastAPI()


@app.get("/")
async def enroll():
    return enroll_controller(model)

@app.get("/verify")
def verify():
    return verify_controller()

if __name__ == "__main__":
    print("hello world")