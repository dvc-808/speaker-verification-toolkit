from typing import Union
from fastapi import FastAPI
from Controller import *

app = FastAPI()


@app.get("/")
async def enroll():
    return enroll_controller()

@app.get("/verify")
async def verify():
    return verify_controller()
