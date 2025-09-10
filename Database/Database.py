from dotenv import load_dotenv
import os
from typing import List, Union

from beanie import init_beanie, Document
#beanie is a Object document mapper for mongo
from motor.motor_asyncio import AsyncIOMotorClient
#motor is a mongoDB driver that is fully async

import Database.db_models as models
from Database.db_models import Speaker, Admin

async def database_connect(secman):
    try: 
        # load_dotenv()
        # MONGO_URI = os.getenv('MONGO_URI')
        MONGO_URI=secman.get_secret(secname="doan/backend/mongouri")
        client = AsyncIOMotorClient(MONGO_URI)
        await init_beanie(database=client.db_name, document_models=models.__all__)
        print("connected to mongoDB succesfully")
    except Exception as e:
        print(f"failed to connect to mongo \n {e}")


class Speaker_CRUD:
    async def new_speaker(_speaker:Speaker) -> Speaker:
        speaker = await _speaker.create()
        return speaker

    async def delete_speaker_by_SSID(ssid: str) -> bool:
        speaker = await Speaker.find_one(Speaker.ssid == ssid)
        if speaker:
            await speaker.delete()
            return True
        return False
    
    async def find_speaker_by_SSID(ssid: str) -> Union[bool, Speaker]:
        speaker = await Speaker.find_one(Speaker.ssid == ssid)
        if speaker:
            return speaker
        return False

    async def update_speaker_by_SSID(ssid:str, ) -> Speaker:
        speaker = await Speaker.find_one(Speaker.ssid == ssid)
        if speaker:
            return True
        return False  
    
    async def get_all_speakers() -> List[Speaker]:
        speakers = await Speaker.find_all().to_list()
        return speakers
