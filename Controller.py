from fastapi.responses import JSONResponse
import soundfile
import os
import time
import torch.nn.functional as F
from torch import FloatTensor

from Database.Database import Speaker_CRUD
from Database.db_models import Speaker

def audio_converter(path):
    return

async def enroll_controller(ssid ,fullname, model, files):
    #check SSID duplication
    speaker = await Speaker_CRUD.find_speaker_by_SSID(ssid)
    if speaker:
        return {
            "status_code": 409,
            "response_type":"failed",
            "data": f"There exist a speaker with {ssid} in the database"
        }
    
    #create folder for the speaker and store audio file
    enroll_path = os.path.join("media/enroll", ssid)
    os.makedirs(enroll_path, exist_ok=True)
    # Save uploaded files to enroll_path
    for i, file in enumerate(files):
        file_location = os.path.join(enroll_path, f"{str(i)}.wav")
        with open(file_location, "wb") as f:
            f.write(await file.read())

    #calculate embeddings
    try:
        mean_embeds = model.enroll_user(enroll_path) #pass the audio path here   
    except Exception as e:
        print (f"failed to enroll speaker \n {e}")

    #save new user to database 
    try:
        new_speaker = Speaker(
            fullname=fullname,
            ssid=ssid,
            audio_path=enroll_path,
            embeddings=mean_embeds
        )
        speaker = await Speaker_CRUD.new_speaker(new_speaker)
    except Exception as e:
        print (f"There was an error occured when creating new speaker \n {e}")
        return {
            "status_code": 500,
            "response_type":"failed",
            "data": "Internal server error"
        }
    
    return {
        "status_code": 200,
        "response_type":"success",
        "data": f"new speaker with {ssid} successfully created"
    }

async def verify_controller(ssid, file, model):
    speaker = await Speaker_CRUD.find_speaker_by_SSID(ssid)
    if not speaker:
        return {
            "status_code": 404,
            "response_type":"failed",
            "data": f"No speaker with {ssid} in the database"
        }
    
    verify_path = f"media/verify/{int(time.time())}.wav"
    # Save uploaded files to enroll_path
    with open(verify_path, "wb") as f:
        f.write(await file.read())

    test_embeddings = model.verify(verify_path)
    target_embeddings=FloatTensor(speaker.embeddings).unsqueeze(0)

    ts=F.normalize(test_embeddings, p=2, dim=1)
    tar=F.normalize(target_embeddings, p=2, dim=1)
    score = F.cosine_similarity(ts, tar)
    os.remove(verify_path)
    return {
        "status_code": 200,
        "response_type":"success",
        "data": f"test with {score.item()}"
    }







