from fastapi.responses import JSONResponse
import soundfile
import os

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
    for file in files:
        file_location = os.path.join(enroll_path, file.filename)
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

async def verify_controller():
    return "dinhvietcuong"





