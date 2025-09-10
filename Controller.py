from fastapi.responses import JSONResponse
import soundfile
import os
import time
import torch.nn.functional as F
from torch import FloatTensor
from pydub import AudioSegment
from google.cloud import speech

from Database.Database import Speaker_CRUD
from Database.db_models import Speaker

def audio_converter(path):
    # Load the audio file (any format supported by ffmpeg)
    audio = AudioSegment.from_file(path)

    # Convert to mono, 16kHz, 16-bit
    audio = audio.set_channels(1)         # mono
    audio = audio.set_frame_rate(16000)   # 16kHz
    audio = audio.set_sample_width(2)     # 2 bytes = 16 bits

    # Generate output path with .wav extension
    base, _ = os.path.splitext(path)
    output_path = base + "_processed.wav"

    # Export as WAV
    audio.export(output_path, format="wav")
    return output_path

async def enroll_controller(ssid ,fullname, model, files):
    #check SSID duplication
    speaker = await Speaker_CRUD.find_speaker_by_SSID(ssid)
    if speaker:
        print ("speaker duplicate")
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
        converted_path = audio_converter(file_location)
        os.replace(converted_path, file_location)

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

async def verify_controller(ssid, file, model, stt_client, stt_config):
    speaker = await Speaker_CRUD.find_speaker_by_SSID(ssid)
    if not speaker:
        return {
            "status_code": 404,
            "response_type":"failed",
            "data": f"No speaker with {ssid} in the database"
        }
    file_extension = os.path.splitext(file.filename)[-1]
    
    os.makedirs("media/verify", exist_ok=True)
    verify_path = f"media/verify/{int(time.time())}{file_extension}"
    # Save uploaded files to enroll_path
    with open(verify_path, "wb") as f:
        f.write(await file.read())
    converted_path = audio_converter(verify_path)
    test_embeddings = model.verify(converted_path)
    target_embeddings=FloatTensor(speaker.embeddings).unsqueeze(0)
    ts=F.normalize(test_embeddings, p=2, dim=1)
    tar=F.normalize(target_embeddings, p=2, dim=1)
    score = F.cosine_similarity(ts, tar)
    valid = 0
    if score.item() >= 0.6:
        valid = 1
    print(f'score: {score}')
    
    with open(converted_path, "rb") as audio_file:
        content = audio_file.read()

    stt_audio = speech.RecognitionAudio(content=content)
    stt_response = stt_client.recognize(config=stt_config, audio=stt_audio)
    text = stt_response.results[0].alternatives[0].transcript
    print(text)
    os.remove(verify_path)
    os.remove(converted_path)
    return {
        "status_code": 200,
        "response_type":"success",
        "data": {"valid":valid, "text":str(text)}
    }


async def get_speakers():
    try:
        speakers = await Speaker_CRUD.get_all_speakers()
        data = [
            {
                "ssid": s.ssid,
                "name": s.fullname,
            }
            for s in speakers
        ]
        return {
            "status_code": 200,
            "response_type": "success",
            "data": data,
        }
    except Exception as e:
        print (f"failed to fetch speakers \n {e}")
        return {
            "status_code": 500,
            "response_type":"failed",
            "data": "Internal server error",
        }




