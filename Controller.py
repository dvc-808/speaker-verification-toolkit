from fastapi.responses import JSONResponse
import soundfile
import os

def preprocess_audio(file):
    return file

def check_file_format(files):
    invalid_format_files = []
    for i in files:
        if(i.content_type != "audio/wav"):
            invalid_format_files.append(i.filename)
        

def enroll_controller(model, files):
    #take request
    
    #parse request and locate audio file

    #calculate embeddings
    mean_embeds = model.enroll_user("audio_enroll")
    print(mean_embeds)
    #save new user to database 
    return {"Hello": str(mean_embeds)}

def verify_controller():
    return "dinhvietcuong"





