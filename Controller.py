def enroll_controller(Inference):
    #take request
    #parse request and locate audio file
    #calculate embeddings
    mean_embeds = Inference.enroll_user("audio_enroll")
    print(mean_embeds)
    #save new user to database
    return {"Hello": mean_embeds}

def verify_controller():
    return "dinhvietcuong"





