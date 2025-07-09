from json import load
import sys
import os
import time
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile

# from torch.cuda.amp import autocast, GradScaler
# from torch.cuda.amp import *
# import wandb
# from DatasetLoader import test_dataset_loader


class WrappedModel(nn.Module):
    
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x)


class SASVNet(nn.Module):
    def __init__(self, model, **kwargs):
        super(SASVNet, self).__init__()
        SASVNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SASVNetModel(**kwargs)

    def forward(self, data):
        data = data.reshape(-1, data.size()[-1])
        return self.__S__.forward(data, aug=False) 


def loadWAV(filename, max_frames = 500, evalmode=True):
    #? define the max audio sample in a file
    #? why 160: every frame is 10ms in length, with 16kHz sampling rate => 160 samples per frame
    max_audio = max_frames * 160
    audio = 0
    # Read wav file and convert to torch tensor
    try:
        audio, _ = soundfile.read(filename)
        #? with soundfile.read, the output audio is a nparray with size of (size, channel) eg. (160,2)
    except Exception as e:
        print(e)

    audiosize = audio.shape[0]

    #?pad audio to maxsize
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1 
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]
    feats = []
    feats += [audio]
    feat = np.stack(feats,axis=0).astype(np.float32)
    return feat

class Inference(object):
    
    def __init__(self, speaker_model):
        self.__model__  = speaker_model
        self.__model__.eval()
        # self.device
        # self.gpu = 0
        # self.ngpu = 1

    def enroll_user(self, audio_path):
        embeddings_list = []
        audio_files = os.listdir(audio_path)
        
        for filename in audio_files:
            filename = os.path.join(audio_path,filename)
            inp1 = torch.FloatTensor(loadWAV(filename))
            with torch.no_grad():
                embed = self.__model__(inp1).detach().cpu()
            embeddings_list += [embed]

        mean_embeds = torch.mean(torch.stack(embeddings_list), dim=0)
        return mean_embeds

    def infer(self, filename, max_frames):
        
        self.__model__.eval()
        inp1 = loadWAV(filename, max_frames)
        with torch.no_grad():
            ref_embed = self.__model__(inp1).detach().cpu()


    def loadParameters(self, path):
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cpu")
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}"
                      .format(origname, self_state[name]
                      .size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
