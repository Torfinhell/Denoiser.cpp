from demucs import Demucs
import torch
import torch.nn as nn
ROOT = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/"
DNS_48_URL = ROOT + "dns48-11decc9d8e3f0998.th"
DNS_64_URL = ROOT + "dns64-a7761ff99a7d5bb6.th"
MASTER_64_URL = ROOT + "master64-8a5dfb4bb92753dd.th"
VALENTINI_NC = ROOT + 'valentini_nc-93fc4337.th'  # This is for later research
#check for existense of folders

def load_url_DNS_48(file_name:str='/home/torfinhell/Denoiser.cpp/output/DNS_48.txt'):
    model=Demucs(hidden=48,sample_rate=16_000)
    state_dict = torch.hub.load_state_dict_from_url(DNS_48_URL, map_location='cpu')
    model.load_state_dict(state_dict)
    with open(file_name, 'w') as f:
        for name, param in model.state_dict().items():
            print(f"{name}: {param}", file=f)
def load_url_DNS_64(file_name:str='/home/torfinhell/Denoiser.cpp/output/DNS_64.txt'):
    model=Demucs(hidden=64,sample_rate=16_000)
    state_dict = torch.hub.load_state_dict_from_url(DNS_64_URL, map_location='cpu')
    model.load_state_dict(state_dict)
    with open(file_name, 'w') as f:
        for name, param in model.state_dict().items():
            print(f"{name}: {param}", file=f)
def load_url_MASTER_64(file_name:str='/home/torfinhell/Denoiser.cpp/output/MASTER_64.txt'):
    model=Demucs(hidden=64,sample_rate=16_000)
    state_dict = torch.hub.load_state_dict_from_url(MASTER_64_URL, map_location='cpu')
    model.load_state_dict(state_dict)
    with open(file_name, 'w') as f:
        for name, param in model.state_dict().items():
            print(f"{name}: {param}", file=f)
def save_model_to_file(model):
    torch.save(model.state_dict(), f'/home/torfinhell/Denoiser.cpp/models/{model.__class__.__name__}.pth')
def GetModel(model, file_name:str='/home/torfinhell/Denoiser.cpp/output/SimpleModel.txt'):
    state_dict =    torch.load(f'/home/torfinhell/Denoiser.cpp/models/{model.__class__.__name__}.pth')
    model.load_state_dict(state_dict)
    return model
def weights_to_txt(model, file_name:str='/home/torfinhell/Denoiser.cpp/output/SimpleModel.txt'):
    model =GetModel(model, file_name)
    with open(file_name, 'w') as f:
        for name, param in model.state_dict().items():
            print(f"{name}: {param}", file=f)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)
model=SimpleModel()
save_model_to_file(model)
weights_to_txt(model)