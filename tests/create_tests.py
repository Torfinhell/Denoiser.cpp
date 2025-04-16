import torch.nn as nn
import torch
from our_denoise_demucs_48 import *
import os
class TensorContainer(nn.Module):
    def __init__(self, tensor_dict):
        super().__init__()
        for key,value in tensor_dict.items():
            setattr(self, key, value)
def createTensorContainer(x:torch.Tensor):
    tensor_dict = {'prior': x}
    return TensorContainer(tensor_dict)
def save_data_to_file(data, file_path):
    data=torch.jit.script(data)
    torch.jit.save(data, file_path)
def CreateTests(model, input:TensorContainer, path:TensorContainer, save_model=None,prediction=None):
    os.makedirs(path, exist_ok=True)
    if(prediction is None):
        prediction=model(input)
    if(save_model is None):
        save_data_to_file(model, f"{path}/model.pth")
    save_data_to_file(createTensorContainer(prediction), f"{path}/prediction.pth")
    save_data_to_file(createTensorContainer(input), f"{path}/input.pth")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(10,2)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.fc(x) 
class OneEncoder(nn.Module):
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 ch_scale = 2,
                 kernel_size=8,
                 stride=4,
                 ):
        super().__init__()
        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.ch_scale = ch_scale
        self.kernel_size=kernel_size
        self.stride=stride
        activation = nn.GLU(1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encode = []
        encode += [
            nn.Conv1d(chin, hidden, self.kernel_size, self.stride),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden * ch_scale, 1), 
            activation,
        ]
        self.encoder.append(nn.Sequential(*encode))
    def forward(self, x):
        for layer in self.encoder:
            x=layer(x)
        return x
class SimpleEncoderDecoder(nn.Module):
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,###########
                 ch_scale = 2,
                 kernel_size=8,
                 stride=4,
                 ):
        super().__init__()
        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.ch_scale = ch_scale
        self.kernel_size=kernel_size
        self.stride=stride
        activation = nn.GLU(1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encode = []
        encode += [
            nn.Conv1d(chin, hidden, self.kernel_size, self.stride),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden * ch_scale, 1), 
            activation,
        ]
        self.encoder.append(nn.Sequential(*encode))
        decode=[]
        decode += [
            nn.Conv1d(hidden, ch_scale * hidden, 1), 
            activation,
            nn.ConvTranspose1d(hidden, chout, self.kernel_size, self.stride),
        ]
        self.decoder.append(nn.Sequential(*decode))
    def forward(self, x):
        for layer in self.encoder:
            x=layer(x)
        for layer in self.decoder:
            x=layer(x)
        return x
class SimpleEncoderDecoderLSTM(nn.Module):
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 ch_scale = 2,
                 kernel_size=8,
                 stride=4,
                 ):
        super().__init__()
        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.ch_scale = ch_scale
        self.kernel_size=kernel_size
        self.stride=stride
        activation = nn.GLU(1)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encode = []
        encode += [
            nn.Conv1d(chin, hidden, self.kernel_size, self.stride),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden * ch_scale, 1), 
            activation,
        ]
        self.encoder.append(nn.Sequential(*encode))
        decode=[]
        decode += [
            nn.Conv1d(hidden, ch_scale * hidden, 1), 
            activation,
            nn.ConvTranspose1d(hidden, chout, self.kernel_size, self.stride),
        ]
        self.decoder.append(nn.Sequential(*decode))
        self.lstm = BLSTM(hidden, bi=False)
    def forward(self, x):
        for layer in self.encoder:
            x=layer(x)
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for layer in self.decoder:
            x=layer(x)
        return x


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    AllTestsPath=f"{current_dir}/test_data"
    CreateTests(SimpleModel(), torch.randn(10),f"{AllTestsPath}/SimpleModel")
    CreateTests(OneEncoder(), torch.randn(2, 1, 20),f"{AllTestsPath}/OneEncoder")
    CreateTests(SimpleEncoderDecoder(), torch.randn(2, 1, 20),f"{AllTestsPath}/SimpleEncoderDecoder")
    CreateTests(SimpleEncoderDecoderLSTM(), torch.randn(2, 1, 20),f"{AllTestsPath}/SimpleEncoderDecoderLSTM")
    sr = 16_000
    x = th.randn(1, int(256)).to("cpu")
    CreateTests(dns48(), x[None], f"{AllTestsPath}/dns48")