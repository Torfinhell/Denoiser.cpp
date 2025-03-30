import torch.nn as nn
import torch
import torch
from torch import nn
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
def CreateTests(model, input:TensorContainer, path:TensorContainer, predictions=None):
    os.makedirs(path, exist_ok=True)
    if(predictions is None):
        predictions=model(input)
    # print(input)
    # for layer in model.encoder:
    #     if isinstance(layer, nn.Sequential):
    #         for sub_layer in layer:
    #             if isinstance(sub_layer, nn.Conv1d):
    #                 print("Convolutional weights:", sub_layer.weight.data)
    # print(predictions)
    save_data_to_file(createTensorContainer(predictions), f"{path}/prediction.pth")
    # save_data_to_file(model, f"{path}/model.pth")
    save_data_to_file(createTensorContainer(input), f"{path}/input.pth")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(10,2)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.fc(x) 
class TestModel(nn.Module):
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
class BasicDemucs(nn.Module):
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 floor=1e-3,
                 sample_rate=16_000):
        super().__init__()
        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate
        self.growth=growth
        self.max_hidden=max_hidden
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1
        for index in range(self.depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, self.kernel_size, self.stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, self.kernel_size, self.stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(self.growth * hidden), self.max_hidden)
        self.lstm = BLSTM(chin, bi=False)#not casual false
    def forward(self, mix: th.Tensor):
        # if mix.dim() == 2:
        #     mix = mix.unsqueeze(1)

        #self.normalize=true
        mono = mix.mean(dim=1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)
        mix = mix / (self.floor + std)
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        x = upsample2(x)
        x = upsample2(x)
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        assert x.dim() == 3
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)
        x = downsample2(x)
        x = downsample2(x)
        x = x[..., :length]
        return std * x
    def valid_length(self, length:int):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length*self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)
def final_test():
    CreateTests(BasicDemucs(), torch.randn(1, 1, 256)
,f"{AllTestsPath}/BasicDemucs")
    sr = 16_000
    sr_ms = sr / 1000
    demucs = dns48()
    # x = th.randn(1, int(sr * 4)).to("cpu")
    x = th.randn(1, int(256)).to("cpu")
    # out = demucs(x[None])[0]
    CreateTests(demucs, x[None], f"{AllTestsPath}/dns48")
def DemucsStreamerTest(x):
    demucs = Demucs()
    streamer = DemucsStreamer(demucs)
    out_rt = []
    frame_size = streamer.total_length
    with th.no_grad():
        while x.shape[1] > 0:
            # print(x[:, :frame_size])
            out_rt.append(streamer.feed(x[:, :frame_size]))
            return out_rt[0]
            x = x[:, frame_size:]
            frame_size = streamer.demucs.total_stride
    # out_rt.append(streamer.flush())
    out_rt = th.cat(out_rt, 1)
    return out_rt
if __name__ == "__main__":
    AllTestsPath="/home/torfinhell/Denoiser.cpp/main_app/tests/test_data"
    # CreateTests(SimpleModel(), torch.randn(10),f"{AllTestsPath}/SimpleModel")
    # CreateTests(OneEncoder(), torch.randn(2, 1, 8),f"{AllTestsPath}/SimpleEncoderDecoder")
    # final_test()
    #read audio
    sr=16_000
    x = th.randn(1, int(4*sr))
    CreateTests(DemucsStreamerTest,x,f"{AllTestsPath}/DemucsStreamer")