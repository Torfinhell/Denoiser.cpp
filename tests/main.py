from demucs import Demucs
import torch
import torch.nn as nn
ROOT = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/"
DNS_48_URL = ROOT + "dns48-11decc9d8e3f0998.th"
DNS_64_URL = ROOT + "dns64-a7761ff99a7d5bb6.th"
MASTER_64_URL = ROOT + "master64-8a5dfb4bb92753dd.th"
VALENTINI_NC = ROOT + 'valentini_nc-93fc4337.th'  # Non causal Demucs on Valentini

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # A simple fully connected layer

    def forward(self, x):
        return self.fc(x)
state_dict = torch.hub.load_state_dict_from_url(DNS_48_URL, map_location='cpu')
model=Demucs(hidden=48,sample_rate=16_000)
# model.load_state_dict(state_dict)
print("Model parameters (state_dict):")
for name, param in model.state_dict().items():
    print(f"{name}: {param}")