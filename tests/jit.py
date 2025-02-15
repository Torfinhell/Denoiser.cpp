import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(3 * 224 * 224, 10)  # Example for an image input

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

model = SimpleModel()
scripted_model = torch.jit.script(model)
scripted_model.save(f"/home/torfinhell/Denoiser.cpp/models/{model.__class__.__name__}_jit.pth")
