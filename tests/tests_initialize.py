import torch.nn as nn
import torch
import torch
from torch import nn

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
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(10, 2)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.fc(x)
def CreateTests(model, input:TensorContainer, path:TensorContainer):
    predictions=model(input)
    save_data_to_file(createTensorContainer(predictions), f"{path}/prediction.pth")
    save_data_to_file(model, f"{path}/model.pth")
    save_data_to_file(createTensorContainer(input), f"{path}/input.pth")
if __name__ == "__main__":
    AllTestsPath="/home/torfinhell/Denoiser.cpp/tests/test_data"
    CreateTests(SimpleModel(), torch.randn(10),f"{AllTestsPath}/SimpleModel")