import torch.nn as nn
import torch
def save_data_to_file(data, file_path):
    if(isinstance(data, torch.nn.Module)):
        torch.save(data.state_dict(), file_path)
    elif(isinstance(data, torch.Tensor)):
        torch.save(data, file_path)
    else:
        print(f'{data.__class__.__name__} cannot be saved')
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(10, 2)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.fc(x)
def CreateTests(model, input, path):
    save_data_to_file(model, f"{path}/model.pth")
    save_data_to_file(input, f"{path}/input.pth")
    predictions=model(input)
    save_data_to_file(predictions, f"{path}/prediction.pth")


if __name__ == "__main__":
    AllTestsPath="/home/torfinhell/Denoiser.cpp/tests/test_data"
    CreateTests(SimpleModel(), torch.randn(10),f"{AllTestsPath}/SimpleModel")