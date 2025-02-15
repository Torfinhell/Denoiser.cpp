import torch
import os
def write_weights(model_weights, output_path):
    with open(output_path, 'w') as f:
        for name, param in model_weights.items():
            f.write(f"{name}: {param.numpy().tolist()}\n")  
for model_path in sorted(os.listdir("/home/torfinhell/Denoiser.cpp/models")):
    if(len(model_path)>=3 and model_path[-3:]==".th"):
        model_weights = torch.load(f"/home/torfinhell/Denoiser.cpp/models/{model_path}")
        write_weights(model_weights, f"/home/torfinhell/Denoiser.cpp/models/info_model/{model_path[:-3]}.txt")

