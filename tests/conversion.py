import torch

# Load the model from the .pth file
model_path = "../models/dns48-11decc9d8e3f0998.th"  # Replace with your model path
model_data = torch.load(model_path)

# Check the type of the loaded data
if isinstance(model_data, dict):
    # If it's a dictionary, it might contain the state_dict and other metadata
    print("Loaded a dictionary:")
    for key in model_data.keys():
        print(f"Key: {key}, Value Type: {type(model_data[key])}")
else:
    print("Loaded a model directly.")

# If it contains a state_dict, you can access it like this:
if 'state_dict' in model_data:
    state_dict = model_data['state_dict']
    print("State dict loaded.")
else:
    state_dict = model_data  # If the model itself is loaded

# You can also check for any other metadata
if 'model_identifier' in model_data:
    model_identifier = model_data['model_identifier']
    print(f"Model Identifier: {model_identifier}")
else:
    print("No model identifier found.")
