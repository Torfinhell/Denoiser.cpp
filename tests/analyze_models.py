import os
import torch
from collections import OrderedDict

def load_state_dict_from_pth(file_path):
    """Load the state dictionary from a .pth or .pt file."""
    state_dict = torch.load(file_path)
    
    if isinstance(state_dict, dict):
        if 'state_dict' in state_dict:
            return state_dict['state_dict']
        else:
            return state_dict
    else:
        raise ValueError("Loaded data is not a dictionary.")

def print_state_dict_info(state_dict, output_file):
    """Print the state dictionary information to a text file."""
    with open(output_file, 'w') as f:
        for key in state_dict.keys():
            f.write(f"{key}: {state_dict[key].shape}\n")

def clear_output_folder(output_folder):
    """Clear all .pt and .th files in the output folder."""
    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path) and (filename.endswith('.pt') or filename.endswith('.th')):
                os.remove(file_path)
                print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

def process_models(input_folder, output_folder):
    """Process all .pt and .th files in the input folder and save info to the output folder."""
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Clear the output folder
    clear_output_folder(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.pt') or filename.endswith('.pth'):
            file_path = os.path.join(input_folder, filename)
            try:
                # Load the state dictionary
                state_dict = load_state_dict_from_pth(file_path)
                
                # Convert to OrderedDict to maintain order (if needed)
                if isinstance(state_dict, dict):
                    state_dict = OrderedDict(state_dict)

                # Prepare the output file path
                output_file = os.path.join(output_folder, filename.rsplit('.', 1)[0] + '.txt')
                
                # Print the state dictionary information to the output file
                print_state_dict_info(state_dict, output_file)
                
                print(f"Processed: {filename} -> {output_file}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Define input and output folders
input_folder = '/home/torfinhell/Denoiser.cpp/models'
output_folder = '/home/torfinhell/Denoiser.cpp/models/info_model'

# Process the models
process_models(input_folder, output_folder)
