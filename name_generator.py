import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#SET INITIAL LETTER SEQUENCE
initial_letter = 'a'
#SET tolerance- higher value => name generator softens
tolerance = 3.0

#SET FOLDER PATH where trained model file, input text file,  
path = "./"

#uncomment, if using colab
'''
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
path = "/content/drive/My Drive/CS559/hw7/"
'''

#MODEL FILE NAME
modelFileName = "0702-662147231-Malpani.pth"
modelSavePath = path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputFilePath = f"{path}names.txt"


with open(inputFilePath, 'r') as inputFile:
    inputNames = inputFile.readlines()
#Read names from input file
with open(inputFilePath, 'r') as file:
    names = file.read().lower().split('\n')

# Adding "end of name" character - 'X' to each name to make them 11 characters long
names = [name.ljust(11, 'X') for name in names]
# print(names)

# Create character indices
chars = sorted(list(set(''.join(names))))
char_indices = {char: i for i, char in enumerate(chars)}
indices_char = {i: char for i, char in enumerate(chars)}

#onehotencoding function
def one_hot_encode(sequence, char_indices):
    tensor = torch.zeros(len(sequence), len(char_indices))
    for i, char in enumerate(sequence):
        tensor[i, char_indices[char]] = 1
    return tensor


#lstm model
class NameGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NameGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

#function to generate 1 name from a letter
#select_thr is a parameter of tye double. Higher its value, more the algorithm softens
def generate_name_recursive(model, input_seq, char_indices, indices_char, select_thr, temperature=1.0):
    if input_seq[-1] == 'X':
        return input_seq[0:-1]  # Stop recursion when 'X' is encountered

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to compute gradients during generation
        modelInput = one_hot_encode(input_seq, char_indices)
        modelInput = modelInput.unsqueeze(0).to(device)  # Add batch dimension and move to GPU
        output = model(modelInput)
        outputOfLastLetter = output[0][-1]
        # print(outputOfLastLetter)
        include_above = outputOfLastLetter.max().item() - select_thr
        filtered_indices = torch.nonzero(outputOfLastLetter >= include_above).squeeze()
        top_indices = filtered_indices

        if top_indices.dim() <= 0:
          next_index = top_indices.item()
        else:
          next_index = top_indices[torch.randint(len(top_indices), (1,)).item()].item()
        
        next_char = indices_char[next_index]
        input_seq += next_char 

    #setting the model back to training mode
    model.train()
    #recursively calling the function with the new sequence
    return generate_name_recursive(model, input_seq, char_indices, indices_char, select_thr, temperature)

#function to generate 20 names, it takes the model and the starting letter
def generate_20_names(model, initial_letter, char_indices=char_indices, indices_char=indices_char, tolerance=3.0, temperature=1.0):
  names = []
  print('\nGenerating names:')
  for i in range(20):
    names.append(generate_name_recursive(model, initial_letter, char_indices, indices_char, tolerance, temperature))
  return names


# Instantiate the model
#params with same values as when model was trained
input_size = len(chars)
hidden_size = 256
output_size = len(chars)

loaded_model = NameGenerator(input_size, hidden_size, output_size)
loaded_model.to(device)

# Load the trained model
loaded_model.load_state_dict(torch.load(f'{modelSavePath}{modelFileName}', map_location=device))
loaded_model.eval()

# Print generated names after training
names = generate_20_names(model=loaded_model, initial_letter=initial_letter, tolerance = tolerance)
print(names)

