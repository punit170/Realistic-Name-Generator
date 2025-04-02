import torch
import torch.nn as nn
import time
from colorama import Fore, Style

#character indexing
def characterIndexing():
    char_indices = {'X': 0}
    
    char_indices |= {chr(i+96): i for i in range(1, 27)}
    indices_char = {i: char for char, i in char_indices.items()}

    return char_indices, indices_char

#onehotencoding function
def one_hot_encode(sequence, char_indices):
    tensor = torch.zeros(len(sequence), len(char_indices))
    for i, char in enumerate(sequence):
        tensor[i, char_indices[char]] = 1
    return tensor

#LSTM model
class NameGenerator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(NameGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output
    
#function to generate 1 name from a letter
#select_thr is a parameter of tye double. Higher its value, more the algorithm softens
def generate_name_recursive(model, input_seq, char_indices, indices_char, device, select_thr, temperature=1.0):
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
    return generate_name_recursive(model, input_seq, char_indices, indices_char, device, select_thr, temperature)

#function to generate n names, it takes the model and the starting letter
def generate_names(n, model, namePrefix, char_indices, indices_char, device, tolerance=3.0, temperature=1.0):
  names = []
  for i in range(n):
    names.append(generate_name_recursive(model, namePrefix, char_indices, indices_char, device, tolerance, temperature))
  return names

# display names to console
def display_output(namePrefix, count, names):
    # Printing the header
    print(Fore.CYAN + "="*40)
    print(Fore.YELLOW + "Neural Network Based Name Generator")
    print(Fore.GREEN + "=====================================")
    
    # Adding timestamp
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(Fore.MAGENTA + f"Timestamp: {current_time}")
    
    # Displaying the input parameters
    print(Fore.GREEN + f"Initial Name Prefix: {namePrefix}")
    print(Fore.GREEN + f"Number of Names to Generate: {count}")
    
    # Printing the names with better formatting
    print(Fore.CYAN + "\nGenerated Names:\n")
    print(Fore.WHITE + "-"*40)
    
    for i, name in enumerate(names, 1):
        print(Fore.LIGHTYELLOW_EX + f"{i}. {name}")
    
    print(Fore.WHITE + "-"*40)
    print(Fore.CYAN + "="*40)
    print(Style.RESET_ALL)  # Reset the styling after printing