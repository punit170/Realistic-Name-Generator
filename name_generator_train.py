import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

path = "./"

#uncomment if using colab
'''
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
path = "/content/drive/My Drive/CS559/hw7/"
'''
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


#Helper Functions
def oneHotDecode(encodedLetter: torch.Tensor):
    assert(encodedLetter.shape == (27,))
    # index = np.where(encodedLetter == 1)[0][0]
    index = torch.nonzero(encodedLetter).item()
    if(index == 0):
        return "eon"
    else:
        return chr(index+97 - 1)

def formatName(name: str):
    def oneHotEncode(letters: str):
        output = torch.zeros(size=(1, 27), dtype=torch.float)
        if(len(letters) == 3):
            assert(letters.lower() == "eon")
            output[0][0] = 1
            return output
        else:
            assert(len(letters) == 1)
            output[0][ord(letters.lower())-97+1] = 1
            return output
    assert(len(name)<=11)
    outputName = torch.empty(size=(11,27), dtype=torch.float)
    lastIndex = 0
    for i,letter in enumerate(name):
        outputName[i] = oneHotEncode(letter)
        lastIndex = i
    leftSize = 11 - len(name)
    for i in range(leftSize):
        outputName[lastIndex + 1 + i] = oneHotEncode('eon')
    return outputName

def deformatName(encodedName: torch.Tensor):
    assert(encodedName.shape == (11,27))
    name = ""
    for encodedLetter in encodedName:
        decodedLetter = oneHotDecode(encodedLetter)
        if decodedLetter == "eon":
             break
        else:
            name = name + decodedLetter
    return name

#print(oneHotEncode('a'))
# print(formatName("brycen")[0])

trainingSet = list()
for name in inputNames:
    name=name.strip()
    trainable = (formatName(name), formatName(name[1:]))
    trainingSet.append(trainable)
    
#Creating Training Set    
trainingSet = [(x.to(device), y.to(device)) for x, y in trainingSet]

# print(deformatName(trainingSet[0][1]))
# print(trainingSet[0][0].unsqueeze(0).shape)


# Create character indices
chars = sorted(list(set(''.join(names))))
char_indices = {char: i for i, char in enumerate(chars)}
indices_char = {i: char for i, char in enumerate(chars)}

#preparing input data
#max length of input sequence
maxlen = 11
# Step to move the sliding window  
step = 3     

sentences = []
next_chars = []
for name in names:
    for i in range(0, len(name) - maxlen + 1, step):
        input_sequence = name[i:i + maxlen]
        output_sequence = name[i + 1:i + maxlen + 1]

        sentences.append(input_sequence)
        next_chars.append(output_sequence)
        
#adding "EON" characters to make each name 11 characters long
next_chars = [name.ljust(11, 'X') for name in next_chars]
print(sentences)
print(next_chars)

# One Hot encoding
def one_hot_encode(sequence, char_indices):
    tensor = torch.zeros(len(sequence), len(char_indices))
    for i, char in enumerate(sequence):
        tensor[i, char_indices[char]] = 1
    return tensor

x = torch.stack([one_hot_encode(seq, char_indices) for seq in sentences])
y = torch.stack([one_hot_encode(seq, char_indices) for seq in next_chars])
# print(x[0])
# print(y[0])

#Definining a simple LSTM model
class NameGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NameGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

#train function
def train_model_damping(model, optimizer, criterion, x, y, batch_size=1, num_epochs=10, damping_factor=0.9):
    prev_loss = float('inf')
    losses = []
    for epoch in range(num_epochs):
        for i in range(0, len(sentences), batch_size):
            input_seq_batch = x[i:i+batch_size].to(device)
            target_batch = y[i:i+batch_size].to(device)

            optimizer.zero_grad()
            output = model(input_seq_batch)
            loss = criterion(output.permute(0, 2, 1), target_batch.argmax(dim=2))
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
        losses.append(loss.item())
        
        #adjusting the learning rate based on the change in loss
        if prev_loss < loss.item():
            for param_group in optimizer.param_groups:
                param_group['lr'] *= damping_factor
                #print(f'Learning rate adjusted to: {param_group["lr"]}')

        prev_loss = loss.item()
    return losses
        
#instantiating the model and definining loss and optimizer
input_size = len(chars)
hidden_size = 256
output_size = len(chars)

model = NameGenerator(input_size, hidden_size, output_size)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#moving the data to device    
model.to(device)
x = x.to(device)
y = y.to(device)    

#training
losses = train_model_damping(model, optimizer, criterion, x, y, batch_size=1, num_epochs=100, damping_factor=0.9)

# Plot the loss versus epochs
plt.plot(range(1, len(losses) + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss versus Epochs')
plt.show()

#saving the trained model
torch.save(model.state_dict(), f'{modelSavePath}{modelFileName}')



