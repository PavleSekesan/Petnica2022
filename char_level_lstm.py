import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from custom_datasets import LinuxDataset
import tqdm
import torch.nn.functional as F
import pickle

hidden_size = 420
sequence_length = 30
num_layers = 2
input_size = 103
num_epochs = 10
batch_size = 1024
learning_rate = 1e-3

#char_to_idx = {'\t': 0, '\n': 1, ' ': 2, '!': 3, '"': 4, '#': 5, '$': 6, '%': 7, '&': 8, "'": 9, '(': 10, ')': 11, '*': 12, '+': 13, ',': 14, '-': 15, '.': 16, '/': 17, '0': 18, '1': 19, '2': 20, '3': 21, '4': 22, '5': 23, '6': 24, '7': 25, '8': 26, '9': 27, ':': 28, ';': 29, '<': 30, '=': 31, '>': 32, '?': 33, '@': 34, 'A': 35, 'B': 36, 'C': 37, 'D': 38, 'E': 39, 'F': 40, 'G': 41, 'H': 42, 'I': 43, 'J': 44, 'K': 45, 'L': 46, 'M': 47, 'N': 48, 'O': 49, 'P': 50, 'Q': 51, 'R': 52, 'S': 53, 'T': 54, 'U': 55, 'V': 56, 'W': 57, 'X': 58, 'Y': 59, 'Z': 60, '[': 61, '\\': 62, ']': 63, '^': 64, '_': 65, '`': 66, 'a': 67, 'b': 68, 'c': 69, 'd': 70, 'e': 71, 'f': 72, 'g': 73, 'h': 74, 'i': 75, 'j': 76, 'k': 77, 'l': 78, 'm': 79, 'n': 80, 'o': 81, 'p': 82, 'q': 83, 'r': 84, 's': 85, 't': 86, 'u': 87, 'v': 88, 'w': 89, 'x': 90, 'y': 91, 'z': 92, '{': 93, '|': 94, '}': 95, '~': 96, '¥': 97, '©': 98, 'Â': 99, 'Ã': 100, 'ƒ': 101, '‚': 102}
char_to_idx = {'\t': 0, '\n': 1, ' ': 2, '!': 3, '"': 4, '#': 5, '$': 6, '%': 7, '&': 8, "'": 9, '(': 10, ')': 11, '*': 12, '+': 13, ',': 14, '-': 15, '.': 16, '/': 17, '0': 18, '1': 19, '2': 20, '3': 21, '4': 22, '5': 23, '6': 24, '7': 25, '8': 26, '9': 27, ':': 28, ';': 29, '<': 30, '=': 31, '>': 32, '?': 33, '@': 34, 'A': 35, 'B': 36, 'C': 37, 'D': 38, 'E': 39, 'F': 40, 'G': 41, 'H': 42, 'I': 43, 'J': 44, 'K': 45, 'L': 46, 'M': 47, 'N': 48, 'O': 49, 'P': 50, 'Q': 51, 'R': 52, 'S': 53, 'T': 54, 'U': 55, 'V': 56, 'W': 57, 'X': 58, 'Y': 59, 'Z': 60, '[': 61, '\\': 62, ']': 63, '^': 64, '_': 65, 'a': 66, 'b': 67, 'c': 68, 'd': 69, 'e': 70, 'f': 71, 'g': 72, 'h': 73, 'i': 74, 'j': 75, 'k': 76, 'l': 77, 'm': 78, 'n': 79, 'o': 80, 'p': 81, 'q': 82, 'r': 83, 's': 84, 't': 85, 'u': 86, 'v': 87, 'w': 88, 'x': 89, 'y': 90, 'z': 91, '{': 92, '|': 93, '}': 94, '~': 95, '»': 96, '¿': 97, 'ï': 98}
idx_to_char = {i:char for char, i in char_to_idx.items()}

def onehot(char):
    vec = np.zeros(input_size, dtype=np.float32)
    vec[char_to_idx[char]] = 1
    return vec

def onehot_reverse(vec):
    idx = vec.argmax().item()
    return idx_to_char[idx]


train_data = LinuxDataset('data/cp-problems.txt', sequence_length, onehot)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

class CodeGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(CodeGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)
    def forward(self, input, hidden=None):
        input = input.cuda()
        if hidden is None:
            out, hidden = self.lstm(input)
        else:
            out, hidden = self.lstm(input, hidden)
        out = self.linear(out)
        out = F.log_softmax(out, dim=2)
        return out, hidden

def generate_sample(epoch, model):
    start_code = open('prompt.txt','r').read()
    seed = torch.tensor([[onehot(char) for char in start_code]]).cuda()
    generated = start_code
    example_length = 1000
    out, hidden = model(seed) 
    generated += onehot_reverse(out[0][-1])
    for curr_example in range(1, example_length):
        prev_char = onehot_reverse(out[0][-1])
        out, hidden = model(torch.tensor((onehot(prev_char))).reshape(1, 1, -1).cuda(), hidden)
        generated += onehot_reverse(out[0][-1])
    example_file = open(f'example_{epoch}.txt', 'w+')
    example_file.write(generated)
    example_file.close()

def main():
    model = CodeGenerator(input_size, hidden_size, num_layers).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate)

    msg = ""
    losses = []
    accs = []
    examples = []
    for epoch in range(num_epochs):
        bar = tqdm.tqdm(total=len(train_data))
        for i, (x, y) in enumerate(train_dataloader):
            examples.append([onehot_reverse(x[0][i]) for i in range(x.shape[1])])
            x = x.cuda()
            y = y.cuda()
            pred, _ = model(x)
            pred = pred.view(-1, input_size)
            y = y.view(-1, input_size)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (pred.argmax(1) == y.argmax(1)).sum().item() / (batch_size*sequence_length)
            accs.append(acc)
            losses.append(loss.item())
            running_acc = np.mean(accs[-100:])
            running_loss = np.mean(losses[-100:])
            bar.set_description(f"Loss: {running_loss:.3f}, Acc: {running_acc:.3f}")
            bar.update(batch_size)
            
        bar.close()

        with torch.no_grad():
            generate_sample(epoch, model)
        
        pickle.dump(model.cpu(), open('model.pkl', 'wb+'))
        model.cuda()
