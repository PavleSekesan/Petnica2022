from char_level_lstm import *
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

start_code = open('prompt.txt').read()

model = pickle.load(open('model.pkl', 'rb')).cuda()

seed = torch.tensor([[onehot(char) for char in start_code]]).cuda()
generated = start_code
example_length = 1000
out, hidden = model(seed) 
generated += onehot_reverse(out[0][-1])
for curr_example in range(1, example_length):
    prev_char = onehot_reverse(out[0][-1])
    out, hidden = model(torch.tensor((onehot(prev_char))).reshape(1, 1, -1).cuda(), hidden)
    generated += onehot_reverse(out[0][-1])
f = open('example.txt', 'w+')
f.write(generated)