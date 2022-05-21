import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

class LinuxDataset(Dataset):
    def __init__(self, file_path, seq_length, transform):
        f = open(file_path, 'r').read()
        # Check if pickled file exists
        pickle_filename = 'dataset_tensor.pkl'
        if os.path.isfile(pickle_filename):
            self.file = pickle.load(open(pickle_filename, 'rb'))
        else:
            self.file = torch.tensor([transform(char) for char in f]).cuda()
            pickle.dump(self.file, open(pickle_filename, 'wb+'))
        self.seq_length = seq_length
        self.transform = transform
        
    def __len__(self):
        return len(self.file) - self.seq_length
    def __getitem__(self, idx):
        input = self.file[idx:idx+self.seq_length]
        output = self.file[idx+1:idx+self.seq_length+1]
        return input, output