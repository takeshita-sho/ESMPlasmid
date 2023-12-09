import esm2
import torch
import torch.nn as nn
from torch.nn import functional as F
#from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.distributed.optim import DistributedOptimizer
import pickle
#import torch.distributed.rpc as rpc
#from torch.distributed.rpc import RRef
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import sys
sys.path.append('/nfshomes/stakeshi/esm/esm')
import constants
import data

dna_alphabet = data.Alphabet(constants.dnaseq_toks["toks"])

# Define a custom dataset to load your metagenome assembly data.
class MetagenomeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        self.data = []
        self.maxlen = 200 #maybe change this
        for i in self.data_files:
            print(i)
            for j in self.process_fa_file(i):
                # tokenizes and encodes data
                # need to add masked function and padding function
                length = len(j)
                if length >= 200:
                    self.data.append(j)
                    if length > self.maxlen:
                        self.maxlen = length

                # I need to store as strings first and get longest sequence
                # then create second loop that adds padding and returns encoding
                
        for i in range(len(self.data)):
            seq = self.data[i]
            length = len(seq)
            if length < self.maxlen:

                pad = self.maxlen-length
                padding = "<pad>"*pad
                
                seq = seq+padding
                
            # Tokenize the DNA sequence
            tokenized_sequence = dna_alphabet.encode(seq)

            # Convert the token IDs to a tensor
            self.data[i] = torch.tensor(tokenized_sequence, dtype=torch.int64)
        
    def process_fa_file(self,file_path):
        sequences = []
        with open(file_path, 'r') as file:
            current_sequence = ''
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if current_sequence:
                        sequences.append(current_sequence)
                        current_sequence = ''
                else:
                    current_sequence += line
            if current_sequence:
                sequences.append(current_sequence)
        return sequences

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        #label = self.labels[index]
        return self.data[index][0:100]