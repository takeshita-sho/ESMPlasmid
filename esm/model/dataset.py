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
kmer_size = 1024

# Define a custom dataset to load your metagenome assembly data.
class MetagenomeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        self.data = []
        self.maxlen = kmer_size
        for i in self.data_files:
            print(i)
            for j in self.process_fa_file(i):
                # tokenizes and encodes data
                length = len(j)
                if length >= kmer_size:
                    k = 0
                    while k < length:
                        # To add later - BOS and EOS tokens to indicated start and end of a real protein
                        if k+kmer_size > length:
                            seq = j[k:]
                            pad = self.maxlen-len(seq)
                            padding = "<pad>"*pad
                            seq = seq+padding
                            tok_seq = dna_alphabet.encode(seq)
                            self.data.append(torch.tensor(tok_seq, dtype=torch.int64))
                            break
                        else:
                            tok_seq = dna_alphabet.encode(j[k:k+kmer_size])
                            self.data.append(torch.tensor(tok_seq, dtype=torch.int64))
                        k += 500
                elif length == kmer_size:
                    tok_seq = dna_alphabet.encode(j)
                    self.data.append(torch.tensor(tok_seq, dtype=torch.int64))
                else:
                    pad = self.maxlen-length
                    padding = "<pad>"*pad
                    seq = j+padding
                    tok_seq = dna_alphabet.encode(seq)
                    self.data.append(torch.tensor(tok_seq, dtype=torch.int64))
                # then create second loop that adds padding and returns encoding
        '''       
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
        '''
        
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
        return self.data[index]