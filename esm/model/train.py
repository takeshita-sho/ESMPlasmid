import esm2
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
import pickle
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import sys
sys.path.append('/nfshomes/stakeshi/esm/esm')
import constants
import data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
rank = args.local_rank
device = torch.device(f'cuda:{rank}')
#print("Using: " + device)
dna_alphabet = data.Alphabet(constants.dnaseq_toks["toks"])

# Define a custom dataset to load your metagenome assembly data.
class MetagenomeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
        self.data = []
        self.maxlen = 0
        for i in self.data_files:
            print(i)
            for j in self.process_fa_file(i):
                # tokenizes and encodes data
                # need to add masked function and padding function
                length = len(j)
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
        item = self.data[index]
        #label = self.labels[index]
        return item


# Create an instance of dataset.
data_dir = '/nfshomes/stakeshi/esm/data'
dataset = MetagenomeDataset(data_dir)

# Define the training parameters.
batch_size = 4
num_epochs = 10
learning_rate = 0.1

# Create a DataLoader to efficiently load and batch the data.
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the loss function and optimizer.
loss_function = torch.nn.CrossEntropyLoss().to(device) # I should use exponentiated cross entropy loss
model = esm2.ESM2(alphabet=dna_alphabet)
model.to(device)
model = DDP(model, device_ids=[rank],output_device=rank)
total_params = sum(p.numel() for p in model.parameters())
print("Total parameters: ", total_params)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = DistributedOptimizer(optimizer)
#batch_converter = dna_alphabet.get_batch_converter() # use the batch converter to convert tensors
# Set the model to training mode.
model.train()

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}')
    
    for step, sequences in progress_bar:
        # Forward pass
        sequences = sequences.to(device)
        output = model(sequences,repr_layers=[33]) # feed in tokenized sequences, repr layer
        
        logits = output['logits']  # Output logits from the model
        
        # Flatten the logits and target sequences to compute the loss.
        logits = logits.view(-1, logits.size(-1))
        sequences = sequences.view(-1)
        
        # Calculate the loss
        loss = loss_function(logits, sequences)#.to(device)
        loss = torch.exp(loss)#.to(device)
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l = loss.item()
        print(f'Epoch {epoch+1} Step {step+1} - Loss: {l:.4f}')
        total_loss += l
        del sequences; del loss; del l; del logits; del output
        torch.cuda.empty_cache()
    
    # Print average loss for the epoch
    average_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch + 1} - Average Loss: {average_loss:.4f}')

# Save the trained model
if rank == 0:
    torch.save(model.state_dict(), 'esm2_pretrained_mlm.pth')
