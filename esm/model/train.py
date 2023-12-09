import esm2
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import dataset
import os
import sys
sys.path.append('/nfshomes/stakeshi/esm/esm')
import constants
import data

device = torch.device('cpu')
#print("Using: " + device)
dna_alphabet = data.Alphabet(constants.dnaseq_toks["toks"])

# Create an instance of dataset.
data_dir = '/nfshomes/stakeshi/esm/data'
train_dataset = dataset.MetagenomeDataset(data_dir)

# Define the training parameters.
batch_size = 1
num_epochs = 1
learning_rate = .001

# Create a DataLoader to efficiently load and batch the data.
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Define the loss function and optimizer.
loss_function = torch.nn.CrossEntropyLoss().to(device) # I should use exponentiated cross entropy loss
model = esm2.ESM2(alphabet=dna_alphabet, num_layers=1,embed_dim=2,attention_heads=1)
model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print("Total parameters: ", total_params)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        output = model(sequences,repr_layers=[1]) # feed in tokenized sequences, repr layer
        
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
torch.save(model.state_dict(), 'esm2_pretrained_mlm.pth')
#make sure not NaN values
#next step try training without pre training
#then try to add pre training
#need also non plasmid data - make sure there are no plasmids
#train binary classifier for plasmids - torch.nn.linear then softmax, then use probabilties to determine plasmids
