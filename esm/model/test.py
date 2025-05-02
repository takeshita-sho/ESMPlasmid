import torch
import sys
sys.path.append('/nfshomes/stakeshi/esm/esm')
import constants
import data

# Define the DNA sequence string
#have to insert certain amount of padding for max length string
#need to also randomly mask data
dna_sequence = "AGCTTAGGCATCGTACCGA<pad>"

# Create an instance of the DNA alphabet
dna_alphabet = data.Alphabet(constants.dnaseq_toks["toks"])

# Tokenize the DNA sequence
tokenized_sequence = dna_alphabet.encode(dna_sequence)

# Convert the token IDs to a tensor
token_tensor = torch.tensor(tokenized_sequence, dtype=torch.int64)

# Print the tokenized sequence
print("DNA Alphabet: ", dna_alphabet)
print("Tokenized DNA Sequence:", tokenized_sequence)
print("Token Tensor:", token_tensor)