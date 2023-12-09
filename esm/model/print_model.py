import esm2
import torch
import data
import os
import sys
sys.path.append('/nfshomes/stakeshi/esm/esm')
import constants
import data

dna_alphabet = data.Alphabet(constants.dnaseq_toks["toks"])
model = esm2.ESM2(alphabet=dna_alphabet, num_layers=1,embed_dim=2,attention_heads=1)
model.load_state_dict(torch.load('/nfshomes/stakeshi/esm/esm/model/esm2_pretrained_mlm.pth'))
for param in model.parameters():
    print(param.data)