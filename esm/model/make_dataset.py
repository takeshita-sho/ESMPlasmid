import esm2
import dataset
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
import pickle
sys.path.append('/nfshomes/stakeshi/esm/esm')
import constants
import data
import bz2,json,contextlib

# Create an instance of dataset.
data_dir = '/nfshomes/stakeshi/esm/data'
set = dataset.MetagenomeDataset(data_dir)

torch.save(set, '/nfshomes/stakeshi/esm/esm/model/dataset.pt')