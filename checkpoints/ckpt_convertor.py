import argparse
from collections import OrderedDict
import pickle
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, required=True)
args = parser.parse_args()
assert args.file.endswith('.pkl'), "the file should be in .pkl format" 
with open(args.file, 'rb') as f:
    ckpt = pickle.load(f)

state_dict = OrderedDict()
for k, v in ckpt.items():
    state_dict[k] = torch.from_numpy(v)

torch.save(state_dict, args.file[:-4]+'.pth')