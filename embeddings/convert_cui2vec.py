#!/usr/bin/env python

import pickle
import torch

# path to cio2vec_pretrained.csv
cui2vec_path = '../cui2vec_pretrained/cui2vec_pretrained.csv'

# output path
output_path = 'cui2vec.pkl'


cui2idx = {}
embeddings = []
with open(cui2vec_path, 'r', encoding='utf8') as f:
    f.readline() # first line is headings
    for idx, line in enumerate(f):
        parts = line.split(',')
        cui = parts[0].strip('"')

        cui2idx[cui] = idx
        embeddings.append([float(x) for x in parts[1:]])

embeddings = torch.tensor(embeddings)
print(embeddings.size())

with open(output_path, 'wb') as f:
    pickle.dump({'cui2idx': cui2idx, 'embeddings': embeddings}, f)
