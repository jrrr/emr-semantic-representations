#!/usr/bin/env python

import sys
import pickle
import argparse
import pandas as pd
import numpy as np
import torch

import patients

parser = argparse.ArgumentParser(
    description='Embed concepts found in notes for each episode'
)
parser.add_argument('out', type=str, help='Path to output file')
parser.add_argument('embeddings', type=str, help='Path to pickled embeddings')
parser.add_argument('--first_24', action='store_true',
    help='Only use notes from the (approximately) first 24 hours')
parser.add_argument('--first_48', action='store_true',
    help='Only use notes from the (approximately) first 48 hours')
args, _ = parser.parse_known_args()

timedelta = None
if args.first_48:
    timedelta = pd.to_timedelta('2 days')
if args.first_24:
    timedelta = pd.to_timedelta('1 days')

print('loading embeddings...')
with open(args.embeddings, 'rb') as f:
    embeddings = pickle.load(f)
    cui2idx = embeddings['cui2idx']
    embeddings = embeddings['embeddings']

print('embedding concepts...')
names = []
epvecs = []
for ep in patients.episodes():
    nc_df = ep.get_noteconcepts()
    if nc_df.empty:
        continue

    stay_df = ep.get_stay()
    admit_time = pd.to_datetime(stay_df['ADMITTIME'])

    concept_embeddings = []
    for _, row in nc_df.iterrows():
        if pd.isna(row['CONCEPTS']):
            continue
        if timedelta:
            note_time = pd.to_datetime(row['CHARTDATE'])
            if note_time - admit_time > timedelta:
                continue

        concepts = set(row['CONCEPTS'].split(' '))
        concept_embeddings += [embeddings[cui2idx[cui]] for cui in concepts if
                    cui[0] != '!' and cui in cui2idx]

    if len(concept_embeddings) == 0:
        continue

    vecs = torch.stack(concept_embeddings)
    epvecs.append(torch.cat([
        vecs.min(dim=0).values,
        vecs.max(dim=0).values,
        vecs.mean(dim=0)
    ]))
    names.append(ep.patient.id + '_episode' + ep.number)

print('building data frame...')
columns = ['emb_' + str(i) for i in range(len(embeddings[0])*3)]
df = pd.DataFrame(index=names, columns=columns, data=np.stack(epvecs))
print(df.shape)

with open(args.out, 'wb') as f:
    pickle.dump({'should_standardize': True, 'should_impute': False,
        'df': df}, f)
