#!/usr/bin/env python

import sys
import pickle
import argparse
import pandas as pd
import numpy as np

import patients

parser = argparse.ArgumentParser(
    description='Create a bag of concepts for each episode'
)
parser.add_argument('out', type=str, help='Path to output file')
parser.add_argument('--first_24', action='store_true',
    help='Only use notes from the first (approximately) 24 hours')
parser.add_argument('--first_48', action='store_true',
    help='Only use notes from the first (approximately) 48 hours')
parser.add_argument('--vocab_out', type=str, help='Path to store vocabulary')
args, _ = parser.parse_known_args()

timedelta = None
if args.first_48:
    timedelta = pd.to_timedelta('2 days')
if args.first_24:
    timedelta = pd.to_timedelta('1 days')

vocabulary = set()
for ep in patients.episodes():
    for concepts in ep.get_concepts(timedelta):
        for concept in concepts:
            if not concept.startswith('!'):
                vocabulary.add(concept)
print('vocabulary size:', len(vocabulary))
vocabulary = list(vocabulary)
if args.vocab_out:
    with open(args.vocab_out, 'wb') as f:
        pickle.dump(vocabulary, f)
concept_idx = dict([(concept, index) for (index, concept) in enumerate(vocabulary)])

names = []
X = []
for i, ep in enumerate(patients.episodes()):
    if ep.get_noteconcepts().empty:
        continue

    x = np.zeros(len(vocabulary), dtype=int)
    for concepts in ep.get_concepts(timedelta):
        for concept in concepts:
            if not concept.startswith('!'):
                x[concept_idx[concept]] = 1
    X.append(x)
    names.append(ep.patient.id + '_episode' + ep.number)

df = pd.DataFrame(index=names, columns=vocabulary, data=np.stack(X))
df = df.astype(pd.SparseDtype(int, 0))
print(df.shape)

with open(args.out, 'wb') as f:
    pickle.dump({'should_standardize': False, 'should_impute': False,
        'df': df}, f)
