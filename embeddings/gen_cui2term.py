#!/usr/bin/env python

import pickle
from umls import UMLS

# path to cui2vec_pretrained.csv
cui2vec_path = '../cui2vec_pretrained/cui2vec_pretrained.csv'

# path to the UMLS directory
UMLS_path = '../UMLS'

# path to write cui2term to
output_path = 'cui2term.pkl'


print('loading target cuis...')
target_cuis = set()
with open(cui2vec_path, 'r') as f:
    f.readline() # first line are headings
    for line in f:
        target_cuis.add(line.split(',')[0].strip('"'))

umls = UMLS()
print('loading UMLS...')
unmatched_cuis = umls.load(UMLS_path, target_cuis)
print('matched cuis:', len(umls.cui2term))
print('unmatched cuis:', len(unmatched_cuis))

with open(output_path, 'wb') as f:
    pickle.dump(umls.cui2term, f)
