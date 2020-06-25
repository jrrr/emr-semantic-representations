#!/usr/bin/env python

import os, sys, string, re, csv, xmlrpc.client, pickle
import pandas as pd

import concept_finder
import patients

# where to write the extracted text
output_path = 'data/text_48.pkl'

# use only notes from the first timedelta in ICU (None = all time)
timedelta = pd.to_timedelta('2 days')

# path to MIMIC-III's NOTEEVENTS.csv
noteevents_path = 'mimic-iii-clinical-database-1.4/NOTEEVENTS.csv'


trans = str.maketrans('-/\n', '   ',
    string.punctuation.replace('-', '').replace('/', ''))
def preprocess(text):
    text = text.replace('\r\n', '\n')
    text = re.sub('\\[(.*?)\\]', '', text) # remove deidentified parts
    text = re.sub('--|__|==', '', text)
    sentences = re.split('\. |\.\n|\n\n|: |:\n', text)
    sentences = [sentence.strip().lower().translate(trans)
        for sentence in sentences]
    sentences = [sentence for sentence in sentences if sentence != '']
    return ' '.join(sentences)

admits = dict()
for pat in patients.patients():
    stays_df = pat.get_stays()
    for _, stay in stays_df.iterrows():
        hadm_id = int(stay['HADM_ID'])
        admit_time = pd.to_datetime(stay['ADMITTIME'])
        admits[hadm_id] = admit_time

num_rows = 2083180

df = pd.DataFrame(index=admits.keys(), columns=['TEXT'])
df['TEXT'] = ''
with open(noteevents_path, 'r') as f:
    csvr = csv.DictReader(f)
    for (i_note, row) in enumerate(csvr):
        if i_note % 100 == 0:
            print(f'{i_note}/{num_rows}', flush=True, end='\r')

        if not row['HADM_ID'] or int(row['HADM_ID']) not in admits:
            continue

        hadm_id = int(row['HADM_ID'])
        if timedelta:
            note_time = pd.to_datetime(row['CHARTDATE'])
            if note_time - admits[hadm_id] > timedelta:
                continue

        text = preprocess(row['TEXT'])
        df.loc[hadm_id] = df.loc[hadm_id]['TEXT'] + ' ' + text
    print(f'{num_rows}/{num_rows}')

with open(output_path, 'wb') as f:
    pickle.dump(df, f)
