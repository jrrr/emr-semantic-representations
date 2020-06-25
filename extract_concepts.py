#!/usr/bin/env python

import os, sys, string, re, csv, xmlrpc.client, pickle, signal
import pandas as pd

import patients
import concept_finder

# path to temporary progress tracking file
progress_path = 'data/mimic/extract_concepts_progress'

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
    return sentences


hadm_id2path = dict()
for ep in patients.episodes():
    ep_df = ep.get_info()
    if len(ep_df) == 0:
        continue
    stay_df = ep.get_stay()

    hadm_id = stay_df['HADM_ID']
    hadm_id2path[hadm_id] = os.path.join(ep.patient.directory,
        'episode' + ep.number + '_noteconcepts.csv')

num_rows = 2083180

# use a file to track the progress since this takes some time
progf = open(progress_path, 'a+')
progf.seek(0)
try:
    done_notes = int(progf.read())
except ValueError:
    done_notes = 0

cf = concept_finder.concept_finder()
with open(noteevents_path, 'r') as f:
    csvr = csv.DictReader(f)

    for (i_note, row) in enumerate(csvr):
        if i_note < done_notes:
            continue # skip already done notes

        if i_note % 100 == 0:
            print(f'{i_note}/{num_rows}', flush=True, end='\r')

        if not row['HADM_ID'] or int(row['HADM_ID']) not in hadm_id2path:
            continue

        sentences = preprocess(row['TEXT'])
        cuis = cf.extract_concepts(sentences)

        # Pause SIGINT (KeyboardInterrupt) while writing the data to avoid
        # corrupting anything. Almost all time should be spent in
        # cf.extract_concepts, but you never know.
        oldhandler = signal.signal(signal.SIGINT, signal.SIG_IGN)

        path = hadm_id2path[int(row['HADM_ID'])]
        f_existed = os.path.isfile(path)
        with open(path, 'a') as epf:
            writer = csv.DictWriter(epf,
                fieldnames=['CHARTDATE', 'CONCEPTS'])
            if not f_existed:
                writer.writeheader()
            writer.writerow({
                'CHARTDATE': row['CHARTDATE'],
                'CONCEPTS': ' '.join(cuis)
            })

        progf.truncate(0)
        progf.write(str(i_note + 1))

        # Resume SIGINT
        signal.signal(signal.SIGINT, oldhandler)

    print(f'{num_rows}/{num_rows}')
