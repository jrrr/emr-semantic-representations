import os
import pandas as pd

# path to MIMIC-III as split into episodes by mimic3-benchmarks
patients_dir = 'data/mimic'

class Patient:
    def __init__(self, directory, id):
        self.directory = directory
        self.id = id

    def get_stays(self):
        return pd.read_csv(os.path.join(self.directory, 'stays.csv'))

class Episode:
    def __init__(self, patient, number):
        self.patient = patient
        self.number = number

    def get_info(self):
        return pd.read_csv(os.path.join(self.patient.directory,
            'episode' + self.number + '.csv'))

    def get_timeseries(self):
        return pd.read_csv(os.path.join(self.patient.directory,
            'episode' + self.number + '_timeseries.csv'))

    def get_noteconcepts(self):
        try:
            return pd.read_csv(os.path.join(self.patient.directory,
                'episode' + self.number + '_noteconcepts.csv'))
        except FileNotFoundError:
            return pd.DataFrame()

    def get_concepts(self, timedelta=None):
        nc_df = self.get_noteconcepts()
        if not nc_df.empty:
            admit_time = pd.to_datetime(self.get_stay()['ADMITTIME'])
            for _, row in nc_df.iterrows():
                concepts = row['CONCEPTS']
                chartdate = pd.to_datetime(row['CHARTDATE'])
                if pd.isna(concepts):
                    continue
                if timedelta and chartdate - admit_time > timedelta:
                    continue

                concept_set = set()
                for concept in filter(lambda c: c[0] != '!', concepts.split(' ')):
                    concept_set.add(concept)
                yield concept_set

    def get_stay(self):
        stays_df = self.patient.get_stays()
        info_df = self.get_info()
        stay_df = stays_df.loc[stays_df['ICUSTAY_ID'] == info_df.iloc[0]['Icustay']]
        assert(len(stay_df) == 1)
        return stay_df.iloc[0]

def patients(partition=None):
    if partition:
        pdir = os.path.join(patients_dir, partition)
        patdirs = [os.path.join(pdir, p) for p in
            filter(str.isdigit, os.listdir(pdir))]
    else:
        train_dir = os.path.join(patients_dir, 'train')
        patdirs = [os.path.join(train_dir, p) for p in
            filter(str.isdigit, os.listdir(train_dir))]
        test_dir = os.path.join(patients_dir, 'test')
        patdirs += [os.path.join(test_dir, p) for p in
            filter(str.isdigit, os.listdir(test_dir))]

    for patdir in patdirs:
        yield Patient(patdir, os.path.basename(patdir))

def episodes(partition=None):
    for pat in patients(partition):
        ts_files = list(filter(lambda x: x.endswith('_timeseries.csv'),
            os.listdir(pat.directory)))
        eps = \
            [ts[7] + (ts[8] if ts[8].isdigit() else '') for ts in ts_files]

        for ep in eps:
            yield Episode(pat, ep)
