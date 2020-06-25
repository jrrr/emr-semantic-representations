#!/usr/bin/env python

import os

class UMLS:
    def __init__(self):
        self.cui2term = {}
        self.ranks = {}
        self.sources = {
            'SNOMEDCT_US',
            'NCI'
        }
        self.priorities = {x: i for (i, x) in enumerate([
            'MTH_OP',
            'MTH_OAP',
            'MTH_PT',
            'OP',
            'OAP',
            'PT'
        ])}

    def load(self, path, target_cuis=None):
        # load data with rankings of preferred terms
        with open(os.path.join(path, 'MRRANK.RRF'), 'r', encoding='utf8') as f:
            for line in f:
                rank, source, term_type, _, _ = line.split('|')
                rank = int(rank)
                self.ranks[(source, term_type)] = rank

        with open(os.path.join(path, 'MRCONSO.RRF'), 'r', encoding='utf8') as f:
            for line in f:
                parts = line.split('|')
                cui = parts[0]
                lang = parts[1]
                source = parts[11]
                term_type = parts[12]
                term = parts[14]

                if lang != 'ENG' or not (source, term_type) in self.ranks:
                    continue

                rank = self.ranks[(source, term_type)]
                if not cui in self.cui2term:
                    if target_cuis:
                        if not cui in target_cuis:
                            continue
                        target_cuis.remove(cui)
                    self.cui2term[cui] = (rank, term)
                elif rank > self.cui2term[cui][0]:
                    self.cui2term[cui] = (rank, term)
                continue

                if (lang != 'ENG' or not source in self.sources or not
                        term_type in self.priorities):
                    continue

                prio = self.priorities[term_type]
                if not cui in self.cui2term:
                    if target_cuis:
                        if not cui in target_cuis:
                            continue
                        target_cuis.remove(cui)
                    self.cui2term[cui] = (prio, term)
                elif prio > self.cui2term[cui][0]:
                    self.cui2term[cui] = (prio, term)

        return target_cuis
