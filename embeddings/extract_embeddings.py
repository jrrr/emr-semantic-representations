#!/usr/bin/env python

import sys
import os
import pickle
import torch
import modeling_readmission # from clinicalBERT
from pytorch_pretrained_bert import BertTokenizer # from the PyTorch implementation of BERT

# path to the pretrained clinicalBERT directory
pretrained_path = 'pretraining'

# cui2term.pkl path, created by gen_cui2term.py
cui2term_path = 'cui2term.pkl'

# output path
output_path = 'bertembeddings.pkl'


def extract_embedding(model, tokenizer, concept):
    tokens = tokenizer.tokenize(concept)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    encoded_layers, _ = model.bert(torch.tensor([token_ids]))
    return encoded_layers

torch.set_grad_enabled(False)

tokenizer = BertTokenizer.from_pretrained(os.path.join(pretrained_path,
    'vocab.txt'))

bert_config = modeling_readmission.BertConfig.from_json_file(
    os.path.join(pretrained_path, 'bert_config.json'))
bert_config.attention_probs_dropout_prob = 0;
bert_config.hidden_dropout_prob = 0;

model = modeling_readmission.BertForSequenceClassification(bert_config, 2)
model.load_state_dict(torch.load(
    os.path.join(pretrained_path, 'pytorch_model.bin'), map_location='cpu'))

with open(cui2term_path, 'rb') as f:
    cui2term = pickle.load(f)

cui2idx = {}
embeddings = torch.zeros([len(cui2term), 768])
for i, (cui, (_, term)) in enumerate(cui2term.items()):
    cui2idx[cui] = i

    encoder_layers = extract_embedding(model, tokenizer, term)
    embedding = (encoder_layers[-4].mean(dim=1) +
                 encoder_layers[-3].mean(dim=1) +
                 encoder_layers[-2].mean(dim=1) +
                 encoder_layers[-1].mean(dim=1))
    embeddings[i,:] = embedding

    if i % 1000 == 0:
        sys.stdout.write('\r%d/%d' % (i, len(cui2term)))
        sys.stdout.flush()
sys.stdout.write('\n')

with open(output_path, 'wb') as f:
    pickle.dump({'cui2idx': cui2idx, 'embeddings': embeddings}, f)
