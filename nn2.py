import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn import metrics

class Net(nn.Module):
    def __init__(self, emb):
        super(Net, self).__init__()
        self.emb = emb.transpose(0, 1)
        self.boc_weights = nn.Parameter(torch.zeros([1, self.emb.shape[1]]))
        self.layer1 = nn.Linear(self.emb.shape[0], 1)

    def forward(self, x):
        x = x * self.boc_weights
        x = torch.mm(self.emb, x.transpose(0, 1)).transpose(0, 1) 
        x = self.layer1(x)
        return x

embed_path = 'embeddings/bertembeddings.pkl'
boc_path = 'data/boc.pkl'
listfile_dir = 'data/readmission'

def load_listfile(path):
    names = []
    labels = []
    with open(path, 'r') as f:
        f.readline() # first line is header
        for line in f:
            line = line.strip()
            parts = line.split(',')
            names.append('_'.join(parts[0].split('_')[0:2]))
            labels.append(int(parts[-1]))
    return pd.DataFrame(index=names, columns=['label'], data=labels, dtype=int)

def load_data():
    with open(boc_path, 'rb') as f:
        boc = pickle.load(f)['df']
    with open(embed_path, 'rb') as f:
        embeddings = pickle.load(f)
        cui2idx = embeddings['cui2idx']
        embeddings = embeddings['embeddings']
    missing_cuis = [cui for cui in boc.columns if cui not in cui2idx]
    boc.drop(columns=missing_cuis, inplace=True)
    emb_matrix = torch.stack([embeddings[cui2idx[cui]] for cui in boc.columns])

    train_y = load_listfile(os.path.join(listfile_dir, 'train_listfile.csv'))
    val_y = load_listfile(os.path.join(listfile_dir, 'val_listfile.csv'))
    test_y = load_listfile(os.path.join(listfile_dir, 'test_listfile.csv'))
    train_X = boc.loc[boc.index.intersection(train_y.index)]
    train_y = train_y.loc[train_X.index]
    val_X = boc.loc[boc.index.intersection(val_y.index)]
    val_y = val_y.loc[val_X.index]
    test_X = boc.loc[boc.index.intersection(test_y.index)]
    test_y = test_y.loc[test_X.index]

    train_X.sort_index(inplace=True)
    train_y.sort_index(inplace=True)
    val_X.sort_index(inplace=True)
    val_y.sort_index(inplace=True)
    test_X.sort_index(inplace=True)
    test_y.sort_index(inplace=True)

    train_X = train_X.sparse.to_dense()
    train_X.fillna(value=0, inplace=True)
    val_X = val_X.sparse.to_dense().fillna(value=0)
    val_X.fillna(value=0, inplace=True)
    test_X = test_X.sparse.to_dense().fillna(value=0)
    test_X.fillna(value=0, inplace=True)

    return (torch.tensor(train_X.values).float(),
            torch.tensor(train_y.values).float().squeeze(),
            torch.tensor(val_X.values).float(),
            torch.tensor(val_y.values).float().squeeze(),
            torch.tensor(test_X.values).float(),
            torch.tensor(test_y.values).float().squeeze(),
            emb_matrix)

train_X, train_y, val_X, val_y, test_X, test_y, emb_matrix = load_data()

net = Net(emb_matrix)
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)

pos_freq = train_y.sum() / len(train_X)
neg_freq = 1 - pos_freq

roc = []
for iEpoch in range(20):
    print()
    print('epoch', iEpoch+1)
    # train
    X = train_X
    y = train_y
    optimizer.zero_grad()
    output = net(X).view([-1])
    weights = torch.zeros(y.shape)
    weights[y == 1] = 1 / pos_freq
    weights[y == 0] = 1 / neg_freq
    loss = F.binary_cross_entropy_with_logits(output, y, weight=weights)
    loss.backward()
    optimizer.step()
    print('train loss:', loss.item())

    # validate
    with torch.no_grad():
        output = net(val_X).view([-1])
        prob = torch.sigmoid(output)
        fpr, tpr, _ = metrics.roc_curve(val_y, prob)
        roc_auc = metrics.auc(fpr, tpr)
        loss = F.binary_cross_entropy_with_logits(output, val_y.float())
        print('validation loss:', loss.item())
        print('validation ROC AUC:', roc_auc)
        print('sum:', net.boc_weights.sum())
        if len(roc) == 0 or roc_auc > max(roc):
            best_net = Net(emb_matrix)
            best_net.load_state_dict(net.state_dict())
        roc.append(roc_auc)

print()
print('best ROC AUC:', max(roc))

with torch.no_grad():
    val_output = best_net(val_X).view([-1])
    val_prob = torch.sigmoid(val_output)
    val_pred = val_prob > 0.5
    fpr, tpr, _ = metrics.roc_curve(val_y, val_prob)
    roc_auc = metrics.auc(fpr, tpr)

    test_output = best_net(test_X).view([-1])
    test_prob = torch.sigmoid(test_output)
    test_pred = test_prob > 0.5
    fpr, tpr, _ = metrics.roc_curve(test_y, test_prob)
    test_roc_auc = metrics.auc(fpr, tpr)
print('\\begin{tabular}{@{}c@{}} %.3f \\\\ \\textbf{%.3f} \end{tabular} &' %
    (metrics.accuracy_score(val_y, val_pred),
     metrics.accuracy_score(test_y, test_pred)))
print('\\begin{tabular}{@{}c@{}} %.3f \\\\ \\textbf{%.3f} \end{tabular} &' %
    (metrics.precision_score(val_y, val_pred),
     metrics.precision_score(test_y, test_pred)))
print('\\begin{tabular}{@{}c@{}} %.3f \\\\ \\textbf{%.3f} \end{tabular} &' %
    (metrics.recall_score(val_y, val_pred),
     metrics.recall_score(test_y, test_pred)))
print('\\begin{tabular}{@{}c@{}} %.3f \\\\ \\textbf{%.3f} \end{tabular} &' %
    (metrics.f1_score(val_y, val_pred),
     metrics.f1_score(test_y, test_pred)))
print('\\begin{tabular}{@{}c@{}} %.3f \\\\ \\textbf{%.3f} \end{tabular} \\\\ \\hline' %
    (roc_auc, test_roc_auc))
