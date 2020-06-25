#!/usr/bin/env python

import sys, os, pickle
import pandas as pd
import numpy as np
from sklearn import linear_model, ensemble, metrics, impute, preprocessing

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

if len(sys.argv) < 3:
    print('usage: {} <listfile_dir> <feature_files...>'.format(sys.argv[0]))
    quit()
listfile_dir = sys.argv[1]
feature_files = sys.argv[2:]

print('loading features...')
features = []
for feature_file in feature_files:
    with open(feature_file, 'rb') as f:
        features.append(pickle.load(f))

print('processing listfiles...')
df_train = load_listfile(os.path.join(listfile_dir, 'train_listfile.csv'))
df_val = load_listfile(os.path.join(listfile_dir, 'val_listfile.csv'))
df_test = load_listfile(os.path.join(listfile_dir, 'test_listfile.csv'))
print(df_train.shape, df_val.shape, df_test.shape)
for feature in features:
    feature['df_train'] = feature['df'].loc[
        feature['df'].index.intersection(df_train.index)]
    feature['df_val'] = feature['df'].loc[
        feature['df'].index.intersection(df_val.index)]
    feature['df_test'] = feature['df'].loc[
        feature['df'].index.intersection(df_test.index)]
    del feature['df']
    print(feature['df_train'].shape, feature['df_val'].shape, feature['df_test'].shape)

print('imputing values...')
for feature in features:
    if feature['should_impute']:
        imputer = impute.SimpleImputer()
        imputer.fit(feature['df_train'])
        feature['df_train'][feature['df_train'].columns] = \
            imputer.transform(feature['df_train'][feature['df_train'].columns])
        feature['df_val'][feature['df_val'].columns] = \
            imputer.transform(feature['df_val'][feature['df_val'].columns])
        feature['df_test'][feature['df_test'].columns] = \
            imputer.transform(feature['df_test'][feature['df_test'].columns])

print('standardizing values...')
for feature in features:
    if feature['should_standardize']:
        scaler = preprocessing.StandardScaler()
        scaler.fit(feature['df_train'])
        std = 0.316
        #std = 0.1
        #std = 1
        feature['df_train'][feature['df_train'].columns] = \
            scaler.transform(feature['df_train'][feature['df_train'].columns])*std
        feature['df_val'][feature['df_val'].columns] = \
            scaler.transform(feature['df_val'][feature['df_val'].columns])*std
        feature['df_test'][feature['df_test'].columns] = \
            scaler.transform(feature['df_test'][feature['df_test'].columns])*std

print('concatenating features...')
df_train = pd.concat([df_train] + [feature['df_train'] for feature in features],
    axis=1, join='inner')
df_val = pd.concat([df_val] + [feature['df_val'] for feature in features],
    axis=1, join='inner')
df_test = pd.concat([df_test] + [feature['df_test'] for feature in features],
    axis=1, join='inner')
print(df_train.shape, df_val.shape, df_test.shape)

# fix for a weird bug where all-0 columns in BoC becomes NaN after concat
df_train.fillna(value=0, inplace=True)
df_val.fillna(value=0, inplace=True)
df_test.fillna(value=0, inplace=True)

train_X = df_train.drop('label', axis=1).values
train_y = df_train['label'].values
val_X = df_val.drop('label', axis=1).values
val_y = df_val['label'].values
test_X = df_test.drop('label', axis=1).values
test_y = df_test['label'].values

# uncomment the model to use
print('fitting model...')
model = linear_model.LogisticRegression(solver='lbfgs', random_state=42,
    penalty='l2', C=0.001, max_iter=10000, class_weight='balanced')
#model = ensemble.RandomForestClassifier(n_estimators=200,
#    class_weight='balanced', random_state=42, max_leaf_nodes=200)
model.fit(train_X, train_y)

train_pred = model.predict(train_X)
print('\n\n\ntrain:')
print(metrics.confusion_matrix(train_y, train_pred))
print(metrics.classification_report(train_y, train_pred))

val_pred = model.predict(val_X)
print('\n\n\nvalidation:')
print(metrics.confusion_matrix(val_y, val_pred))
print(metrics.classification_report(val_y, val_pred))

val_prob = model.predict_proba(val_X)[:, 1]
fpr, tpr, _ = metrics.roc_curve(val_y, val_prob)
roc_auc = metrics.auc(fpr, tpr)
print('ROC AUC:', roc_auc)

quit()

test_pred = model.predict(test_X)
test_prob = model.predict_proba(test_X)[:, 1]
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

#variables = df_train.drop('label', axis=1).columns
#coefs = model.coef_
#significance = [(v, c) for (v, c) in zip(variables, coefs[0,:])]
#significance.sort(key=lambda x: x[1])
#print(significance[0:10])
#print(significance[-1:-11:-1])
