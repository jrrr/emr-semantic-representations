import sys, os, pickle
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import nltk.corpus
stopwords = set([stopword.replace('\'', '')
    for stopword in nltk.corpus.stopwords.words('english')])

# a bit ugly: assumes hadm_id2path.pkl has been written by extract_concepts.py
with open('hadm_id2path.pkl', 'rb') as f:
    hadm_id2path = pickle.load(f)
tmplist = []
for (hadm_id, path) in hadm_id2path.items():
    (path, epfile) = os.path.split(path)
    (_, pat_id) = os.path.split(path)
    timeseries_file = (pat_id + '_' +
        epfile.replace('_noteconcepts', '_timeseries'))
    tmplist.append((timeseries_file, hadm_id))
hadm_table = dict(tmplist)

def load_listfile(path):
    hadm_ids = []
    ys = []
    with open(path, 'r') as f:
        f.readline() # first line is header
        for line in f:
            line = line.strip()
            timeseries_file = line.split(',')[0]
            y = int(line[-1])
            hadm_ids.append(hadm_table[timeseries_file])
            ys.append(y)
    return pd.DataFrame(data=ys, index=hadm_ids, columns=['TARGET'])

def add_text(df_train, df_val, df_test):
    with open(sys.argv[2], 'rb') as f:
        df_text = pickle.load(f)
    df_train['TEXT'] = df_text
    df_val['TEXT'] = df_text
    df_test['TEXT'] = df_text
    return df_train, df_val, df_test

print('loading listfiles...')
df_train = load_listfile(os.path.join(sys.argv[1], 'train_listfile.csv'))
df_val = load_listfile(os.path.join(sys.argv[1], 'val_listfile.csv'))
df_test = load_listfile(os.path.join(sys.argv[1], 'test_listfile.csv'))

print('loading text...')
df_train, df_val, df_test = add_text(df_train, df_val, df_test)

X_train = df_train['TEXT'].values
X_val = df_val['TEXT'].values
X_test = df_test['TEXT'].values
y_train = df_train['TARGET'].values
y_val = df_val['TARGET'].values
y_test = df_test['TARGET'].values

print('fitting model...')
model = Pipeline([('vect', CountVectorizer(stop_words=stopwords, binary=True)),
                  #('tfidf', TfidfTransformer(use_idf=True)),
#                  ('lr', LogisticRegression(solver='lbfgs', penalty='l2',
#                            C=0.001, max_iter=10000, class_weight='balanced'))
                   ('rf', RandomForestClassifier(n_estimators=200,
                             class_weight='balanced', random_state=42,
                             max_leaf_nodes=750))
                  ])
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_prob = model.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, val_prob)
roc_auc = auc(fpr, tpr)
print(confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred))
print('ROC AUC:', roc_auc)

# only uncomment when hyperparameters have been determined
quit()

test_pred = model.predict(X_test)
test_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, test_prob)
test_roc_auc = auc(fpr, tpr)
print('\\begin{tabular}{@{}c@{}} %.3f \\\\ \\textbf{%.3f} \end{tabular} &' %
    (accuracy_score(y_val, val_pred), accuracy_score(y_test, test_pred)))
print('\\begin{tabular}{@{}c@{}} %.3f \\\\ \\textbf{%.3f} \end{tabular} &' %
    (precision_score(y_val, val_pred), precision_score(y_test, test_pred)))
print('\\begin{tabular}{@{}c@{}} %.3f \\\\ \\textbf{%.3f} \end{tabular} &' %
    (recall_score(y_val, val_pred), recall_score(y_test, test_pred)))
print('\\begin{tabular}{@{}c@{}} %.3f \\\\ \\textbf{%.3f} \end{tabular} &' %
    (f1_score(y_val, val_pred), f1_score(y_test, test_pred)))
print('\\begin{tabular}{@{}c@{}} %.3f \\\\ \\textbf{%.3f} \end{tabular} \\\\ \\hline' %
    (roc_auc, test_roc_auc))

#vocab = [(word, index) for (word, index) in model['vect'].vocabulary_.items()]
#vocab.sort(key=lambda x: x[1])
#vocab = [item[0] for item in vocab]
#coefs = model['lr'].coef_
#significance = [(w, c) for (w, c) in zip(vocab, coefs[0,:])]
#significance.sort(key=lambda x: x[1])
#print(significance[0:10])
#print(significance[-1:-11:-1])
