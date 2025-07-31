import pandas as pd
import numpy as np
import scanpy as sc
import pickle
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.linear_model import LogisticRegression
from utils import RelationClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import mygene
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# prediction of gene-gene interaction

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

with open(f"/home/lyh/project/sc/main/model/data/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle", "rb") as fp:
    GPT_3_5_gene_embeddings = pickle.load(fp)

panglao = sc.read_h5ad('/home/lyh/project/sc/main/model/data/panglao_10000.h5ad')
scMomer = np.load('/home/lyh/project/sc/main/model/data/mean_values.npy')


gene_name = panglao.var_names
name = GPT_3_5_gene_embeddings.keys()
intersection_name = gene_name.intersection(name)


GPT_3_5_gene_embeddings = {}
for name, row in zip(gene_name, scMomer):
    GPT_3_5_gene_embeddings[name] = row

# gene gene interaction
train_text_PPI = pd.read_csv('/home/lyh/project/sc/main/model/data/HuRI.tsv',sep='\t',
                            header=None)
train_text_PPI.columns = ['gene_1','gene_2']

mg = mygene.MyGeneInfo()
gene1 = mg.querymany(train_text_PPI['gene_1'], species='human')
gene2 = mg.querymany(train_text_PPI['gene_2'], species='human')
gene1_name = [x['symbol'] for x in gene1 if 'symbol' in x]
gene2_name = [x['symbol'] for x in gene2 if 'symbol' in x]


X_array = []
y_array = []
for gene1_n, gene2_n in zip(gene1_name, gene2_name):
    if gene1_n in  intersection_name \
        and gene2_n in intersection_name:
        X_array.append(
            np.concatenate((GPT_3_5_gene_embeddings[gene1_n], GPT_3_5_gene_embeddings[gene2_n])))
        # X_array_train.append(GPT_3_5_gene_embeddings[row['gene_1']]+\
        #                      GPT_3_5_gene_embeddings[row['gene_2']])
        y_array.append(1)
        X_array.append(
            np.concatenate((random.choice(list(GPT_3_5_gene_embeddings.values())), random.choice(list(GPT_3_5_gene_embeddings.values())))))
        y_array.append(0)
X_array = np.array(X_array)
y_array = np.array(y_array)

cv = StratifiedKFold(n_splits=5)

# Lists to store ROC AUC scores for each fold
roc_auc_logistic = []
roc_auc_rf = []

# Lists to store TPR and FPR for each fold
tpr_logistic = []
fpr_logistic = []
tpr_rf = []
fpr_rf = []

for train_index, test_index in cv.split(X_array, y_array):
    X_train, X_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y_array[train_index], y_array[test_index]

    # Logistic Regression
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score_logistic)
    roc_auc = auc(fpr, tpr)
    roc_auc_logistic.append(roc_auc)
    tpr_logistic.append(tpr)
    fpr_logistic.append(fpr)


    # Random Forest
    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train, y_train)
    y_score_rf = random_forest_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score_rf)
    roc_auc = auc(fpr, tpr)
    roc_auc_rf.append(roc_auc)
    tpr_rf.append(tpr)
    fpr_rf.append(fpr)

# Print ROC AUC scores
print(f"PPI prediction: Logistic Regression ROC AUC: {np.mean(roc_auc_logistic):.3f} +/- {np.std(roc_auc_logistic):.3f}")
print(f"PPI prediction: Random Forest ROC AUC: {np.mean(roc_auc_rf):.3f} +/- {np.std(roc_auc_rf):.3f}")


