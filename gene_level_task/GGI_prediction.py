import pandas as pd
import numpy as np
import scanpy as sc
import pickle
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.linear_model import LogisticRegression
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

def balance_dataset(X_array, y_array):
    """
    平衡数据集中的正负样本数量。

    参数:
    X_array (np.ndarray): 特征数组。
    y_array (np.ndarray): 标签数组，其中1表示正样本，0表示负样本。

    返回:
    np.ndarray, np.ndarray: 平衡后的特征数组和标签数组。
    """
    # 计算正样本的数量
    num_positive_samples = np.sum(y_array == 1)

    # 找到负样本的索引
    positive_indices = np.where(y_array == 1)[0]
    negative_indices = np.where(y_array == 0)[0]

    # 如果负样本数量大于正样本数量，进行随机下采样
    if len(negative_indices) > num_positive_samples:
        # 随机选择与正样本数量相同的负样本索引
        selected_negative_indices = np.random.choice(negative_indices, num_positive_samples, replace=False)

        # 合并正样本和下采样后的负样本
        positive_indices = np.where(y_array == 1)[0]
        balanced_indices = np.concatenate([positive_indices, selected_negative_indices])

        # 重新排列索引以保持随机性

    else:
        # 如果负样本数量不多于正样本数量，则直接返回原始数据
        selected_positive_indices = np.random.choice(positive_indices, len(negative_indices) , replace=False)

        # 合并下采样后的正样本和负样本
        balanced_indices = np.concatenate([selected_positive_indices, negative_indices])
    np.random.shuffle(balanced_indices)

    # 使用平衡后的索引重新构建数据集
    X_array_balanced = X_array[balanced_indices]
    y_array_balanced = y_array[balanced_indices]
    return X_array_balanced, y_array_balanced

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

with open(f"/home/lyh/project/sc/main/model/data/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle", "rb") as fp:
    GPT_3_5_gene_embeddings = pickle.load(fp)

panglao = sc.read_h5ad('/home/lyh/project/sc/main/model/data/panglao_10000.h5ad')
scMomer_emb = np.load('/home/lyh/project/sc/main/model/data/mean_values.npy')


gene_name = panglao.var_names

name = GPT_3_5_gene_embeddings.keys()
intersection_name = gene_name.intersection(name)

#use our dataset
GPT_3_5_gene_embeddings = {}
for name, row in zip(gene_name, scMomer_emb):
    GPT_3_5_gene_embeddings[name] = row


# gene gene interaction
train_text_GGI = pd.read_csv('/home/lyh/project/sc/main/model/data/predictionData/train_text.txt',sep=' ',
                            header=None)
train_label_GGI = pd.read_csv('/home/lyh/project/sc/main/model/data/predictionData/train_label.txt',
                              header=None)
train_text_GGI.columns = ['gene_1','gene_2']
train_label_GGI.columns = ['label']
train_text_GGI_df = pd.concat([train_text_GGI,train_label_GGI], axis=1)

X_array_train = []
y_array_train = []
for i, row in train_text_GGI_df.iterrows():
    if row['gene_1'] in  intersection_name \
        and row['gene_2'] in intersection_name:
        X_array_train.append(
            np.concatenate((GPT_3_5_gene_embeddings[row['gene_1']], GPT_3_5_gene_embeddings[row['gene_2']])))
        # X_array_train.append(GPT_3_5_gene_embeddings[row['gene_1']]+\
        #                      GPT_3_5_gene_embeddings[row['gene_2']])
        y_array_train.append(row['label'])

X_array = np.array(X_array_train)
y_array = np.array(y_array_train)
X_array = X_array.reshape(X_array.shape[0],-1)
# X_array, y_array = balance_dataset(X_array, y_array)


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

    print(roc_auc_logistic)


    # # Random Forest
    # random_forest_model = RandomForestClassifier()
    # random_forest_model.fit(X_train, y_train)
    # y_score_rf = random_forest_model.predict_proba(X_test)[:, 1]
    # fpr, tpr, _ = roc_curve(y_test, y_score_rf)
    # roc_auc = auc(fpr, tpr)
    # roc_auc_rf.append(roc_auc)
    # tpr_rf.append(tpr)
    # fpr_rf.append(fpr)

# Print ROC AUC scores
print(f"PPI prediction: Logistic Regression ROC AUC: {np.mean(roc_auc_logistic):.3f} +/- {np.std(roc_auc_logistic):.3f}")
# print(f"PPI prediction: Random Forest ROC AUC: {np.mean(roc_auc_rf):.3f} +/- {np.std(roc_auc_rf):.3f}")
