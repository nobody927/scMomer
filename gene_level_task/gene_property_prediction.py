from datasets import load_from_disk, load_dataset
import argparse
import pandas as pd
import pickle
import numpy as np
import mygene
import random
import scanpy as sc
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt

panglao = sc.read_h5ad('/home/lyh/project/sc/main/model/data/panglao_10000.h5ad')

scMomer_emb = np.load('/home/lyh/project/sc/main/model/data/mean_values.npy')
gene_name = panglao.var_names

GPT_3_5_gene_embeddings = {}
for name, row in zip(gene_name, scMomer_emb):
    GPT_3_5_gene_embeddings[name] = row


with open("/home/lyh/project/sc/main/model/data/gene_level/example_input_files/gene_classification/tf_regulatory_range/tf_regulatory_range.pickle", "rb") as f:
    p = pickle.load(f)

long_range_tf_gene = p["long_range"]
short_range_tf_gene = p["short_range"]

mg = mygene.MyGeneInfo()
# long_range_tf_gene_name = [x['gene'] for x in results]
# gene_name1 = mg.querymany(gene_name, species='human', scopes='ensembl.gene', fields='ensembl.gene')
# short_range_tf_gene = mg.query(short_range_tf_gene, species='human', scopes='ensembl.gene', fields='ensembl.gene')
long_range_tf_gene = mg.querymany(long_range_tf_gene, species='human')
short_range_tf_gene = mg.querymany(short_range_tf_gene, species='human')
long_range_tf_gene_name = [x['symbol'] for x in long_range_tf_gene]
short_range_tf_gene_name = [x['symbol'] for x in short_range_tf_gene if 'symbol' in x]

# #applied to our model
# long_range_tf_gene_name1 = set(name).intersection(p["long_range"])
# short_range_tf_gene_name1 = set(name).intersection(p["short_range"])

#applied to our model
long_range_tf_gene_name = gene_name.intersection(long_range_tf_gene_name)
short_range_tf_gene_name = gene_name.intersection(short_range_tf_gene_name)

# umap_model = umap.UMAP(n_components=2, random_state=42)
# reduced_data = umap_model.fit_transform(gene2vec_weight)
# colors = []
# alpha = []
# for gene in gene_name:
#     if gene in long_range_tf_gene_name:
#         colors.append('red')  # 集合 A 的基因用红色表示
#         alpha.append(1.0)    # 不透明
#     elif gene in short_range_tf_gene_name:
#         colors.append('blue')  # 集合 B 的基因用蓝色表示
#         alpha.append(1.0)    # 不透明
#     else:
#         colors.append('gray')  # 其他基因用灰色表示
#         alpha.append(0.2)
#
# plt.figure(figsize=(8, 6))
# gray_indices = [i for i, color in enumerate(colors) if color == 'gray']
# plt.scatter(reduced_data[gray_indices, 0], reduced_data[gray_indices, 1],
#             c=[colors[i] for i in gray_indices], s=5, alpha=[alpha[i] for i in gray_indices])
#
# # 再绘制红色和蓝色点
# highlight_indices = [i for i, color in enumerate(colors) if color in ['red', 'blue']]
# plt.scatter(reduced_data[highlight_indices, 0], reduced_data[highlight_indices, 1],
#             c=[colors[i] for i in highlight_indices], s=5, alpha=[alpha[i] for i in highlight_indices])
#
# plt.title('UMAP Visualization of Gene2Vec Weights')
# plt.xlabel('UMAP Component 1')
# plt.ylabel('UMAP Component 2')
# plt.show()

x_long_range_tf = [GPT_3_5_gene_embeddings[name] for name in long_range_tf_gene_name\
               if name in GPT_3_5_gene_embeddings]
x_short_range_tf = [GPT_3_5_gene_embeddings[name] for name in short_range_tf_gene_name \
                 if name in GPT_3_5_gene_embeddings]
X_array = x_long_range_tf.copy()
X_array.extend(x_short_range_tf)
y_array = np.concatenate((np.repeat(1,len(x_long_range_tf)),np.repeat(0,len(x_short_range_tf))))


np.random.seed(2023)
random.seed(2023)

X_array = np.array(X_array)
y_array = np.array(y_array)
# Assuming x and y are your data
# For demonstration, let's create some dummy data.
# Ensure your data is in NumPy array format for compatibility
# X_array = np.concatenate((x_long_range_tf,x_short_range_tf))
# y_array =  np.concatenate((np.repeat(1,len(x_long_range_tf)),np.repeat(0,len(x_short_range_tf))))

# Set up Stratified K-Folds cross-validator
# It provides train/test indices to split data into train/test sets
cv = StratifiedKFold(n_splits=5)

# Lists to store ROC AUC scores for each fold
roc_auc_logistic = []
roc_auc_rf = []

results_range = {
    'LogisticRegression': {
        'ROC_AUC': [],
        'Accuracy': [],
        'tpr':[],
        'fpr':[],
        'f1':[]
    },
    'RandomForestClassifier': {
        'ROC_AUC': [],
        'Accuracy': [],
        'tpr': [],
        'fpr': [],
        'f1':[]
    }
}

# Lists to store TPR and FPR for each fold
tpr_logistic = []
fpr_logistic = []
tpr_rf = []
fpr_rf = []
acc_rf = []
acc_logistic = []
f1_rf = []
f1_logistic = []
for train_index, test_index in cv.split(X_array, y_array):
    X_train, X_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y_array[train_index], y_array[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)  # 预测测试集
    y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred_logistic,average='macro')
    fpr, tpr, _ = roc_curve(y_test, y_score_logistic)
    roc_auc = auc(fpr, tpr)

    roc_auc_logistic.append(roc_auc)
    tpr_logistic.append(tpr)
    fpr_logistic.append(fpr)
    acc = accuracy_score(y_test, y_pred_logistic)
    acc_logistic.append(acc)
    results_range['LogisticRegression']['ROC_AUC'].append(roc_auc)
    results_range['LogisticRegression']['Accuracy'].append(acc)
    results_range['LogisticRegression']['tpr'].append(tpr)
    results_range['LogisticRegression']['fpr'].append(fpr)
    results_range['LogisticRegression']['f1'].append(f1)


    # Random Forest
    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train, y_train)
    y_pred_rf = random_forest_model.predict(X_test)
    y_score_rf = random_forest_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_logistic, average='macro')
    fpr, tpr, _ = roc_curve(y_test, y_score_rf)
    roc_auc = auc(fpr, tpr)
    roc_auc_rf.append(roc_auc)
    tpr_rf.append(tpr)
    fpr_rf.append(fpr)
    acc_rf.append(acc)
    results_range['RandomForestClassifier']['ROC_AUC'].append(roc_auc)
    results_range['RandomForestClassifier']['Accuracy'].append(acc)
    results_range['RandomForestClassifier']['tpr'].append(tpr)
    results_range['RandomForestClassifier']['fpr'].append(fpr)
    results_range['RandomForestClassifier']['f1'].append(f1)


# Print ROC AUC scores
print(f"Long- vs short- range TFs prediction: Logistic Regression ROC AUC: {np.mean(roc_auc_logistic):.3f} +/- {np.std(roc_auc_logistic):.3f}")
print(f"Long- vs short- range TFs prediction: Random Forest ROC AUC: {np.mean(roc_auc_rf):.3f} +/- {np.std(roc_auc_rf):.3f}")


#sensitive prediction

with open("/home/lyh/project/sc/main/model/data/gene_level/example_input_files/gene_classification/dosage_sensitive_tfs/dosage_sensitivity_TFs.pickle", "rb") as f:
    p = pickle.load(f)
sensitive = p["Dosage-sensitive TFs"]
insensitive = p["Dosage-insensitive TFs"]


mg = mygene.MyGeneInfo()
sensitive_query = mg.querymany(sensitive, species='human')
in_sensitive_query = mg.querymany(insensitive, species='human')
sensitive_gene_name = [x['symbol'] for x in sensitive_query]
in_sensitive_gene_name = [x['symbol'] for x in in_sensitive_query if 'symbol' in x]

# applied to our model
sensitive_gene_name = gene_name.intersection(sensitive_gene_name)
in_sensitive_gene_name = gene_name.intersection(in_sensitive_gene_name)

x_sensitive = [GPT_3_5_gene_embeddings[name] for name in sensitive_gene_name\
               if name in GPT_3_5_gene_embeddings]
x_insensitive = [GPT_3_5_gene_embeddings[name] for name in in_sensitive_gene_name \
                 if name in GPT_3_5_gene_embeddings]
x_dosage = x_sensitive.copy()
x_dosage.extend(x_insensitive)
y_dosage = np.concatenate((np.repeat(1,len(x_sensitive)),np.repeat(0,len(x_insensitive))))

# Ensure your data is in NumPy array format for compatibility
X_array = np.array(x_dosage)
y_array = np.array(y_dosage)


# Set up Stratified K-Folds cross-validator
# It provides train/test indices to split data into train/test sets
cv = StratifiedKFold(n_splits=5)

results_sene = {
    'LogisticRegression': {
        'ROC_AUC': [],
        'Accuracy': [],
        'tpr':[],
        'fpr':[],
        'f1': []
    },
    'RandomForestClassifier': {
        'ROC_AUC': [],
        'Accuracy': [],
        'tpr': [],
        'fpr': [],
        'f1': []
    }
}


# Lists to store ROC AUC scores for each fold
roc_auc_logistic = []
roc_auc_rf = []

# Lists to store TPR and FPR for each fold
tpr_logistic = []
fpr_logistic = []
tpr_rf = []
fpr_rf = []
acc_rf = []
acc_logistic = []
f1_rf = []
f1_logistic = []
for train_index, test_index in cv.split(X_array, y_array):
    X_train, X_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y_array[train_index], y_array[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)  # 预测测试集
    y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred_logistic, average='macro')
    fpr, tpr, _ = roc_curve(y_test, y_score_logistic)
    roc_auc = auc(fpr, tpr)
    roc_auc_logistic.append(roc_auc)
    tpr_logistic.append(tpr)
    fpr_logistic.append(fpr)
    acc = accuracy_score(y_test, y_pred_logistic)
    acc_logistic.append(acc)
    results_sene['LogisticRegression']['ROC_AUC'].append(roc_auc)
    results_sene['LogisticRegression']['Accuracy'].append(acc)
    results_sene['LogisticRegression']['tpr'].append(tpr)
    results_sene['LogisticRegression']['fpr'].append(fpr)
    results_sene['LogisticRegression']['f1'].append(f1)

    # Random Forest
    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train, y_train)
    y_pred_rf = random_forest_model.predict(X_test)
    y_score_rf = random_forest_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_logistic, average='macro')
    fpr, tpr, _ = roc_curve(y_test, y_score_rf)
    roc_auc = auc(fpr, tpr)
    roc_auc_rf.append(roc_auc)
    tpr_rf.append(tpr)
    fpr_rf.append(fpr)
    acc_rf.append(acc)
    results_sene['RandomForestClassifier']['ROC_AUC'].append(roc_auc)
    results_sene['RandomForestClassifier']['Accuracy'].append(acc)
    results_sene['RandomForestClassifier']['tpr'].append(tpr)
    results_sene['RandomForestClassifier']['fpr'].append(fpr)
    results_sene['RandomForestClassifier']['f1'].append(f1)

# Print ROC AUC scores
print(f"Sensitive prediction: Logistic Regression ROC AUC: {np.mean(roc_auc_logistic):.3f} +/- {np.std(roc_auc_logistic):.3f}")
print(f"Sensitive prediction: Random Forest ROC AUC: {np.mean(roc_auc_rf):.3f} +/- {np.std(roc_auc_rf):.3f}")



# Methylation state prediction

with open("/home/lyh/project/sc/main/model/data/gene_level/example_input_files/gene_classification/bivalent_promoters/bivalent_vs_lys4_only_genomewide.pickle", "rb") as f:
    bivalent_gene = pickle.load(f)
with open("/home/lyh/project/sc/main/model/data/gene_level/example_input_files/gene_classification/bivalent_promoters/bivalent_vs_no_methyl.pickle", "rb") as f:
    bivalent_vs_no_methyl = pickle.load(f)
with open("/home/lyh/project/sc/main/model/data/gene_level/example_input_files/gene_classification/bivalent_promoters/bivalent_vs_lys4_only.pickle", "rb") as f:
    bivalent_vs_lys4_only = pickle.load(f)

bivalent_gene_labels = bivalent_gene["bivalent"]
no_methylation_gene_labels = bivalent_vs_no_methyl["no_methylation"]
lysine_gene_labels = bivalent_gene["lys4_only"]

mg = mygene.MyGeneInfo()

bivalent_query = mg.querymany(list(bivalent_gene_labels), species='human')
no_methylation_query = mg.querymany(list(no_methylation_gene_labels), species='human')
lysine_query = mg.querymany(list(lysine_gene_labels), species='human')

bivalent_name = [x['symbol'] for x in bivalent_query if 'symbol' in x]
no_methylation_name = [x['symbol'] for x in no_methylation_query if 'symbol' in x]
lysine_name = [x['symbol'] for x in lysine_query if 'symbol' in x]

#applied to our model
bivalent_name = gene_name.intersection(bivalent_name)
no_methylation_name = gene_name.intersection(no_methylation_name)
lysine_name = gene_name.intersection(lysine_name)

x_bivalent = [GPT_3_5_gene_embeddings[name] for name in bivalent_name\
               if name in GPT_3_5_gene_embeddings]
x_no_methylation = [GPT_3_5_gene_embeddings[name] for name in no_methylation_name \
                 if name in GPT_3_5_gene_embeddings]
x_lysine = [GPT_3_5_gene_embeddings[name] for name in lysine_name \
                 if name in GPT_3_5_gene_embeddings]
X_array = x_bivalent.copy()
X_array.extend(x_no_methylation)
y_array = np.concatenate((np.repeat(1,len(x_bivalent)),np.repeat(0,len(x_no_methylation))))

# bivalent versus non-methylated
np.random.seed(2023)
random.seed(2023)

X_array = np.array(X_array)
y_array = np.array(y_array)
# Set up Stratified K-Folds cross-validator
# It provides train/test indices to split data into train/test sets
cv = StratifiedKFold(n_splits=5)

results_non_methylated = {
    'LogisticRegression': {
        'ROC_AUC': [],
        'Accuracy': [],
        'tpr':[],
        'fpr':[],
        'f1': []
    },
    'RandomForestClassifier': {
        'ROC_AUC': [],
        'Accuracy': [],
        'tpr': [],
        'fpr': [],
        'f1': []
    }
}

# Lists to store ROC AUC scores for each fold
roc_auc_logistic = []
roc_auc_rf = []

# Lists to store TPR and FPR for each fold
tpr_logistic = []
fpr_logistic = []
tpr_rf = []
fpr_rf = []
acc_rf = []
acc_logistic = []
f1_rf = []
f1_logistic = []
for train_index, test_index in cv.split(X_array, y_array):
    X_train, X_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y_array[train_index], y_array[test_index]

    # Logistic Regression
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)  # 预测测试集
    y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred_logistic, average='macro')
    fpr, tpr, _ = roc_curve(y_test, y_score_logistic)
    roc_auc = auc(fpr, tpr)
    roc_auc_logistic.append(roc_auc)
    tpr_logistic.append(tpr)
    fpr_logistic.append(fpr)
    acc = accuracy_score(y_test, y_pred_logistic)
    acc_logistic.append(acc)
    results_non_methylated['LogisticRegression']['ROC_AUC'].append(roc_auc)
    results_non_methylated['LogisticRegression']['Accuracy'].append(acc)
    results_non_methylated['LogisticRegression']['tpr'].append(tpr)
    results_non_methylated['LogisticRegression']['fpr'].append(fpr)
    results_non_methylated['LogisticRegression']['f1'].append(f1)

    # Random Forest
    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train, y_train)
    y_pred_rf = random_forest_model.predict(X_test)
    y_score_rf = random_forest_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_logistic, average='macro')
    fpr, tpr, _ = roc_curve(y_test, y_score_rf)
    roc_auc = auc(fpr, tpr)
    roc_auc_rf.append(roc_auc)
    tpr_rf.append(tpr)
    fpr_rf.append(fpr)
    acc_rf.append(acc)
    results_non_methylated['RandomForestClassifier']['ROC_AUC'].append(roc_auc)
    results_non_methylated['RandomForestClassifier']['Accuracy'].append(acc)
    results_non_methylated['RandomForestClassifier']['tpr'].append(tpr)
    results_non_methylated['RandomForestClassifier']['fpr'].append(fpr)
    results_non_methylated['RandomForestClassifier']['f1'].append(f1)
# Print ROC AUC scores
print(f"Bivalent versus non-methylated: Logistic Regression ROC AUC: {np.mean(roc_auc_logistic):.3f} +/- {np.std(roc_auc_logistic):.3f}")
print(f"Bivalent versus non-methylated: Random Forest ROC AUC: {np.mean(roc_auc_rf):.3f} +/- {np.std(roc_auc_rf):.3f}")

# bivalent versus lys-4 methylated

np.random.seed(2023)
random.seed(2023)

# Assuming x and y are your data
# For demonstration, let's create some dummy data.
# Ensure your data is in NumPy array format for compatibility

results_lys4 = {
    'LogisticRegression': {
        'ROC_AUC': [],
        'Accuracy': [],
        'tpr':[],
        'fpr':[],
        'f1': []
    },
    'RandomForestClassifier': {
        'ROC_AUC': [],
        'Accuracy': [],
        'tpr': [],
        'fpr': [],
        'f1': []
    }
}

X_array = x_lysine.copy()
X_array.extend(x_bivalent)
y_array = np.concatenate((np.repeat(1,len(x_lysine)),np.repeat(0,len(x_bivalent))))
X_array = np.array(X_array)
y_array = np.array(y_array)

# Set up Stratified K-Folds cross-validator
# It provides train/test indices to split data into train/test sets
cv = StratifiedKFold(n_splits=5)

# Lists to store ROC AUC scores for each fold
roc_auc_logistic = []
roc_auc_rf = []
roc_auc_xgb = []

# Lists to store TPR and FPR for each fold
tpr_logistic = []
fpr_logistic = []
tpr_rf = []
fpr_rf = []
acc_rf = []
acc_logistic = []
f1_rf = []
f1_logistic = []
for train_index, test_index in cv.split(X_array, y_array):
    X_train, X_test = X_array[train_index], X_array[test_index]
    y_train, y_test = y_array[train_index], y_array[test_index]

    # Logistic Regression
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)  # 预测测试集
    y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred_logistic, average='macro')
    fpr, tpr, _ = roc_curve(y_test, y_score_logistic)
    roc_auc = auc(fpr, tpr)
    roc_auc_logistic.append(roc_auc)
    tpr_logistic.append(tpr)
    fpr_logistic.append(fpr)
    acc = accuracy_score(y_test, y_pred_logistic)
    acc_logistic.append(acc)
    results_lys4['LogisticRegression']['ROC_AUC'].append(roc_auc)
    results_lys4['LogisticRegression']['Accuracy'].append(acc)
    results_lys4['LogisticRegression']['tpr'].append(tpr)
    results_lys4['LogisticRegression']['fpr'].append(fpr)
    results_lys4['LogisticRegression']['f1'].append(f1)

    # Random Forest
    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(X_train, y_train)
    y_pred_rf = random_forest_model.predict(X_test)
    y_score_rf = random_forest_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred_rf)
    f1 = f1_score(y_test, y_pred_logistic, average='macro')
    fpr, tpr, _ = roc_curve(y_test, y_score_rf)
    roc_auc = auc(fpr, tpr)
    roc_auc_rf.append(roc_auc)
    tpr_rf.append(tpr)
    fpr_rf.append(fpr)
    acc_rf.append(acc)
    results_lys4['RandomForestClassifier']['ROC_AUC'].append(roc_auc)
    results_lys4['RandomForestClassifier']['Accuracy'].append(acc)
    results_lys4['RandomForestClassifier']['tpr'].append(tpr)
    results_lys4['RandomForestClassifier']['fpr'].append(fpr)
    results_lys4['RandomForestClassifier']['f1'].append(f1)

results ={"range":results_range,
          "sene":results_sene,
          "non_methylated":results_non_methylated,
          "lys4":results_lys4}
with open('/home/lyh/project/result/gene_level/scMomer.pkl', 'wb') as f:
    pickle.dump(results, f)
# Print ROC AUC scores
print(f"Bivalent versus lys-4 methylated: Logistic Regression ROC AUC: {np.mean(roc_auc_logistic):.3f} +/- {np.std(roc_auc_logistic):.3f}")
print(f"Bivalent versus lys-4 methylated: Random Forest ROC AUC: {np.mean(roc_auc_rf):.3f} +/- {np.std(roc_auc_rf):.3f}")

