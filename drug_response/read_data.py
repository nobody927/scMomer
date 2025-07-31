import pandas as pd
import torch
import numpy as np
import csv
import hickle as hkl
import os, random
import scipy.sparse as sp
from torch_geometric.data import Data as GraphData
from DeepCDR import DeepCDR
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

TCGA_label_set = ["ALL","BLCA","BRCA","CESC","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                  "LUSC","MESO","MM","NB","OV","PAAD","SCLC","SKCM",
                  "STAD","THCA",'COAD/READ']

def drug_response_data_gen(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,IC50_thred_file,use_thred=False):
    # drug_id --> pubchem_id

    reader = csv.reader(open(Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}
    name2pubchemid = {item[1]:item[5] for item in rows if item[5].isdigit()}
    drug_names = [item.strip() for item in open(IC50_thred_file).readlines()[0].split('\t')]
    IC50_threds = [float(item.strip()) for item in open(IC50_thred_file).readlines()[1].split('\t')]
    drug2thred = {name2pubchemid[a]:b for a,b in zip(drug_names,IC50_threds) if a in name2pubchemid.keys()}

    # map cellline --> cancer type
    cellline2cancertype = {}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        # if TCGA_label in TCGA_label_set:
        cellline2cancertype[cellline_id] = TCGA_label

    # load demap cell lines genomic mutation features
    mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])
    cell_line_id_set = list(mutation_feature.index)

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat, adj_list, degree_list = hkl.load('%s/%s' % (Drug_feature_file, each))
        # edge_list = []
        # for u, neighbors in enumerate(adj_list):
        #     for v in neighbors:
        #         # 确保 (u, v) 和 (v, u) 不重复
        #         edge_list.append((u, v))

        drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
    assert len(drug_pubchem_id_set) == len(drug_feature.values())

    # load gene expression faetures
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])

    # only keep overlapped cell lines
    mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]

    # load methylationf
    methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])
    assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
    experiment_data = pd.read_csv(Cancer_response_exp_file, sep=',', header=0, index_col=[0])
    # filter experiment data
    drug_match_list = [item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]

    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:
                if not np.isnan(experiment_data_filtered.loc[
                                    each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug, each_cellline])
                    if False:
                        if pubchem_id in drug2thred.keys():
                            binary_IC50 = 1 if ln_IC50 < drug2thred[pubchem_id] else 0
                            data_idx.append(
                                (each_cellline, pubchem_id, binary_IC50, cellline2cancertype[each_cellline]))
                    else:
                        # binary_IC50 = 1 if ln_IC50 > -2 else 0
                        data_idx.append((each_cellline, pubchem_id, ln_IC50, cellline2cancertype[each_cellline]))

    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.' % (len(data_idx), nb_celllines, nb_drugs))
    return mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx

Drug_info_file = './data/Drug_info.csv'
Cell_line_info_file = './data/Cell_lines_annotations.txt'
Drug_feature_file = './data/drug_graph_feat'
Cancer_response_exp_file = './data/GDSC_IC50.csv'
Gene_expression_file = './data/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = './data/genomic_methylation_561celllines_808genes_demap_features.csv'
Genomic_mutation_file = './data/genomic_mutation_34673_demap_features.csv'
IC50_thred_file = './data/IC50_thred.txt'
Max_atoms = 100
israndom =False

def DataSplit(data_idx,ratio = 0.95):
    random.seed(42)
    data_train_idx,data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1]==each_type]
        train_list = random.sample(data_subtype_idx,int(ratio*len(data_subtype_idx)))
        test_list = [item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    return data_train_idx,data_test_idx

def DataSplit_drug_blind(data_idx,ratio = 0.95):
    random.seed(42)
    data_train_idx,data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1]==each_type]
        train_list = random.sample(data_subtype_idx,int(ratio*len(data_subtype_idx)))
        test_list = [item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    return data_train_idx,data_test_idx


def FeatureExtract(data_idx,drug_feature,mutation_feature,gexpr_feature,methylation_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_drug_feature = next(iter(drug_feature.values()))[0].shape[1]
    nb_mutation_feature = mutation_feature.shape[1]
    nb_gexpr_features = gexpr_feature.shape[1]
    nb_methylation_features = methylation_feature.shape[1]
    drug_data = [[] for item in range(nb_instance)]
    graph = []
    # drug_f = np.zeros((nb_instance,75),dtype='float32')
    feat_mats = np.zeros((nb_instance, Max_atoms, nb_drug_feature), dtype='float32')
    adj_lists = np.zeros((nb_instance, Max_atoms, Max_atoms), dtype='float32')
    mutation_data = np.zeros((nb_instance,1, nb_mutation_feature),dtype='float32')
    gexpr_data = np.zeros((nb_instance,nb_gexpr_features),dtype='float32')
    methylation_data = np.zeros((nb_instance, nb_methylation_features),dtype='float32')
    target = np.zeros(nb_instance,dtype='float32')
    for idx in range(nb_instance):
        cell_line_id,pubchem_id,binary_IC50,cancer_type = data_idx[idx]
        #modify
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)]
        #fill drug data,padding to the same size with zeros
        feat,adj_mat = CalculateGraphFeat(feat_mat,adj_list)
        feat_mats[idx,:,:] = feat
        adj_lists[idx, :,:] = adj_mat
        # drug_data[idx] = GraphData(x=feat_mat, edge_index=edge_list)
        # drug_data[idx] = [feat_mat, edge_list]
        #randomlize X A
        mutation_data[idx,0,:] = mutation_feature.loc[cell_line_id].values
        gexpr_data[idx,:] = gexpr_feature.loc[cell_line_id].values
        methylation_data[idx,:] = methylation_feature.loc[cell_line_id].values
        target[idx] = binary_IC50
        cancer_type_list.append([cancer_type,cell_line_id,pubchem_id])

    batch_data = convert_to_pyg_batch(torch.from_numpy(feat_mats), torch.from_numpy(adj_lists))
    dataloader = CustomDataset(batch_data,gexpr_data,methylation_data,mutation_data,target)
    return dataloader,cancer_type_list

def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm

def random_adjacency_matrix(n):
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix

def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2
    return feat,adj_mat


def convert_to_pyg_batch(feat, adj_mat):
    """
    将批量特征和邻接矩阵转换为 PyTorch Geometric 的批量图数据。
    :param feat: 形状为 (batch_size, Max_atoms, feat_dim) 的张量。
    :param adj_mat: 形状为 (batch_size, Max_atoms, Max_atoms) 的张量。
    :return: PyTorch Geometric 的 Batch 对象。
    """
    batch_size, max_atoms, feat_dim = feat.shape

    # 初始化节点特征和边索引的列表
    data_list = []

    for i in range(batch_size):
        # 当前图的节点特征
        x = feat[i]

        # 当前图的邻接矩阵
        adj = adj_mat[i]
        edge_index, _ = dense_to_sparse(adj)

        # 创建单个图数据对象
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

    # 将所有图数据合并为一个批量
    batch = Batch.from_data_list(data_list)
    return batch

class CustomDataset(Dataset):
    def __init__(self, drug_feature, gene, me, mu, label):
        self.drug_feature = drug_feature  # 这个是图结构，保留原始 list of Data
        self.gene = torch.tensor(gene, dtype=torch.float32)
        self.me = torch.tensor(me, dtype=torch.float32)
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return self.gene.shape[0]

    def __getitem__(self, idx):
        # drug_dict = self.drug_feature,
            # "edge_list_path": self.drug_edge

        drug_dict = self.drug_feature[idx]
        # "edge_list_path": self.drug_edge
        gene_expression = self.gene[idx]
        methylation = self.me[idx]
        mutation = self.mu[idx]
        label = self.label[idx]

        return drug_dict, gene_expression, methylation, mutation, label
