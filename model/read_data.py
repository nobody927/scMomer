from torch.utils.data import Dataset, random_split, Subset
import muon as mu
import scanpy as sc
import numpy as np
from sklearn.model_selection import  StratifiedShuffleSplit
import anndata as ad
import torch
from scipy.sparse import csr_matrix
from scipy import sparse


def read_data(modal, data_path, backed = False, split=0.9 , cell_type = True, latent_train = None, latent_val = None):
    if modal == 'multimodal':
        mdata = mu.read(data_path, backed=backed)
    else:
        mdata = sc.read_h5ad(data_path)
    mdata.mod['rna'].var_names_make_unique()
    mdata.mod['atac'].var_names_make_unique()

    label_dict, label = np.unique(np.array(mdata.mod['rna'].obs['cell_type']), return_inverse=True)

    label = torch.from_numpy(label)
    atac = process_atac(mdata)
    mdata.mod["atac"] = atac


    rna = process_rna(mdata)
    mdata.mod["rna"] = rna

    if isinstance(split, float):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=split, random_state=42)#14
        for index_train, index_val in sss.split(mdata, label):

            train_rna, train_atac, label_train = mdata.mod['rna'].X[index_train], mdata.mod['atac'].X[index_train], label[index_train]
            val_rna, val_atac, label_val = mdata.mod['rna'].X[index_val], mdata.mod['atac'].X[index_val], label[index_val]

    elif isinstance(split, str):
        if split.endswith("_r"):
            split = split.strip("_r")
            reverse = True
        else:
            reverse = False
        assert (
                split in mdata.obs["cell_type"].unique()
        ), f"Cell type {split} not found"
        if reverse:
            train_idx = np.where(mdata.obs["cell_type"] != split)[0]
            val_idx = np.where(mdata.obs["cell_type"] == split)[0]
        else:
            train_idx = np.where(mdata.obs["cell_type"] == split)[0]
            val_idx = np.where(mdata.obs["cell_type"] != split)[0]

    elif isinstance(split, list):  # split multiple cell types # TO DO
        train_idx = mdata.obs[mdata.obs["cell_type"].isin(split)].index
        val_idx = mdata.obs[~mdata.obs["cell_type"].isin(split)].index
    if latent_train is None:
        train_dataset = CustomDataset(train_rna, train_atac, label_train)
        val_dataset = CustomDataset(val_rna, val_atac, label_val)
    else:
        train_dataset = CustomDataset1(train_rna, train_atac, label_train, latent_train)
        val_dataset = CustomDataset1(val_rna, val_atac, label_val, latent_val)
    return train_dataset, val_dataset, label_dict

def process_atac(mdata):
    atac = mdata.mod['atac']
    atac.X.data[atac.X.data > 0] = 1
    # atac.var_names_make_unique()
    return atac

def process_rna(mdata):
    rna = mdata.mod["rna"]
    # sc.pp.normalize_total(rna, target_sum=1e4)
    # sc.pp.log1p(rna)
    # rna.X = rna.X.toarray()
    rna.X.data[rna.X.data > 5] = 5

    return rna

class CustomDataset(Dataset):
    def __init__(self, rna, atac, labels):
        self.rna = rna
        self.atac = atac
        self.labels = labels

    def __len__(self):
        return self.rna.shape[0]

    def __getitem__(self, idx):
        return self.atac[idx].toarray().flatten(), self.rna[idx].toarray().flatten(), self.labels[idx]


class CustomDataset1(Dataset):
    def __init__(self, rna, atac, labels, latent):
        self.rna = rna
        self.atac = atac
        self.labels = labels
        self.latent = latent

    def __len__(self):
        return self.rna.shape[0]

    def __getitem__(self, idx):
        return self.atac[idx].toarray().flatten(), self.rna[idx].toarray().flatten(), self.labels[idx], self.latent[idx]


