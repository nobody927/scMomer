from torch.utils.data import Dataset, random_split, Subset
import muon as mu
import scanpy as sc
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import anndata as ad
import torch
from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import os
import random
import hickle as hkl
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_geometric.utils import dense_to_sparse


def prepare_fixed_eval_data(data_path,
                            fraction=0.8,
                            return_mod='rna',
                            val_ratio=0.1,
                            test_ratio=0.1,
                            cell_type_col='rna:cell_type',
                            seed=2024):
    """
    Logic:
    1. Always split out fixed val_ratio and test_ratio (e.g. 10% each) from the total data.
    2. The remaining 80% serves as the “training pool”.
    3. Sample the corresponding fraction from the training pool as the training set.
    """

    mdata = mu.read(data_path)
    atac = process_atac(mdata)
    mdata.mod["atac"] = atac
    rna = process_rna(mdata)
    mdata.mod["rna"] = rna
    cell_barcodes = mdata.obs.index.values
    cell_types = mdata.obs[cell_type_col].values

    # Step 1: split out fixed test set (10%)
    remain_idx, test_idx, remain_types, test_types = train_test_split(
        cell_barcodes, cell_types,
        test_size=test_ratio,
        random_state=seed,
        stratify=cell_types
    )

    # Step 2: split out fixed validation set from the remainder (10% / 90% ≈ 11.1%)
    # Use ratio conversion to ensure the absolute proportion is 10% of the original
    actual_val_size = val_ratio / (1 - test_ratio)
    pool_idx, val_idx, pool_types, val_types = train_test_split(
        remain_idx, remain_types,
        test_size=actual_val_size,
        random_state=seed,
        stratify=remain_types
    )

    # Step 3: sample the desired fraction from the remaining 80% training pool
    # Since fraction is relative to the total, convert to the proportion relative to the pool
    # e.g. to sample 40% of the original, sample 50% from the remaining 80% pool
    actual_train_size = fraction / (1 - val_ratio - test_ratio)

    # Prevent fraction from exceeding the pool size (e.g. 90% requested but pool is only 80%)
    if actual_train_size >= 1.0:
        train_idx = pool_idx
    else:
        train_idx, _, _, _ = train_test_split(
            pool_idx, pool_types,
            train_size=actual_train_size,
            random_state=seed,
            stratify=pool_types
        )

    print(f"--- Sensitivity analysis (fixed eval set) ---")
    print(f"Training fraction: {fraction * 100}%")
    print(f"Training set size: {len(train_idx)}")
    print(f"Validation set size: {len(val_idx)} (fixed)")
    print(f"Test set size: {len(test_idx)} (fixed)")

    if return_mod == 'multi':
        unique_labels = np.unique(cell_types)
        label_dict = {label: i for i, label in enumerate(unique_labels)}

        # Helper function: extract sparse matrices and convert labels
        def get_mod_data_and_labels(indices):
            sub_mdata = mdata[indices]
            rna_X = sub_mdata.mod['rna'].X
            atac_X = sub_mdata.mod['atac'].X
            # Convert string labels to integer labels
            labels = np.array([label_dict[l] for l in sub_mdata.obs[cell_type_col].values], dtype=np.int64)
            return rna_X, atac_X, labels

        # Get data
        train_rna, train_atac, train_labels = get_mod_data_and_labels(train_idx)
        val_rna, val_atac, val_labels = get_mod_data_and_labels(val_idx)
        test_rna, test_atac, test_labels = get_mod_data_and_labels(test_idx)

        # Wrap as CustomDataset
        train_dataset = CustomDataset(train_rna, train_atac, train_labels)
        val_dataset = CustomDataset(val_rna, val_atac, val_labels)
        test_dataset = CustomDataset(test_rna, test_atac, test_labels)

        # Return label_dict together so that 0, 1, 2 can be mapped back to real cell types during inference
        return train_dataset, val_dataset, test_dataset, label_dict
    else:
        target = mdata.mod[return_mod]

    return target[train_idx], target[val_idx], target[test_idx]

def prepare_train_val_data(data_path,
                           train_ratio=0.9,
                           return_mod='rna',
                           cell_type_col='rna:cell_type',
                           seed=2024):
    """
    Split data into train and val sets (no test set).

    Args:
        data_path: path to h5mu file
        train_ratio: proportion of data used for training (default 0.9)
        return_mod: 'rna', 'atac', or 'multi'
        cell_type_col: column name for cell type labels
        seed: random seed

    Returns:
        train_dataset, val_dataset (if return_mod='multi', also returns label_dict)
    """
    mdata = mu.read(data_path)
    atac = process_atac(mdata)
    mdata.mod["atac"] = atac
    rna = process_rna(mdata)
    mdata.mod["rna"] = rna
    cell_barcodes = mdata.obs.index.values
    cell_types = mdata.obs[cell_type_col].values
    try:
        train_idx, val_idx, train_types, val_types = train_test_split(
            cell_barcodes, cell_types,
            test_size=1 - train_ratio,
            random_state=seed,
            stratify=cell_types
        )
    except:
        train_idx, val_idx, train_types, val_types = train_test_split(
            cell_barcodes, cell_types,
            test_size=1 - train_ratio,
            random_state=seed,
            stratify=None
        )
    print(f"--- Train/Val split ---")
    print(f"Train ratio: {train_ratio * 100:.0f}%  ({len(train_idx)} samples)")
    print(f"Val ratio:   {(1 - train_ratio) * 100:.0f}%  ({len(val_idx)} samples)")

    if return_mod == 'multi':
        unique_labels = np.unique(cell_types)
        label_dict = {label: i for i, label in enumerate(unique_labels)}

        def get_mod_data_and_labels(indices):
            sub_mdata = mdata[indices]
            rna_X = sub_mdata.mod['rna'].X
            atac_X = sub_mdata.mod['atac'].X
            labels = np.array([label_dict[l] for l in sub_mdata.obs[cell_type_col].values], dtype=np.int64)
            return rna_X, atac_X, labels

        train_rna, train_atac, train_labels = get_mod_data_and_labels(train_idx)
        val_rna, val_atac, val_labels = get_mod_data_and_labels(val_idx)

        train_dataset = CustomDataset(train_rna, train_atac, train_labels)
        val_dataset = CustomDataset(val_rna, val_atac, val_labels)

        return train_dataset, val_dataset, label_dict
    else:
        target = mdata.mod[return_mod]
        return target[train_idx], target[val_idx]


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
        sss = StratifiedShuffleSplit(n_splits=1, train_size=split, random_state=14)
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
        train_data, val_data = Subset(mdata, train_idx), Subset(mdata, val_idx)
        train_label, val_label = Subset(label, train_idx), Subset(label, val_idx)

    elif isinstance(split, list):  # split multiple cell types # TO DO
        train_idx = mdata.obs[mdata.obs["cell_type"].isin(split)].index
        val_idx = mdata.obs[~mdata.obs["cell_type"].isin(split)].index
        train_data, val_data = Subset(mdata, train_idx), Subset(mdata, val_idx)
        train_label, val_label = Subset(label, train_idx), Subset(label, val_idx)

    train_dataset = CustomDataset(train_rna, train_atac, label_train)
    val_dataset = CustomDataset(val_rna, val_atac, label_val)

    return train_dataset, val_dataset, label_dict

def process_atac(mdata):
    atac = mdata.mod['atac']
    atac.X.data[atac.X.data > 0] = 1
    return atac

def process_rna(mdata):
    rna = mdata.mod["rna"]
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna, base=2)
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


# ====================================================================
# Drug Response Data Loading
# ====================================================================

TCGA_LABEL_SET = [
    "ALL", "BLCA", "BRCA", "CESC", "DLBC", "LIHC", "LUAD",
    "ESCA", "GBM", "HNSC", "KIRC", "AML", "LCML", "LGG",
    "LUSC", "MESO", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
    "STAD", "THCA", "COAD/READ",
]


def align_gene_expression(gexpr_df, panglao_path):
    """Align 697-gene DeepCDR expression to 16906 model gene space.

    Genes present in panglao but missing in DeepCDR are filled with 0.
    Genes in DeepCDR but not in panglao are dropped.
    After alignment, applies normalize_total(1e4) + log1p(base=2) to match
    scMomer pretraining distribution.

    Args:
        gexpr_df: DataFrame [cell_lines x 697 genes], indexed by cell line ID.
        panglao_path: Path to panglao_10000.h5ad reference (var_names = 16906 genes).

    Returns:
        aligned_df: DataFrame [cell_lines x 16906 genes], normalized and log-transformed.
    """
    ref = ad.read_h5ad(panglao_path)
    ref_genes = list(ref.var_names)
    n_model_genes = len(ref_genes)

    gexpr_genes = list(gexpr_df.columns)
    gene2idx = {g: i for i, g in enumerate(gexpr_genes)}

    aligned = np.zeros((gexpr_df.shape[0], n_model_genes), dtype=np.float32)
    gexpr_vals = gexpr_df.values.astype(np.float32)

    matched = 0
    for j, gene in enumerate(ref_genes):
        idx = gene2idx.get(gene)
        if idx is not None:
            aligned[:, j] = gexpr_vals[:, idx]
            matched += 1

    print(f"Gene alignment: {matched}/{len(gexpr_genes)} DeepCDR genes matched "
          f"out of {n_model_genes} model genes (filled 0 for {n_model_genes - matched})")

    # Normalize to match scMomer pretraining: normalize_total(1e4) + log1p(base=2)
    cell_sums = aligned.sum(axis=1, keepdims=True)
    cell_sums[cell_sums == 0] = 1.0  # avoid division by zero
    aligned = aligned / cell_sums * 1e4
    aligned = np.log2(aligned + 1)

    return pd.DataFrame(aligned, index=gexpr_df.index, columns=ref_genes)


def load_drug_info(drug_info_file):
    """Load drug ID to pubchem ID mapping."""
    reader = csv.reader(open(drug_info_file, 'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}
    name2pubchemid = {item[1]: item[5] for item in rows if item[5].isdigit()}
    return drugid2pubchemid, name2pubchemid


def load_cell_line_info(cell_line_info_file):
    """Load cell line to cancer type mapping."""
    cellline2cancertype = {}
    for line in open(cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        cellline2cancertype[cellline_id] = TCGA_label
    return cellline2cancertype


def load_drug_features(drug_feature_dir):
    """Load drug molecular graph features (feat_mat, adj_list, degree_list)."""
    drug_feature = {}
    for fname in os.listdir(drug_feature_dir):
        pubchem_id = fname.split('.')[0]
        feat_mat, adj_list, degree_list = hkl.load(os.path.join(drug_feature_dir, fname))
        drug_feature[pubchem_id] = [feat_mat, adj_list, degree_list]
    return drug_feature


def load_ic50_thresholds(ic50_thred_file, name2pubchemid):
    """Load per-drug IC50 thresholds from file.

    File format:
        Line 0: drug names (tab-separated)
        Line 1: thresholds (tab-separated)

    Returns:
        dict {pubchem_id: threshold}
    """
    lines = open(ic50_thred_file).readlines()
    drug_names = [item.strip() for item in lines[0].split('\t')]
    IC50_threds = [float(item.strip()) for item in lines[1].split('\t')]
    drug2thred = {}
    for name, thred in zip(drug_names, IC50_threds):
        if name in name2pubchemid:
            drug2thred[name2pubchemid[name]] = thred
    print(f"IC50 thresholds: loaded {len(drug2thred)} drugs from {ic50_thred_file}")
    return drug2thred


def parse_drug_response(drug_info_file, cell_line_info_file,
                        cancer_response_file, drug_feature_dir,
                        classification=False, ic50_thred_file=None):
    """Parse drug response data, returning valid (cell_line, drug, value, cancer_type) tuples.

    Only keeps entries where the drug has graph features and the cell line exists.
    In classification mode, value is binary (1 if IC50 < per-drug threshold, 0 otherwise).
    Only samples whose drug has a threshold entry are kept in classification mode.

    Args:
        classification: If True, binarize labels using per-drug thresholds.
        ic50_thred_file: Path to IC50 threshold file (required if classification=True).
    """
    drugid2pubchemid, name2pubchemid = load_drug_info(drug_info_file)
    cellline2cancertype = load_cell_line_info(cell_line_info_file)

    drug_pubchem_id_set = set()
    for fname in os.listdir(drug_feature_dir):
        drug_pubchem_id_set.add(fname.split('.')[0])

    # Load per-drug thresholds for classification
    drug2thred = None
    if classification:
        assert ic50_thred_file is not None, "ic50_thred_file required for classification mode"
        drug2thred = load_ic50_thresholds(ic50_thred_file, name2pubchemid)

    experiment_data = pd.read_csv(cancer_response_file, sep=',', header=0, index_col=[0])
    drug_match_list = [item for item in experiment_data.index
                       if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]

    data_idx = []
    skipped_no_thred = 0
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if (str(pubchem_id) in drug_pubchem_id_set and
                    each_cellline in cellline2cancertype):
                val = experiment_data_filtered.loc[each_drug, each_cellline]
                if np.isnan(val):
                    continue
                if classification:
                    # Classification: binarize with per-drug threshold
                    if pubchem_id not in drug2thred:
                        skipped_no_thred += 1
                        continue
                    binary_label = 1 if float(val) < drug2thred[pubchem_id] else 0
                    data_idx.append((each_cellline, str(pubchem_id),
                                     binary_label, cellline2cancertype[each_cellline]))
                else:
                    # Regression: keep raw IC50
                    data_idx.append((each_cellline, str(pubchem_id),
                                     float(val), cellline2cancertype[each_cellline]))

    nb_celllines = len(set(item[0] for item in data_idx))
    nb_drugs = len(set(item[1] for item in data_idx))
    mode_str = "classification" if classification else "regression"
    print(f"Drug response ({mode_str}): {len(data_idx)} samples, "
          f"{nb_celllines} cell lines, {nb_drugs} drugs")
    if classification and skipped_no_thred > 0:
        print(f"  Skipped {skipped_no_thred} samples (drug has no threshold)")
    return data_idx


def split_drug_data(data_idx, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split drug response data into train/val/test by cancer type (stratified)."""
    random.seed(seed)
    data_train, data_val, data_test = [], [], []

    for cancer_type in TCGA_LABEL_SET:
        subtype = [item for item in data_idx if item[-1] == cancer_type]
        if len(subtype) == 0:
            continue
        # First split out test
        remain, test = train_test_split(
            subtype, test_size=test_ratio, random_state=seed) if len(subtype) > 2 else (subtype, [])
        # Then split val from remain
        if len(remain) > 2:
            actual_val = val_ratio / (1 - test_ratio)
            train, val = train_test_split(remain, test_size=actual_val, random_state=seed)
        else:
            train, val = remain, []

        data_train.extend(train)
        data_val.extend(val)
        data_test.extend(test)

    print(f"Drug split: train={len(data_train)}, val={len(data_val)}, test={len(data_test)}")
    return data_train, data_val, data_test


MAX_ATOMS = 100


def calculate_graph_feat(feat_mat, adj_list, max_atoms=MAX_ATOMS):
    """Pad drug graph features to fixed size and normalize adjacency."""
    feat = np.zeros((max_atoms, feat_mat.shape[-1]), dtype='float32')
    adj_mat = np.zeros((max_atoms, max_atoms), dtype='float32')
    feat[:feat_mat.shape[0], :] = feat_mat
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            adj_mat[i, int(j)] = 1
    # Normalize: D^{-1/2} A D^{-1/2}
    adj_sub = adj_mat[:len(adj_list), :len(adj_list)]
    adj_sub = adj_sub + np.eye(adj_sub.shape[0])
    d = np.power(np.array(adj_sub.sum(1)), -0.5).flatten()
    d[np.isinf(d)] = 0.0
    d_mat = np.diag(d)
    adj_norm = d_mat @ adj_sub @ d_mat
    adj_mat[:len(adj_list), :len(adj_list)] = adj_norm
    return feat, adj_mat


def drug_graphs_to_pyg(feat_mats, adj_mats):
    """Convert batch of drug features + adjacency to PyG Batch."""
    data_list = []
    for i in range(feat_mats.shape[0]):
        x = torch.from_numpy(feat_mats[i])
        adj = torch.from_numpy(adj_mats[i])
        edge_index, _ = dense_to_sparse(adj)
        data_list.append(PyGData(x=x, edge_index=edge_index))
    return PyGBatch.from_data_list(data_list)


class DrugResponseDataset(Dataset):
    """Dataset for drug response prediction.

    Each sample: (drug_graph, gene_expression, label).
    Gene expression is already aligned to 16906 model gene space.
    For classification, labels are pre-binarized (0/1) in data_idx.
    For regression, labels are raw IC50 values.
    """

    def __init__(self, data_idx, drug_feature, gexpr_aligned):
        """
        Args:
            data_idx: list of (cell_line, pubchem_id, value, cancer_type)
                value is IC50 (regression) or binary 0/1 (classification).
            drug_feature: dict {pubchem_id: [feat_mat, adj_list, degree_list]}
            gexpr_aligned: DataFrame [cell_lines x 16906], aligned gene expression
        """
        self.data_idx = data_idx
        self.drug_feature = drug_feature
        self.gexpr_aligned = gexpr_aligned

        # Pre-compute drug graphs
        n = len(data_idx)
        nb_drug_feat = next(iter(drug_feature.values()))[0].shape[1]
        self.feat_mats = np.zeros((n, MAX_ATOMS, nb_drug_feat), dtype='float32')
        self.adj_mats = np.zeros((n, MAX_ATOMS, MAX_ATOMS), dtype='float32')
        self.gexpr_data = np.zeros((n, gexpr_aligned.shape[1]), dtype='float32')
        self.targets = np.zeros(n, dtype='float32')

        for idx, (cell_line, pubchem_id, value, _) in enumerate(data_idx):
            feat_mat, adj_list, _ = drug_feature[pubchem_id]
            feat, adj = calculate_graph_feat(feat_mat, adj_list)
            self.feat_mats[idx] = feat
            self.adj_mats[idx] = adj
            if cell_line in gexpr_aligned.index:
                self.gexpr_data[idx] = gexpr_aligned.loc[cell_line].values
            self.targets[idx] = value

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        return (
            self.feat_mats[idx],
            self.adj_mats[idx],
            self.gexpr_data[idx],
            self.targets[idx],
        )


def collate_drug_batch(batch):
    """Collate function for DrugResponseDataset.

    Converts drug graphs to PyG Batch on-the-fly.
    """
    feat_mats, adj_mats, gexpr, targets = zip(*batch)
    feat_mats = np.stack(feat_mats)
    adj_mats = np.stack(adj_mats)
    drug_batch = drug_graphs_to_pyg(feat_mats, adj_mats)
    gexpr_tensor = torch.tensor(np.stack(gexpr), dtype=torch.float32)
    target_tensor = torch.tensor(np.stack(targets), dtype=torch.float32)
    return drug_batch, gexpr_tensor, target_tensor


def prepare_drug_data(drug_info_file, cell_line_info_file,
                      cancer_response_file, drug_feature_dir,
                      gexpr_file, panglao_path,
                      val_ratio=0.1, test_ratio=0.1, seed=42,
                      classification=False, ic50_thred_file=None):
    """Full pipeline: load, align, split, and create datasets for drug response.

    Args:
        drug_info_file: Path to Drug_info.csv
        cell_line_info_file: Path to Cell_lines_annotations.txt
        cancer_response_file: Path to GDSC_IC50.csv
        drug_feature_dir: Path to drug_graph_feat/ directory
        gexpr_file: Path to genomic_expression_561celllines_697genes_demap_features.csv
        panglao_path: Path to panglao_10000.h5ad (for gene alignment)
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        classification: If True, treat as classification task with per-drug thresholds
        ic50_thred_file: Path to IC50 threshold file (required if classification=True)

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # 1. Load gene expression and align to model gene space
    gexpr_df = pd.read_csv(gexpr_file, sep=',', header=0, index_col=[0])
    gexpr_aligned = align_gene_expression(gexpr_df, panglao_path)

    # 2. Load drug features
    drug_feature = load_drug_features(drug_feature_dir)

    # 3. Parse drug response pairs (classification applies per-drug thresholds here)
    data_idx = parse_drug_response(
        drug_info_file, cell_line_info_file,
        cancer_response_file, drug_feature_dir,
        classification=classification, ic50_thred_file=ic50_thred_file)

    # 4. Split
    data_train, data_val, data_test = split_drug_data(
        data_idx, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    # 5. Build datasets
    train_dataset = DrugResponseDataset(data_train, drug_feature, gexpr_aligned)
    val_dataset = DrugResponseDataset(data_val, drug_feature, gexpr_aligned)
    test_dataset = DrugResponseDataset(data_test, drug_feature, gexpr_aligned)

    return train_dataset, val_dataset, test_dataset
