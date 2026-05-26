"""
Evaluate a trained RNA -> ATAC translation model.

Supports two input modes:
  1) Separate h5ad files:  --rna rna.h5ad --atac atac.h5ad
  2) Single h5mu file:     --data_path data.h5mu

Usage:
    python evaluate_translation.py --data_path data.h5mu --checkpoint translation_model.pth
    python evaluate_translation.py --rna rna.h5ad --atac atac.h5ad --checkpoint translation_model.pth
"""

import os, argparse, types, logging
import numpy as np
import scipy.sparse
import anndata as ad
import scanpy as sc
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model.scmomer import scMomer
from utils import Reconstruct_net

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ====================================================================
# Data
# ====================================================================
def to_dense(X):
    if scipy.sparse.issparse(X):
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def binarize_atac(adata_raw, min_cells=3):
    """Binarize ATAC (non-zero -> 1). Filter genes by min_cells."""
    adata = adata_raw.copy()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    X = adata.X
    if scipy.sparse.issparse(X):
        X = X.copy()
        X.data[:] = 1.0
    else:
        X = (X > 0).astype(np.float32)
    adata.X = X
    return adata


def load_data(rna_path=None, atac_path=None, data_path=None, atac_min_cells=3):
    """Load RNA + ATAC data (h5mu or separate h5ad)."""
    if data_path is not None:
        import muon as mu
        logger.info("Loading h5mu: %s", data_path)
        mdata = mu.read(data_path)
        rna = mdata.mod['rna'].copy()
        atac_raw = mdata.mod['atac'].copy()
        logger.info("Binarizing ATAC ...")
        atac = binarize_atac(atac_raw, atac_min_cells)
    elif rna_path is not None and atac_path is not None:
        logger.info("Loading RNA: %s", rna_path)
        rna = ad.read_h5ad(rna_path)
        logger.info("Loading ATAC: %s", atac_path)
        atac_raw = ad.read_h5ad(atac_path)
        common = rna.obs_names.intersection(atac_raw.obs_names)
        logger.info("Common cells: %d", len(common))
        rna = rna[common].copy()
        atac_raw = atac_raw[common].copy()
        logger.info("Binarizing ATAC ...")
        atac = binarize_atac(atac_raw, atac_min_cells)
    else:
        raise ValueError("Provide either --data_path (h5mu) or both --rna and --atac (h5ad).")

    logger.info("Data: %d cells, RNA %d features, ATAC %d peaks",
                rna.shape[0], rna.shape[1], atac.shape[1])
    return rna, atac


# ====================================================================
# Model
# ====================================================================
class StudentEncoder(nn.Module):
    """MLP encoder: RNA -> projection_dim (1 hidden layer)."""

    def __init__(self, input_dim=16906, output_dim=128, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, rna_value):
        return self.encoder(rna_value)


def load_model(checkpoint_path, rna_dim, atac_dim, projection_dim=128,
               normalize=True, device="cpu"):
    """Load trained translation model checkpoint."""
    args = types.SimpleNamespace(
        projection_dim=projection_dim,
        normalize=normalize,
    )
    d_atac = Reconstruct_net(atac_dim, projection_dim)
    encoder = StudentEncoder(input_dim=rna_dim, output_dim=projection_dim)

    model = scMomer(
        args,
        atac_config=None,
        rna_decoder=None,
        atac_decoder=None,
        encoder=encoder,
    )

    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    clean_sd = {}
    for k, v in sd.items():
        clean_sd[k[7:] if k.startswith('module.') else k] = v

    model.load_state_dict(clean_sd, strict=False)
    model.ATAC = d_atac
    model = model.to(device)
    model.eval()

    valid_keys = set(model.state_dict().keys())
    unmatched = [k for k in clean_sd if k not in valid_keys]
    missing = [k for k in valid_keys if k not in clean_sd]
    logger.info("  Loaded %d keys  Unmatched: %d  Missing: %d",
                len(clean_sd), len(unmatched), len(missing))
    return model


# ====================================================================
# Evaluation: AUC
# ====================================================================
def compute_auc(preds, truth):
    """Per-cell AUROC. Skip cells with only one class."""
    n_cells = truth.shape[0]
    auc_list = []

    for i in range(n_cells):
        y_true = (truth[i] > 0).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        auc_list.append(roc_auc_score(y_true, preds[i]))

    return float(np.mean(auc_list)) if auc_list else 0.0, len(auc_list)


def predict(model, X_rna, batch_size, device):
    """Run model inference, return sigmoid predictions."""
    model.eval()
    preds = []
    with torch.no_grad():
        for s in tqdm(range(0, len(X_rna), batch_size), desc="Predicting"):
            rna_t = torch.tensor(X_rna[s:s + batch_size], dtype=torch.float32, device=device)
            out = model(None, rna_t, mode='one')
            preds.append(torch.sigmoid(out).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ====================================================================
# Main
# ====================================================================
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Input data (two modes)
    p.add_argument("--rna", default=None, help="RNA h5ad path (use with --atac)")
    p.add_argument("--atac", default=None, help="ATAC h5ad path (use with --rna)")
    p.add_argument("--data_path", default=None, help="h5mu path (alternative to --rna/--atac)")

    # Model
    p.add_argument("--checkpoint", required=True, help="Trained translation model checkpoint (.pth)")
    p.add_argument("--projection_dim", type=int, default=128)

    # Evaluation
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load data ----
    rna, atac = load_data(rna_path=args.rna, atac_path=args.atac, data_path=args.data_path)
    X_rna = to_dense(rna.X)
    y_true = to_dense(atac.X)
    rna_dim = X_rna.shape[1]
    atac_dim = y_true.shape[1]
    logger.info("Test set: %d cells, RNA=%d, ATAC=%d", X_rna.shape[0], rna_dim, atac_dim)

    # ---- Load model ----
    model = load_model(
        args.checkpoint, rna_dim, atac_dim,
        projection_dim=args.projection_dim, device=device)

    # ---- Predict ----
    logger.info("Running inference ...")
    preds = predict(model, X_rna, args.batch_size, device)
    logger.info("Predictions shape: %s", preds.shape)

    # ---- Evaluate ----
    auc, n_eval = compute_auc(preds, y_true)
    logger.info("Results:")
    logger.info("  Evaluated cells: %d / %d", n_eval, X_rna.shape[0])
    logger.info("  Mean AUROC: %.4f", auc)

    print(f"\n{'=' * 40}")
    print(f"  Mean AUROC: {auc:.4f}  ({n_eval} cells)")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
