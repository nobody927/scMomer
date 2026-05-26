"""
RNA -> ATAC cross-modal translation (train + validation).

Supports two input modes:
  1) Separate h5ad files:  --rna rna.h5ad --atac atac.h5ad
  2) Single h5mu file:     --data_path data.h5mu

Usage:
    python train_translation.py --data_path data.h5mu --checkpoint model.pth --outdir ./output
    python train_translation.py --rna rna.h5ad --atac atac.h5ad --checkpoint model.pth --outdir ./output
"""

import os, copy, logging, argparse, types
import numpy as np
import scipy.sparse
import anndata as ad
import scanpy as sc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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
    """
    Load RNA + ATAC data. Two modes:

    Mode 1 (separate h5ad):
        --rna rna.h5ad --atac atac.h5ad
        Aligns by cell barcode intersection.

    Mode 2 (single h5mu):
        --data_path data.h5mu
        Extracts 'rna' and 'atac' modalities.
    """
    if data_path is not None:
        # Mode 2: h5mu
        import muon as mu
        logger.info("Loading h5mu: %s", data_path)
        mdata = mu.read(data_path)
        rna = mdata.mod['rna'].copy()
        atac_raw = mdata.mod['atac'].copy()
        logger.info("Binarizing ATAC ...")
        atac = binarize_atac(atac_raw, atac_min_cells)
    elif rna_path is not None and atac_path is not None:
        # Mode 1: separate h5ad
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


def split_data(rna, atac, val_ratio=0.1, seed=42):
    """Split into train/val sets. Returns dense numpy arrays."""
    X_rna = to_dense(rna.X)
    y = to_dense(atac.X)

    n_cells = X_rna.shape[0]
    indices = np.arange(n_cells)
    tr_idx, va_idx = train_test_split(
        indices, test_size=val_ratio, random_state=seed)

    logger.info("Split: train=%d, val=%d", len(tr_idx), len(va_idx))

    data = {
        'X_rna_train': X_rna[tr_idx], 'y_train': y[tr_idx],
        'X_rna_val': X_rna[va_idx],   'y_val': y[va_idx],
        'rna_dim': X_rna.shape[1],     'atac_dim': y.shape[1],
    }
    return data


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
    """Load pretrained scMomer checkpoint with StudentEncoder."""
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

    valid_keys = set(model.state_dict().keys())
    unmatched = [k for k in clean_sd if k not in valid_keys]
    missing = [k for k in valid_keys if k not in clean_sd]
    logger.info("  Loaded %d keys  Unmatched: %d  Missing: %d",
                len(clean_sd), len(unmatched), len(missing))
    return model


def freeze_rna_model(model):
    """Freeze RNA encoder, then selectively unfreeze norm + last performer layer + to_out."""
    for param in model.rna_model.model.parameters():
        param.requires_grad = False

    for param in model.rna_model.model.norm.parameters():
        param.requires_grad = True
    for param in model.rna_model.model.performer.net.layers[-2].parameters():
        param.requires_grad = True
    for param in model.rna_model.model.to_out.parameters():
        param.requires_grad = True
    for param in model.encoder.parameters():
        param.requires_grad = False
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Total params: %d  Trainable: %d (%.1f%%)",
                total, trainable, 100.0 * trainable / total)


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

    return float(np.mean(auc_list)) if auc_list else 0.0


def evaluate_on_set(model, X_rna, y_true, batch_size, device):
    """Run model, compute AUROC on the given set."""
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(X_rna), batch_size):
            rna_t = torch.tensor(X_rna[s:s + batch_size], dtype=torch.float32, device=device)
            out = model(None, rna_t, mode='one')
            preds.append(torch.sigmoid(out).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return compute_auc(preds, y_true)


# ====================================================================
# Training
# ====================================================================
def train(model, X_train_rna, y_train, X_val_rna, y_val,
          pos_weight, lr, batch_size, max_epochs, patience, device,
          save_path):
    """Train with early stopping on validation AUC."""
    model = model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    train_ds = TensorDataset(
        torch.tensor(X_train_rna),
        torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(
        torch.tensor(X_val_rna),
        torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_auc = 0.0
    best_state = None
    wait = 0

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} [train]", leave=False)
        for rna_b, atac_b in pbar:
            rna_b, atac_b = rna_b.to(device), atac_b.to(device)
            out = model(None, rna_b, mode='one')
            loss = criterion(out, atac_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            train_loss += loss.item() * rna_b.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
        train_loss /= len(train_ds)

        # Validate
        val_auc = evaluate_on_set(model, X_val_rna, y_val, batch_size, device)
        scheduler.step(val_auc)

        logger.info("  Epoch %3d  train_loss=%.4f  val_AUC=%.4f  lr=%.2e  %s",
                     epoch, train_loss, val_auc,
                     optimizer.param_groups[0]['lr'],
                     "*" if val_auc > best_auc else "")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            torch.save({
                'model_state_dict': best_state,
                'val_auc': best_auc,
                'epoch': epoch,
            }, save_path)
            logger.info("  Model saved (AUC=%.4f)", best_auc)
        else:
            wait += 1
            if wait >= patience:
                logger.info("  Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Best val AUC: %.4f", best_auc)
    return model


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
    p.add_argument("--checkpoint", required=True, help="Pretrained model checkpoint (.pth)")
    p.add_argument("--projection_dim", type=int, default=128)

    # Training
    p.add_argument("--outdir", default="./output_translation/", help="Output directory")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    p.add_argument("--pos_weight", type=float, default=8, help="BCE positive class weight")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load data ----
    rna, atac = load_data(rna_path=args.rna, atac_path=args.atac, data_path=args.data_path)
    data = split_data(rna, atac, val_ratio=args.val_ratio, seed=args.seed)

    # ---- Load pretrained model ----
    logger.info("Loading pretrained model ...")
    model = load_model(
        args.checkpoint, data['rna_dim'], data['atac_dim'],
        projection_dim=args.projection_dim, device=device)
    logger.info("Freezing RNA encoder ...")
    freeze_rna_model(model)

    # ---- Train ----
    save_path = os.path.join(args.outdir, "translation_model.pth")
    model = train(
        model,
        data['X_rna_train'], data['y_train'],
        data['X_rna_val'], data['y_val'],
        pos_weight=args.pos_weight, lr=args.lr,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        patience=args.patience, device=device,
        save_path=save_path,
    )

    logger.info("Done. Model saved to: %s", save_path)


if __name__ == "__main__":
    main()
