#!/usr/bin/env python
"""
Predict cell types using a trained scMomer classifier.

Usage:
    python predict_celltype.py \
        --h5ad new_data.h5ad \
        --checkpoint ./output_celltype/model.pt \
        --label_classes ./output_celltype/label_classes.npy \
        --outdir ./predictions/ \
        --device 0
"""

import os, sys, types, logging, argparse
import numpy as np
import scipy.sparse
import anndata as ad
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SCMOMER_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scmomer_origin")
sys.path.insert(0, SCMOMER_ROOT)

from model.scmomer import scMomer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ShallowMLPEncoder(nn.Module):
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


class Classifier(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.linear1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(64, class_num)

    def forward(self, feature):
        out = self.linear1(feature)
        out = self.dropout1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class CellDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        if scipy.sparse.issparse(x):
            x = np.asarray(x.todense(), dtype=np.float32).flatten()
        else:
            x = np.asarray(x, dtype=np.float32).flatten()
        return torch.from_numpy(x)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--h5ad", required=True, help="Input h5ad (RNA, already processed)")
    p.add_argument("--checkpoint", required=True, help="Trained classifier checkpoint (model.pt)")
    p.add_argument("--label_classes", required=True, help="Label classes file (label_classes.npy)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # ---- Load data ----
    logger.info("Loading: %s", args.h5ad)
    adata = ad.read_h5ad(args.h5ad)
    logger.info("Shape: %s", adata.shape)
    rna_dim = adata.shape[1]

    dataset = CellDataset(adata.X)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ---- Load label classes ----
    label_classes = np.load(args.label_classes, allow_pickle=True)
    n_classes = len(label_classes)
    logger.info("Label classes: %d", n_classes)

    # ---- Build model ----
    args_model = types.SimpleNamespace(projection_dim=128, normalize=True)
    encoder = ShallowMLPEncoder(input_dim=rna_dim, output_dim=128)

    model = scMomer(
        args_model,
        atac_config=None,
        rna_decoder=None,
        atac_decoder=None,
        encoder=encoder,
    )
    model.sub_task = Classifier(class_num=n_classes)

    # ---- Load checkpoint ----
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    clean_sd = {}
    for k, v in ckpt.items():
        clean_sd[k[7:] if k.startswith('module.') else k] = v
    model.load_state_dict(clean_sd, strict=False)

    model = model.to(device)
    model.eval()

    # ---- Predict ----
    logger.info("Predicting...")
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x in tqdm(loader, desc="Predicting"):
            x = x.to(device)
            logits = model(None, x, mode='one', reconstruct=False)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    pred_labels = label_classes[all_preds]

    # ---- Save results ----
    adata.obs['predicted_cell_type'] = pred_labels
    adata.obs['prediction_confidence'] = all_probs.max(axis=1)

    out_h5ad = args.h5ad.replace('.h5ad', '_predicted.h5ad')
    adata.write_h5ad(out_h5ad)
    logger.info("Predictions saved to: %s", out_h5ad)

    logger.info("Prediction distribution:\n%s",
                adata.obs['predicted_cell_type'].value_counts().to_string())

    logger.info("Done.")


if __name__ == "__main__":
    main()
