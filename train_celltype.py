#!/usr/bin/env python
"""
Cell type classification with pretrained scMomer.
Loads a merged clean checkpoint, freezes most layers, trains a Classifier head.

Usage:
    python train_celltype.py \
        --h5ad data.h5ad \
        --checkpoint merged_clean.pth \
        --label_col cell_type \
        --outdir ./output_celltype \
        --device 0

Key points:
    - RNA .X  = already processed (no extra preprocessing)
    - Frozen: rna_model + encoder
    - Unfrozen: norm + last 2 performer layers + to_out + rna_projection + sub_task
    - Metrics: Accuracy, Macro F1
"""

import os, json, copy, logging, argparse, types
import numpy as np
import scipy.sparse
import anndata as ad
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.scmomer import scMomer
from utils import Reconstruct_net

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ====================================================================
# Model components
# ====================================================================
class StudentEncoder(nn.Module):
    """Shallow MLP: RNA -> ATAC representation (1 hidden layer)"""
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


# ====================================================================
# Dataset
# ====================================================================
class CellDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if scipy.sparse.issparse(x):
            x = np.asarray(x.todense(), dtype=np.float32).flatten()
        else:
            x = np.asarray(x, dtype=np.float32).flatten()
        return torch.from_numpy(x), self.y[idx]


# ====================================================================
# Model loading
# ====================================================================
def load_model(checkpoint_path, rna_dim, n_classes, device="cpu"):
    """Load merged checkpoint, attach Classifier head, freeze layers."""
    args = types.SimpleNamespace(projection_dim=128, normalize=True)
    encoder = StudentEncoder(input_dim=rna_dim, output_dim=128)

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
    logger.info("  Loaded %d keys", len(clean_sd))

    # Attach classifier head
    model.sub_task = Classifier(class_num=n_classes)
    logger.info("  Classifier: %d classes", n_classes)

    # Freeze
    # 1. Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # 2. Unfreeze norm
    for param in model.rna_model.model.norm.parameters():
        param.requires_grad = True

    # 3. Unfreeze last 2 performer layers
    for param in model.rna_model.model.performer.net.layers[-2].parameters():
        param.requires_grad = True

    # 4. Unfreeze to_out
    for param in model.rna_model.model.to_out.parameters():
        param.requires_grad = True

    # 5. Unfreeze rna_projection
    for param in model.rna_projection.parameters():
        param.requires_grad = True

    # 6. Unfreeze classifier
    for param in model.sub_task.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Total: %d  Trainable: %d (%.1f%%)", total, trainable, 100 * trainable / total)

    return model.to(device)


# ====================================================================
# Training
# ====================================================================
def train(model, train_loader, val_loader, lr, max_epochs, patience, device):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = -1
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [train]", leave=False)
        for x, y in train_bar:
            x, y = x.to(device), y.to(device)
            logits = model(None, x, mode='one', reconstruct=False)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
        train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(None, x, mode='one', reconstruct=False)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        improved = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            improved = " *"
        else:
            wait += 1

        if (epoch + 1) % 5 == 0 or wait == 0:
            logger.info("    Epoch %3d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  val_f1=%.4f  wait=%d%s",
                        epoch + 1, train_loss, val_loss, val_acc, val_f1, wait, improved)

        if wait >= patience:
            logger.info("    Early stopping at epoch %d", epoch + 1)
            break

    model.load_state_dict(best_state)
    return model, history, best_acc


# ====================================================================
# Main
# ====================================================================
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--h5ad", required=True, help="Input h5ad (RNA, already processed)")
    p.add_argument("--checkpoint", required=True, help="Merged clean checkpoint")
    p.add_argument("--label_col", default="cell_type", help="Column in obs for labels")
    p.add_argument("--outdir", default="./output_celltype/")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batchsize", type=int, default=6)
    p.add_argument("--max_epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5)
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
    logger.info("Loading: %s", args.h5ad)
    adata = ad.read_h5ad(args.h5ad)
    logger.info("Shape: %s", adata.shape)

    # Find label column
    label_col = args.label_col
    if label_col not in adata.obs.columns:
        candidates = ["cell_type", "CellType", "celltype", "cell_type_major",
                       "cell_type_minor", "cluster", "annotation", "labels"]
        for c in candidates:
            if c in adata.obs.columns:
                label_col = c
                break
    logger.info("Label column: '%s'", label_col)

    le = LabelEncoder()
    labels = le.fit_transform(adata.obs[label_col].astype(str).values)
    n_classes = len(le.classes_)
    logger.info("Classes: %d  Samples: %d", n_classes, len(labels))
    logger.info("Label distribution:\n%s", adata.obs[label_col].value_counts().to_string())

    X = adata.X
    rna_dim = adata.shape[1]

    # ---- Train/Val split ----
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_ratio, random_state=args.seed, stratify=labels
    )
    logger.info("Train: %d  Val: %d", len(train_idx), len(val_idx))

    train_ds = CellDataset(X[train_idx], labels[train_idx])
    val_ds = CellDataset(X[val_idx], labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batchsize, shuffle=False, num_workers=0)

    # ---- Train ----
    model = load_model(args.checkpoint, rna_dim, n_classes, device=device)

    model, history, best_val_acc = train(
        model, train_loader, val_loader,
        lr=args.lr, max_epochs=args.max_epochs,
        patience=args.patience, device=device,
    )

    # ---- Evaluate on validation set ----
    val_acc, val_f1 = evaluate(model, val_loader, device)
    logger.info("Validation  ACC=%.4f  Macro F1=%.4f", val_acc, val_f1)

    # ---- Save ----
    pd.DataFrame(history).to_csv(os.path.join(args.outdir, "loss_history.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pt"))

    metrics = {
        "best_val_acc": float(best_val_acc),
        "val_acc": float(val_acc),
        "val_f1": float(val_f1),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_epochs": len(history["train_loss"]),
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save label encoder classes
    np.save(os.path.join(args.outdir, "label_classes.npy"), le.classes_)

    logger.info("Done -> %s", args.outdir)


def evaluate(model, loader, device):
    """Evaluate on a loader, return accuracy and macro F1."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(None, x, mode='one', reconstruct=False)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')


if __name__ == "__main__":
    main()
