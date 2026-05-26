"""
Distill a student FFN to mimic the teacher's latent embeddings.

Usage:
    python train_distill_single.py \
        --data_path /path/to/data.h5mu \
        --emb_dir /path/to/embeddings/ \
        --ckpt_dir /path/to/save/
"""

import torch
import torch.nn as nn
import argparse
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from model.read_data import prepare_train_val_data
from utils import seed_all


class StudentEncoder(nn.Module):
    """Shallow MLP: RNA -> latent embedding (1 hidden layer)."""

    def __init__(self, input_dim=16906, output_dim=128, hidden_dim=256, dropout=0.1):
        super(StudentEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, rna_value):
        return self.encoder(rna_value)


class StudentDataset(Dataset):
    """Student training dataset: RNA expression + target latent embedding."""

    def __init__(self, rna_data, atac_emb):
        self.rna = torch.from_numpy(rna_data).float()
        self.atac_emb = torch.from_numpy(atac_emb).float()

    def __getitem__(self, index):
        return self.rna[index], self.atac_emb[index]

    def __len__(self):
        return len(self.rna)


def evaluate(model, val_loader, loss_fn, device):
    """Validation loop."""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for rna, atac_emb in val_loader:
            rna = rna.to(device)
            atac_emb = atac_emb.to(device)

            pred = model(rna)
            loss = loss_fn(pred, atac_emb)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def train(model, train_loader, val_loader,
          loss_fn, optimizer, scheduler, device,
          max_epochs, save_path):
    """Train with early stopping on validation loss."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params:,} (trainable: {trainable_params:,})")
    print(f"   Train: {len(train_loader.dataset)} samples")
    print(f"   Val: {len(val_loader.dataset)} samples")

    best_val_loss = float("inf")
    patience = 0
    max_patience = 10

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"   Epoch {epoch}/{max_epochs}")
        for rna, atac_emb in pbar:
            rna, atac_emb = rna.to(device), atac_emb.to(device)
            pred = model(rna)
            loss = loss_fn(pred, atac_emb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        epoch_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(f"   Epoch {epoch} | Train: {epoch_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, save_path)
            print(f"   Model saved (Val Loss: {val_loss:.6f})")
        else:
            patience += 1
            if patience >= max_patience:
                print(f"   Early stopping.")
                break

    # Final evaluation on val set with best model
    print(f"\n   Final validation evaluation...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device)['model_state_dict'])

    model.eval()
    all_preds = []
    all_targets = []
    val_loss = 0.0

    with torch.no_grad():
        for rna, atac_emb in val_loader:
            rna, atac_emb = rna.to(device), atac_emb.to(device)
            pred = model(rna)
            val_loss += loss_fn(pred, atac_emb).item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(atac_emb.cpu().numpy())

    val_loss /= len(val_loader)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    val_correlation = np.corrcoef(all_preds.flatten(), all_targets.flatten())[0, 1]

    print(f"   Val MSE: {val_loss:.6f} | Val Correlation: {val_correlation:.6f}")

    return val_loss, val_correlation


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path", required=True, help="Path to h5mu data")
    p.add_argument("--emb_dir", required=True, help="Directory containing train/val embeddings (.npy)")
    p.add_argument("--ckpt_dir", default="./student_model/", help="Directory to save student model")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--projection_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension of student FFN")
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=2024)
    return p


def main():
    args = build_parser().parse_args()
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"Device: {device}")

    # ==========================================
    # 1. Load data (RNA expression)
    # ==========================================
    print("1. Loading data...")
    train_data, val_data, label_dict = prepare_train_val_data(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        return_mod='multi',
        cell_type_col='rna:cell_type'
    )

    train_rna_np = np.array(train_data.rna.todense()) if hasattr(train_data.rna, 'todense') else np.array(train_data.rna)
    val_rna_np = np.array(val_data.rna.todense()) if hasattr(val_data.rna, 'todense') else np.array(val_data.rna)
    print(f"   Train RNA: {train_rna_np.shape}")
    print(f"   Val RNA: {val_rna_np.shape}")

    # ==========================================
    # 2. Load target embeddings (from get_latent.py)
    # ==========================================
    print("2. Loading target embeddings...")
    train_emb = np.load(os.path.join(args.emb_dir, "train_embeddings.npy"))
    val_emb = np.load(os.path.join(args.emb_dir, "val_embeddings.npy"))
    print(f"   Train embedding: {train_emb.shape}")
    print(f"   Val embedding: {val_emb.shape}")

    # ==========================================
    # 3. Build model
    # ==========================================
    rna_dim = train_rna_np.shape[1]
    model = StudentEncoder(
        input_dim=rna_dim,
        output_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    # ==========================================
    # 4. Train
    # ==========================================
    print("3. Training student model...")
    train_loader = DataLoader(
        StudentDataset(train_rna_np, train_emb),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        StudentDataset(val_rna_np, val_emb),
        batch_size=args.batch_size,
        shuffle=False
    )

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    save_path = os.path.join(args.ckpt_dir, "student_ffn.pth")
    val_loss, val_corr = train(
        model, train_loader, val_loader,
        loss_fn, optimizer, scheduler, device,
        args.epoch, save_path
    )

    print(f"\nDone. Best model saved to: {save_path}")


if __name__ == "__main__":
    main()
