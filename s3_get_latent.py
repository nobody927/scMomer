"""
Extract latent ATAC embeddings from a pretrained multimodal scMomer model.

Usage:
    python get_latent.py \
        --data_path /path/to/data.h5mu \
        --model_path /path/to/model.pth \
        --save_dir /path/to/save/ \
        --device 0
"""

import torch
import argparse
import numpy as np
import os
import anndata as ad
from tqdm import tqdm
from torch.utils.data import DataLoader
import types

from model.scmomer import scMomer
from model.atac_encoder import MAEConfig
from model.read_data import prepare_train_val_data
from utils import seed_all


def extract_embeddings(model, data_loader, device, desc="Extracting"):
    """Extract embeddings from the model's intermediate hidden layer."""
    all_embs = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader, desc=desc):
            atac = data[0].to(device)
            rna = data[1].to(device)
            emb = model(atac, rna)
            all_embs.append(emb.cpu())

    return torch.cat(all_embs, dim=0)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path", required=True, help="Path to h5mu data")
    p.add_argument("--model_path", required=True, help="Path to pretrained model checkpoint")
    p.add_argument("--save_dir", default="./embeddings/", help="Directory to save embeddings")
    p.add_argument("--batch_size", type=int, default=640)
    p.add_argument("--projection_dim", type=int, default=128)
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of training data")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--save_h5ad", action="store_true", help="Also save as h5ad with obs metadata")
    return p


def main():
    args = build_parser().parse_args()
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ==========================================
    # 1. Load data
    # ==========================================
    print("1. Loading data...")
    train_data, val_data, label_dict = prepare_train_val_data(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        return_mod='multi',
        cell_type_col='rna:cell_type'
    )
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # ==========================================
    # 2. Load model (no decoder heads needed)
    # ==========================================
    print(f"2. Loading model: {args.model_path}")

    atac_config = MAEConfig(feature_size=train_loader.dataset.atac.shape[-1])

    model_args = types.SimpleNamespace(
        projection_dim=args.projection_dim,
        normalize=args.normalize,
    )

    model = scMomer(
        model_args,
        atac_config=atac_config,
        rna_decoder=None,
        atac_decoder=None,
        sub_task=None,
    )

    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # Strip DDP 'module.' prefix
    clean_sd = {}
    for key in state_dict:
        clean_sd[key[7:] if key.startswith('module.') else key] = state_dict[key]

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    print(f"   Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    # ==========================================
    # 3. Extract embeddings
    # ==========================================
    print("3. Extracting train embeddings...")
    train_emb = extract_embeddings(model, train_loader, device, desc="Train")
    print(f"   Shape: {train_emb.shape}")

    print("4. Extracting val embeddings...")
    val_emb = extract_embeddings(model, val_loader, device, desc="Val")
    print(f"   Shape: {val_emb.shape}")

    # ==========================================
    # 4. Save
    # ==========================================
    # Save as npy
    train_npy_path = os.path.join(args.save_dir, "train_embeddings.npy")
    val_npy_path = os.path.join(args.save_dir, "val_embeddings.npy")
    np.save(train_npy_path, train_emb.numpy())
    np.save(val_npy_path, val_emb.numpy())
    print(f"   Saved: {train_npy_path}")
    print(f"   Saved: {val_npy_path}")

    # Save as h5ad (with obs metadata)
    if args.save_h5ad:
        import muon as mu
        mdata = mu.read(args.data_path)

        train_adata = ad.AnnData(
            X=train_emb.numpy().astype(np.float32),
            obs=mdata[train_data.obs_names].obs.copy() if hasattr(train_data, 'obs_names') else None,
        )
        val_adata = ad.AnnData(
            X=val_emb.numpy().astype(np.float32),
            obs=mdata[val_data.obs_names].obs.copy() if hasattr(val_data, 'obs_names') else None,
        )

        train_h5ad_path = os.path.join(args.save_dir, "train_embeddings.h5ad")
        val_h5ad_path = os.path.join(args.save_dir, "val_embeddings.h5ad")
        train_adata.write_h5ad(train_h5ad_path)
        val_adata.write_h5ad(val_h5ad_path)
        print(f"   Saved: {train_h5ad_path}")
        print(f"   Saved: {val_h5ad_path}")

    print("Done.")


if __name__ == "__main__":
    main()
