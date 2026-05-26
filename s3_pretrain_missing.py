"""
Pretrain missing modality: train student encoder to mimic ATAC embedding from RNA-only input.

Losses:
  1. RNA reconstruction loss (MSE)
  2. ATAC reconstruction loss (BCEWithLogitsLoss)
  3. Distillation loss (MSE between student ATAC embedding and teacher ATAC embedding)

Model weights:
  - Load pretrained multimodal model (RNA encoder, to_out, decoders)
  - Student encoder: random init OR load from --encoder_path
  - Frozen: RNA encoder (ATAC encoder not used in mode='one')
  - Trainable: student encoder, to_out, RNA decoder, ATAC decoder

Usage:
    # Student encoder from scratch:
    torchrun --nproc_per_node=4 pretrain_missing.py --data_path data.h5mu \
        --model_path pretrained_multimodal.pth --train_latent train_emb.npy --val_latent val_emb.npy

    # Load pretrained student encoder:
    torchrun --nproc_per_node=4 pretrain_missing.py --data_path data.h5mu \
        --model_path pretrained_multimodal.pth --encoder_path student_encoder.pth \
        --train_latent train_emb.npy --val_latent val_emb.npy
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm

from model.scmomer import scMomer
from model.atac_encoder import MAEConfig
from model.read_data import prepare_train_val_data
from utils import seed_all, CosineAnnealingWarmupRestarts, get_reduced, save_ckpt


# ====================================================================
# Student Encoder
# ====================================================================
class StudentEncoder(nn.Module):
    """MLP encoder: RNA -> 128-dim ATAC-like embedding."""

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


# ====================================================================
# Dataset with pre-computed latent embeddings
# ====================================================================
class LatentDataset(Dataset):
    """Wraps a multimodal dataset, adding pre-computed teacher ATAC embeddings."""

    def __init__(self, base_dataset, latent_npy):
        """
        Args:
            base_dataset: CustomDataset with (atac, rna, labels)
            latent_npy: np.array [N, 128] pre-computed teacher ATAC embeddings
        """
        self.base = base_dataset
        self.latent = latent_npy.astype(np.float32)
        assert len(base_dataset) == len(latent_npy), \
            f"Dataset size {len(base_dataset)} != latent size {len(latent_npy)}"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        atac, rna, label = self.base[idx]
        return atac, rna, label, self.latent[idx]


# ====================================================================
# Parser
# ====================================================================
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    p.add_argument("--data_path", required=True, help="Path to h5mu data")
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--train_latent", required=True,
                   help="Path to pre-computed train embeddings .npy (from get_latent.py)")
    p.add_argument("--val_latent", required=True,
                   help="Path to pre-computed val embeddings .npy (from get_latent.py)")

    # Model
    p.add_argument("--model_path", required=True,
                   help="Pretrained multimodal model checkpoint (.pth)")
    p.add_argument("--encoder_path", type=str, default=None,
                   help="Pretrained student encoder checkpoint (.pth). If None, random init.")
    p.add_argument("--bin_num", type=int, default=5)
    p.add_argument("--gene_num", type=int, default=16906)
    p.add_argument("--projection_dim", type=int, default=128)
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)

    # Training
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_acc", type=int, default=10)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--pos_weight", type=float, default=5.0,
                   help="BCEWithLogitsLoss positive weight for ATAC loss")
    p.add_argument("--distill_weight", type=float, default=1.0,
                   help="Weight for distillation loss")

    # Output
    p.add_argument("--ckpt_dir", type=str, default='./saved_model/')
    p.add_argument("--model_name", type=str, default='pretrain_missing')
    p.add_argument("--seed", type=int, default=2024)

    return p


# ====================================================================
# DDP Setup
# ====================================================================
def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    return local_rank, rank, world_size


# ====================================================================
# Main
# ====================================================================
def main():
    args = build_parser().parse_args()

    local_rank, rank, world_size = setup_ddp()
    is_master = rank == 0
    device = torch.device(f'cuda:{local_rank}')
    seed_all(args.seed)

    CLASS = args.bin_num + 2

    # ==========================================
    # 1. Load data
    # ==========================================
    if is_master:
        print("1. Loading data...")
    train_data, val_data, label_dict = prepare_train_val_data(
        data_path=args.data_path,
        train_ratio=args.train_ratio,
        return_mod='multi',
        cell_type_col='rna:cell_type'
    )

    # Load pre-computed teacher embeddings
    train_latent = np.load(args.train_latent)
    val_latent = np.load(args.val_latent)
    if is_master:
        print(f"   Train latent: {train_latent.shape}, Val latent: {val_latent.shape}")

    train_dataset = LatentDataset(train_data, train_latent)
    val_dataset = LatentDataset(val_data, val_latent)

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    # ==========================================
    # 2. Build model
    # ==========================================
    if is_master:
        print("2. Building model...")

    atac_config = MAEConfig(feature_size=train_data.atac.shape[-1])

    # Import Reconstruct_net for decoders
    from utils import Reconstruct_net
    d_rna = Reconstruct_net(train_data.rna.shape[-1], args.projection_dim)
    d_atac = Reconstruct_net(train_data.atac.shape[-1], args.projection_dim)
    encoder = StudentEncoder(input_dim=args.gene_num, output_dim=args.projection_dim)

    model = scMomer(
        args,
        atac_config=atac_config,
        rna_decoder=d_rna,
        atac_decoder=d_atac,
        encoder=encoder,
    )

    # Load pretrained multimodal weights
    if is_master:
        print(f"   Loading multimodal weights: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location='cpu')
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    clean_sd = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if is_master:
        print(f"   Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        encoder_missing = [k for k in missing if k.startswith('encoder.')]
        if encoder_missing:
            print(f"   Student encoder not in checkpoint ({len(encoder_missing)} keys) — random init")

    # Load pretrained student encoder if provided
    if args.encoder_path:
        if is_master:
            print(f"   Loading student encoder: {args.encoder_path}")
        enc_ckpt = torch.load(args.encoder_path, map_location='cpu')
        enc_sd = enc_ckpt['model_state_dict'] if 'model_state_dict' in enc_ckpt else enc_ckpt
        enc_clean = {k[7:] if k.startswith('module.') else k: v for k, v in enc_sd.items()}
        # Filter to only encoder keys
        enc_keys = {k: v for k, v in enc_clean.items() if k.startswith('encoder.')}
        model.load_state_dict(enc_keys, strict=False)
        if is_master:
            print(f"   Loaded {len(enc_keys)} encoder keys")

    # Freeze all, then unfreeze what we train
    for param in model.parameters():
        param.requires_grad = False

    # Trainable: student encoder, to_out, RNA decoder, ATAC decoder
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.to_out.parameters():
        param.requires_grad = True
    for param in model.ATAC.parameters():
        param.requires_grad = True
    for param in model.RNA.parameters():
        param.requires_grad = True
    # Note: atac_model (ATAC encoder) is not used in mode='one', no need to freeze/unfreeze

    if is_master:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total: {total:,}  Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ==========================================
    # 3. Optimizer & scheduler
    # ==========================================
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=args.lr,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
    )

    # Losses (same as pretrain_multimodal.py)
    rna_loss_fn = nn.MSELoss()
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    atac_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    distill_loss_fn = nn.MSELoss()

    dist.barrier()

    # ==========================================
    # 4. Training loop
    # ==========================================
    max_loss = float('inf')
    patient = 0

    for i in range(1, args.epoch + 1):
        train_loader.sampler.set_epoch(i)
        model.train()
        dist.barrier()
        running_loss = 0.0
        total_batches = len(train_loader)

        for index, data in enumerate(train_loader):
            index += 1
            atac_raw = data[0].float().to(device, non_blocking=True)
            rna = data[1].to(device, non_blocking=True)
            latent_real = data[3].to(device, non_blocking=True)

            # ATAC target: binarized (same as pretrain_multimodal.py)
            atac_target = (atac_raw > 0).float()

            if index % args.grad_acc != 0:
                with model.no_sync():
                    r_atac, r_rna, atac_embeds = model(rna_values=rna, reconstruct=True, mode='one')
                    loss_atac = atac_loss_fn(r_atac, atac_target)
                    loss_rna = rna_loss_fn(r_rna, rna)
                    loss_distill = distill_loss_fn(atac_embeds, latent_real)
                    loss = (loss_atac + loss_rna + args.distill_weight * loss_distill) / args.grad_acc
                    loss.backward()
            else:
                r_atac, r_rna, atac_embeds = model(rna_values=rna, reconstruct=True, mode='one')
                loss_atac = atac_loss_fn(r_atac, atac_target)
                loss_rna = rna_loss_fn(r_rna, rna)
                loss_distill = distill_loss_fn(atac_embeds, latent_real)
                loss = (loss_atac + loss_rna + args.distill_weight * loss_distill) / args.grad_acc
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e4))
                optimizer.step()
                optimizer.zero_grad()

            running_loss += (loss_atac + loss_rna + loss_distill).item()

        epoch_loss = running_loss / total_batches
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)

        if is_master:
            print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f}  ==')
        dist.barrier()
        scheduler.step()

        # ---- Validation ----
        if i % 1 == 0:
            model.eval()
            dist.barrier()
            val_running_loss = 0.0

            with torch.no_grad():
                for val_index, data_v in enumerate(val_loader):
                    val_index += 1
                    atac_v = data_v[0].float().to(device, non_blocking=True)
                    rna_v = data_v[1].to(device, non_blocking=True)
                    latent_real_v = data_v[3].to(device, non_blocking=True)

                    atac_v_target = (atac_v > 0).float()
                    r_atac_v, r_rna_v, atac_embeds_v = model(rna_values=rna_v, reconstruct=True, mode='one')

                    loss_atac_v = atac_loss_fn(r_atac_v, atac_v_target)
                    loss_rna_v = rna_loss_fn(r_rna_v, rna_v)
                    loss_distill_v = distill_loss_fn(atac_embeds_v, latent_real_v)

                    val_running_loss += (loss_atac_v + loss_rna_v + loss_distill_v).item()

                val_loss = val_running_loss / val_index
                val_loss = get_reduced(val_loss, local_rank, 0, world_size)

            if is_master:
                print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f}  ==')

            # ---- Early stopping & save ----
            stop_flag = torch.tensor([0], dtype=torch.int, device=device)

            if is_master:
                if val_loss < max_loss:
                    max_loss = val_loss
                    patient = 0
                    save_ckpt(i, model, val_loss, args.model_name, args.ckpt_dir)
                    print(f'    == model saved at epoch: {i} ==')
                else:
                    patient += 1
                    if patient > args.patience:
                        stop_flag += 1

            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            if stop_flag.item() > 0:
                if is_master:
                    print(f' Early stopping triggered at epoch {i}.')
                break

    # ==========================================
    # 5. Save clean checkpoint (cell embedding only)
    # ==========================================
    if is_master:
        print("\n5. Saving clean checkpoint for cell representation...")
        # Build a minimal model to get valid keys
        import types as _types
        clean_args = _types.SimpleNamespace(
            projection_dim=args.projection_dim,
            normalize=args.normalize,
        )
        clean_encoder = StudentEncoder(input_dim=args.gene_num, output_dim=args.projection_dim)
        clean_model = scMomer(
            clean_args,
            atac_config=None,
            rna_decoder=None,
            atac_decoder=None,
            encoder=clean_encoder,
        )
        valid_keys = set(clean_model.state_dict().keys())

        # Extract matching keys from trained model (strip DDP 'module.' prefix)
        raw_sd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        clean_sd = {}
        for k, v in raw_sd.items():
            k2 = k[7:] if k.startswith('module.') else k
            if k2 in valid_keys:
                clean_sd[k2] = v

        clean_path = os.path.join(args.ckpt_dir, "pretrained_scmomer.pth")
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        torch.save({'model_state_dict': clean_sd}, clean_path)
        print(f"   Saved: {clean_path}  ({len(clean_sd)}/{len(valid_keys)} keys)")

        skipped = valid_keys - set(clean_sd.keys())
        if skipped:
            print(f"   Missing keys ({len(skipped)}): {sorted(skipped)}")


if __name__ == "__main__":
    main()
