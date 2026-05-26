# -*- coding: utf-8 -*-
"""
RNA pretraining with masked autoencoder (Performer + Gene2vec).

Three training modes:
  1) From scratch:  no --model_path, train all parameters
  2) Finetune:      --model_path provided, freeze most layers (norm + last 2 performer layers)
  3) Resume:        --resume provided, continue from saved checkpoint (optimizer/scheduler/epoch)

Usage:
    # From scratch (4 GPUs):
    torchrun --nproc_per_node=4 pretrain_rna.py --data_path data.h5mu

    # Finetune from pretrained:
    torchrun --nproc_per_node=4 pretrain_rna.py --data_path data.h5mu --model_path pretrained.pth

    # Resume from checkpoint:
    torchrun --nproc_per_node=4 pretrain_rna.py --data_path data.h5mu --resume saved/pretrain_rna.pth
"""

import os
import argparse
import math
from functools import reduce
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model.performer.performer_pytorch import PerformerLM
from tqdm import tqdm
from utils import *
from model.read_data import prepare_train_val_data


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    p.add_argument("--data_path", required=True, help="Path to h5mu or h5ad data")
    p.add_argument("--train_ratio", type=float, default=0.9)

    # Model
    p.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
    p.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
    p.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')

    # Training mode
    p.add_argument("--model_path", type=str, default=None,
                   help='Pretrained model path for finetune mode. If None, train from scratch.')
    p.add_argument("--resume", type=str, default=None,
                   help='Checkpoint path to resume training (loads optimizer/scheduler/epoch).')

    # Hyperparameters
    p.add_argument("--epoch", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--grad_acc", type=int, default=20)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--replace_prob", type=float, default=0.9)
    p.add_argument("--valid_every", type=int, default=1)
    p.add_argument("--patience", type=int, default=3, help='Early stopping patience.')

    # Output
    p.add_argument("--ckpt_dir", type=str, default='./saved_model/')
    p.add_argument("--model_name", type=str, default='pretrain_rna')
    p.add_argument("--seed", type=int, default=2024)

    return p


def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    return local_rank, rank, world_size


# ====================================================================
# Masking
# ====================================================================
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)
    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0), mask.size(-1)).to(device)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)
    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


def data_mask(data, mask_prob, replace_prob, pad_token_id, mask_token_id):
    mask_ignore_token_ids = {0, pad_token_id}
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)
    masked_input = data.clone().detach()
    replace_prob = prob_mask_like(data, replace_prob)
    masked_input = masked_input.masked_fill(mask * replace_prob, mask_token_id)
    labels = data.masked_fill(~mask, pad_token_id)
    return masked_input, labels


# ====================================================================
# Dataset
# ====================================================================
class SCDataset(Dataset):
    def __init__(self, data, class_limit):
        super().__init__()
        self.data = data
        self.class_limit = class_limit

    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > (self.class_limit - 2)] = self.class_limit - 2
        full_seq = torch.from_numpy(full_seq).long()
        return full_seq

    def __len__(self):
        return self.data.shape[0]


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
    PAD_TOKEN_ID = CLASS - 1
    MASK_TOKEN_ID = CLASS - 1

    # ==========================================
    # 1. Load data
    # ==========================================
    if is_master:
        print("1. Loading data...")
    if args.data_path.endswith('.h5ad'):
        import anndata as ad
        import scanpy as sc
        from sklearn.model_selection import train_test_split
        adata = ad.read_h5ad(args.data_path)
        if is_master:
            print(f"   Loaded h5ad: {adata.shape}")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata, base=2)
        indices = np.arange(adata.shape[0])
        train_idx, val_idx = train_test_split(
            indices, test_size=1 - args.train_ratio, random_state=args.seed)
        rna_train = adata[train_idx]
        rna_val = adata[val_idx]
    else:
        rna_train, rna_val = prepare_train_val_data(
            data_path=args.data_path,
            train_ratio=args.train_ratio,
            return_mod='rna',
            cell_type_col='rna:cell_type'
        )

    train_dataset = SCDataset(rna_train.X, CLASS)
    val_dataset = SCDataset(rna_val.X, CLASS)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    # ==========================================
    # 2. Build model
    # ==========================================
    model = PerformerLM(
        num_tokens=CLASS,
        dim=200,
        depth=6,
        max_seq_len=args.gene_num,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=args.pos_embed
    )

    start_epoch = 1

    if args.resume:
        # ---- Resume mode: load model + optimizer + scheduler + epoch ----
        if is_master:
            print(f"2. Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        if is_master:
            print(f"   Resuming from epoch {start_epoch}")

    elif args.model_path:
        # ---- Finetune mode: load pretrained, freeze most layers ----
        if is_master:
            print(f"2. Finetuning from: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        for param in model.parameters():
            param.requires_grad = False
        for param in model.norm.parameters():
            param.requires_grad = True
        for param in model.performer.net.layers[-2:].parameters():
            param.requires_grad = True
        if is_master:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Total: {total:,}  Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

    else:
        # ---- From scratch: random init, all trainable ----
        if is_master:
            print("2. Training from scratch (random init)")
            total = sum(p.numel() for p in model.parameters())
            print(f"   Total parameters: {total:,}")

    model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ==========================================
    # 3. Optimizer & scheduler
    # ==========================================
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=args.learning_rate,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
    )

    # Restore optimizer/scheduler state if resuming
    if args.resume and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if is_master:
            print("   Optimizer state restored")
    if args.resume and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if is_master:
            print("   Scheduler state restored")

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='mean').to(local_rank)
    softmax = nn.Softmax(dim=-1)

    dist.barrier()

    # ==========================================
    # 4. Training loop
    # ==========================================
    max_loss = float('inf')
    patient = 0

    for i in range(start_epoch, args.epoch + 1):
        train_loader.sampler.set_epoch(i)
        model.train()
        dist.barrier()
        running_loss = 0.0
        cum_acc = 0.0
        total_batches = len(train_loader)

        for index, data in enumerate(train_loader):
            index += 1
            data = data.to(device)
            masked_input, labels = data_mask(
                data, args.mask_prob, args.replace_prob,
                PAD_TOKEN_ID, MASK_TOKEN_ID)

            if index % args.grad_acc != 0:
                with model.no_sync():
                    logits = model(masked_input)
                    loss = loss_fn(logits.transpose(1, 2), labels) / args.grad_acc
                    loss.backward()
            else:
                logits = model(masked_input)
                loss = loss_fn(logits.transpose(1, 2), labels) / args.grad_acc
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            final = softmax(logits)[..., 1:-1]
            final = final.argmax(dim=-1) + 1
            pred_num = (labels != PAD_TOKEN_ID).sum(dim=-1)
            correct_num = ((labels != PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()

        epoch_loss = running_loss / total_batches
        epoch_acc = 100 * cum_acc / total_batches
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)

        if is_master:
            print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss*args.grad_acc:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
        dist.barrier()
        scheduler.step()

        # ---- Validation ----
        if i % args.valid_every == 0:
            model.eval()
            dist.barrier()
            val_running_loss = 0.0
            predictions = []
            truths = []
            with torch.no_grad():
                for val_index, data in enumerate(val_loader):
                    val_index += 1
                    data = data.to(device)
                    masked_input, labels = data_mask(
                        data, args.mask_prob, args.replace_prob,
                        PAD_TOKEN_ID, MASK_TOKEN_ID)
                    logits = model(masked_input)
                    loss = loss_fn(logits.transpose(1, 2), labels)
                    val_running_loss += loss.item()
                    final = softmax(logits)[..., 1:-1]
                    final = final.argmax(dim=-1) + 1
                    predictions.append(final)
                    truths.append(labels)

                predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
                truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
                correct_num = ((truths != PAD_TOKEN_ID) * (predictions == truths)).sum().item()
                val_num = (truths != PAD_TOKEN_ID).sum().item()
                val_loss = val_running_loss / val_index
                val_loss = get_reduced(val_loss, local_rank, 0, world_size)

            if is_master:
                val_acc = 100 * correct_num / val_num
                print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:6.4f}%  ==')

            del predictions, truths

            # ---- Early stopping & save ----
            stop_flag = torch.tensor([0], dtype=torch.int, device=device)

            if is_master:
                if val_loss < max_loss:
                    max_loss = val_loss
                    patient = 0
                    # Save checkpoint with optimizer/scheduler state for resume
                    if not os.path.exists(args.ckpt_dir):
                        os.makedirs(args.ckpt_dir)
                    save_path = os.path.join(args.ckpt_dir, f"{args.model_name}.pth")
                    torch.save({
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'losses': val_loss,
                    }, save_path)
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


if __name__ == "__main__":
    main()
