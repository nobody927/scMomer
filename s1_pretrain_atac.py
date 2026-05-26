# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model.atac_encoder import *
import anndata as ad
from utils import *
from model.read_data import prepare_train_val_data
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=50, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2024, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=32, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=2, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--data_path", type=str, required=True,
                    help='Path of data.')
parser.add_argument("--ckpt_dir", type=str, default='./saved_model/',
                    help='Directory of checkpoint.')
parser.add_argument("--model_name", type=str, default='pretrain_atac', help='Pretrained model name.')
parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
parser.add_argument("--train_ratio", type=float, default=0.9, help='Ratio of training data.')

args = parser.parse_args()
SEED = args.seed
# Fix: standard torchrun DDP environment variable setup
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
is_master = rank == 0

# Required: set default device before init_process_group
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')

dist.init_process_group(backend='nccl')
seed_all(SEED)


class SCDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        # Fix: use the real index assigned by DDP
        full_seq = self.data[index].toarray()[0]
        full_seq = torch.from_numpy(full_seq)
        return full_seq  # Fix: do NOT call .to(device) here

    def __len__(self):
        return self.data.shape[0]


data_train, data_val = prepare_train_val_data(
    data_path=args.data_path,
    train_ratio=args.train_ratio,
    return_mod='atac',
    cell_type_col='rna:cell_type'
)

config = MAEConfig(feature_size=data_train.X.shape[-1])
train_dataset = SCDataset(data_train.X)
val_dataset = SCDataset(data_val.X)

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(val_dataset)

# TODO: consider increasing num_workers for faster data loading and pin_memory=True for faster GPU transfer
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=2)

model = ViTModel(config, use_mask_token=True)
model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

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


def data_mask(data, mask_prob=args.mask_prob):
    B, L = data.size(0), config.num_patches
    noise = torch.rand(B, L, device=data.device)
    len_keep = int(L * (1 - mask_prob))
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    ids_mask = ids_shuffle[:, len_keep:]

    bool_masked_pos = torch.zeros(B, L, dtype=torch.bool, device=data.device)
    bool_masked_pos.scatter_(1, ids_mask, True)
    return bool_masked_pos


dist.barrier()

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
        data = data.to(device, non_blocking=True)

        bool_masked_pos = data_mask(data)
        raw_loss = model(data, bool_masked_pos=bool_masked_pos)
        loss = raw_loss / args.grad_acc

        # Fix: add 'or index == total_batches' to ensure the last incomplete batch also gets a step
        is_update_step = (index % args.grad_acc == 0) or (index == total_batches)

        if not is_update_step:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e4)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        running_loss += raw_loss.item()

    epoch_loss = running_loss / total_batches
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)

    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f}  ==')

    dist.barrier()
    scheduler.step()

    if i % args.valid_every == 0:
        model.eval()
        dist.barrier()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_index, data in enumerate(val_loader):
                val_index += 1
                data = data.to(device, non_blocking=True)
                bool_masked_pos = data_mask(data)
                loss = model(data, bool_masked_pos=bool_masked_pos)
                val_running_loss += loss.item()

            val_loss = val_running_loss / val_index
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)

        if is_master:
            print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f}  ==')

        stop_flag = torch.tensor([0], dtype=torch.int, device=device)

        if is_master:
            if val_loss < max_loss:
                max_loss = val_loss
                patient = 0
                save_ckpt(i, model, val_loss, args.model_name, args.ckpt_dir)
                print(f'    == model saved at epoch: {i} ==')
            else:
                patient += 1
                if patient > 3:  # TODO: ATAC patience is only 3, consider if too short
                    stop_flag += 1

        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)

        if stop_flag.item() > 0:
            if is_master:
                print(f' 🛑 Early stopping triggered at epoch {i}. All GPUs stopping gracefully.')
            break

        torch.cuda.empty_cache()