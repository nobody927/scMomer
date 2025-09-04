# -*- coding: utf-8 -*-

import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model.performer.performer_pytorch import PerformerLM
import scanpy as sc
from model.atac_model import *
import anndata as ad
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2024, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-5, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=30, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--data_path", type=str, default=None, help='Path of data for pretraining.')
parser.add_argument("--ckpt_dir", type=str, default='/home/lyh/project/sc/main/model/saved_model/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='pretrain_atac', help='Pretrained model name.')
parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')


args = parser.parse_args()
rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
VALIDATE_EVERY = args.valid_every
MASK_PROB = args.mask_prob
model_name = args.model_name
ckpt_dir = args.ckpt_dir

dist.init_process_group(backend='nccl')
is_master = dist.get_rank() == 0
local_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
device_id = local_rank % torch.cuda.device_count()
device = torch.device(device_id)
seed_all(SEED)

class SCDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq = torch.from_numpy(full_seq)
        return full_seq.to(device)

    def __len__(self):
        return self.data.shape[0]

data = sc.read_h5ad(args.data_path)
data = data.X

data_train, data_val = train_test_split(data, test_size=0.05,random_state=SEED)
config = MAEConfig(feature_size=data.shape[-1])
train_dataset = SCDataset(data_train)
val_dataset = SCDataset(data_val)

train_sampler = DistributedSampler(train_dataset)
val_sampler = SequentialDistributedSampler(val_dataset, batch_size=BATCH_SIZE, world_size=world_size)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

model = ViTModel(config, use_mask_token=True)
model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

# optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
# learning rate scheduler
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)

def data_mask(data, mask_prob = MASK_PROB):
    B, L = data.size(0), config.num_patches

    # 1. 随机 mask
    noise = torch.rand(B, L, device=data.device)
    len_keep = int(L * (1 - mask_prob))
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    ids_mask = ids_shuffle[:, len_keep:]

    bool_masked_pos = torch.zeros(B, L, dtype=torch.bool, device=data.device)
    bool_masked_pos.scatter_(1, ids_mask, True)
    return bool_masked_pos


dist.barrier()
for i in range(1, EPOCHS+1):
    train_loader.sampler.set_epoch(i)
    model.train()
    dist.barrier()
    running_loss = 0.0
    ProgressBar = tqdm(train_loader)
    for index, data in enumerate(ProgressBar, 0):
        ProgressBar.set_description("Epoch %d" % i)
    # for index, data in enumerate(train_loader):
        index += 1
        data = data.to(device)

        bool_masked_pos = data_mask(data)
        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                loss = model(data, bool_masked_pos=bool_masked_pos) / GRADIENT_ACCUMULATION
                loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            loss = model(data, bool_masked_pos=bool_masked_pos) / GRADIENT_ACCUMULATION
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        ProgressBar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f}  ==')
    dist.barrier()
    scheduler.step()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        dist.barrier()
        running_loss = 0.0
        with torch.no_grad():
            for index, data in enumerate(val_loader):
                index += 1
                data = data.to(device)
                bool_masked_pos = data_mask(data)
                loss = model(data, bool_masked_pos=bool_masked_pos)
                running_loss += loss.item()
            del data
            val_loss = running_loss / index
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
        if is_master:
            print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f}  ==')


    if is_master:
        save_ckpt(i, model, epoch_loss, model_name, ckpt_dir)