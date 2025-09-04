import torch
import torch.nn as nn
from model.read_data import read_data
import argparse
from model.multifoundation import scMomer
from model.atac_model import MAEConfig
from torch.utils.data import Subset, DataLoader, Dataset
from torch.optim import Adam, SGD, AdamW
from tqdm import tqdm
import numpy as np
from utils import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from functools import reduce
import torch.nn.functional as F

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()

parser.add_argument("--local_rank", type=int, default=0, help='Local process rank.')
parser.add_argument("--seed", type=int, default=2024, help='Random seed.')
parser.add_argument("--data_path", type=str, default=None, help='Path of data.')
parser.add_argument("--mod", type=str, default='multimodal', help='Used modalities: one or multimodal.')
parser.add_argument("--batch_size", type=int, default=3, help='Batch size of the model.')
parser.add_argument("--projection_dim", type=int, default=128)
parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--model_path", type=str, default=None, help='Path of saved model.')
parser.add_argument("--ckpt_dir", type=str, default='/home/lyh/project/sc/main/model/saved_model/', help='Directory of ATAC embedding to save.')

args = parser.parse_args()


args = parser.parse_args()
rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0
SEED = args.seed
BATCH_SIZE = args.batch_size
dist.init_process_group(backend='nccl')
is_master = dist.get_rank() == 0

local_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
device_id = local_rank % torch.cuda.device_count()
device = torch.device(device_id)
ckpt_dir = args.ckpt_dir
seed_all(SEED)

train_data, val_data, label_dict = read_data(args.mod, args.data_path)
train_sampler = DistributedSampler(train_data)
val_sampler = DistributedSampler(val_data)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, sampler=val_sampler)

print("========ending load data========")
atac_config = MAEConfig(feature_size=train_loader.dataset.atac.shape[-1])
model = scMomer(
    args,
    atac_config=atac_config,
    rna_decoder=None,
    atac_decoder=None,
    sub_task=None,
)

path = args.model_path
model_dict = torch.load(path)
new_dict = {}
for key in model_dict['model_state_dict']:
    new_dict[key[7:]] = model_dict['model_state_dict'][key]
model.load_state_dict(new_dict, strict=False)
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
dist.barrier()
model.eval()
with torch.no_grad():
    train_loader.sampler.set_epoch(1)
    all_train_atac_emb = []
    ProgressBar = tqdm(train_loader, desc=f"Getting training embedding")
    for index, data in enumerate(ProgressBar, 0):
        index += 1
        atac, rna, _ = data[0].to(device), data[1].to(device), data[2].to(device)
        emb = model(atac, rna, get_distill=True)
        all_train_atac_emb.append(emb)
    train_atac_emb = distributed_concat(torch.cat(all_train_atac_emb, dim=0), len(train_loader.dataset), world_size)
    if is_master:
        np.save(ckpt_dir+'train_atac_emb.npy', train_atac_emb.cpu().numpy())
    model.eval()
    running_loss = 0.0
    predictions = []
    truths = []
    dist.barrier()
    all_val_atac_emb = []
    for index, data_v in enumerate(val_loader):
        index += 1
        atac_v, rna_v, _ = data_v[0].to(device), data_v[1].to(device), data_v[2].to(device)
        emb = model(atac_v, rna_v, get_distill=True)
        all_val_atac_emb.append(emb)
    val_atac_emb = distributed_concat(torch.cat(all_val_atac_emb, dim=0), len(train_loader.dataset), world_size)
    if is_master:
        np.save(ckpt_dir+'val_atac_emb.npy', val_atac_emb.cpu().numpy())

