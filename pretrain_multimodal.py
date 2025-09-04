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
parser.add_argument("--grad_acc", type=int, default=30, help='Number of gradient accumulation.')
parser.add_argument("--local_rank", type=int, default=0, help='Local process rank.')
parser.add_argument("--seed", type=int, default=2024, help='Random seed.')
parser.add_argument("--data_path", type=str, default=None, help='Path of data.')
parser.add_argument("--mod", type=str, default='multimodal', help='Used modalities: rna, atac, or multimodal.')
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=3, help='Batch size of the model.')
parser.add_argument("--epoch", type=int, default=100, help='Epoch of the training.')
parser.add_argument("--VALIDATE_EVERY", type=int, default=1, help='VALIDATE_EVERY')
parser.add_argument("--projection_dim", type=int, default=128)
parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--rna_model_path", type=str, default=None)
parser.add_argument("--atac_model_path", type=str, default=None)
parser.add_argument("--model_name", type=str, default='pretrain_multimodel')
parser.add_argument("--ckpt_dir", type=str, default='/home/lyh/project/sc/main/model/saved_model/', help='Directory of checkpoint to save.')

args = parser.parse_args()


args = parser.parse_args()
rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
PATIENCE = 10
UNASSIGN_THRES = 0.0

model_name = args.model_name
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
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(128,64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(64, 2)
    def forward(self, cell_emb):
        out1 = self.linear1(cell_emb)
        out1 = self.dropout1(out1)
        out1 = self.relu(out1)
        out2 = self.linear2(out1)
        return out2


def create_pairs(atac, rna, fake_ratio=0.25):
    batch_size = atac.size(0)
    num_fake_pairs = int(batch_size * fake_ratio)
    device = atac.device

    # 真实配对数据
    real_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    if num_fake_pairs > 0:
        # 为atac样本选择fake索引
        fake_idx_atac = torch.randperm(batch_size, device=device)[:num_fake_pairs]

        # 为rna样本选择与atac不同的fake索引
        # 更安全的方法：生成所有可能索引的掩码
        all_indices = torch.arange(batch_size, device=device)

        # 为每个fake_idx_atac生成可用的rna索引（排除自身）
        fake_idx_rna = torch.zeros(num_fake_pairs, dtype=torch.long, device=device)

        for i, idx in enumerate(fake_idx_atac):
            # 创建可用索引的掩码（排除当前atac索引）
            mask = all_indices != idx
            # 从可用索引中随机选择一个
            available_indices = all_indices[mask]
            selected = torch.randint(0, len(available_indices), (1,))
            fake_idx_rna[i] = available_indices[selected]

        # 创建不配对数据
        fake_atac = atac[fake_idx_atac]
        fake_rna = rna[fake_idx_rna]
        fake_labels = torch.ones(num_fake_pairs, dtype=torch.long, device=device)

        # 合并数据
        combined_atac = torch.cat([atac, fake_atac])
        combined_rna = torch.cat([rna, fake_rna])
        combined_labels = torch.cat([real_labels, fake_labels])
    else:
        # 如果没有fake pairs
        combined_atac = atac
        combined_rna = rna
        combined_labels = real_labels

    # 打乱顺序
    perm = torch.randperm(len(combined_atac), device=device)
    return combined_atac[perm], combined_rna[perm], combined_labels[perm]

atac_config = MAEConfig(feature_size=train_loader.dataset.atac.shape[-1])

d_rna = Reconstruct_net(train_loader.dataset.rna.shape[-1],args.projection_dim)
d_atac = Reconstruct_net(train_loader.dataset.atac.shape[-1],args.projection_dim)
discrim = Discriminator()
model = scMomer(
    args,
    atac_config=atac_config,
    rna_decoder=d_rna,
    atac_decoder=d_atac,
    sub_task=discrim,
)

path = args.rna_model_path
rna_modal = torch.load(path)

path = args.atac_model_path
atac_model = torch.load(path)
# model.load_state_dict(ckpt['model_state_dict'])

new_state_dict_rna = {}
prefix = 'rna_model.model.'



for key in rna_modal['model_state_dict']:
    new_state_dict_rna[prefix + key[7:]] = rna_modal['model_state_dict'][key]

model.load_state_dict(new_state_dict_rna, strict=False)

new_state_dict_atac = {}
prefix = 'atac_model.'

for key in atac_model['model_state_dict']:
    new_state_dict_atac[prefix + key[7:]] = atac_model['model_state_dict'][key]

model.load_state_dict(new_state_dict_atac, strict=False)

# from itertools import chain
# all_loaded_keys = set(new_state_dict_rna.keys()) | set(new_state_dict_atac.keys())
# model_keys = set(model.state_dict().keys())
#
# still_missing = model_keys - all_loaded_keys
# print("最终仍未被加载的参数：", still_missing)
for param in model.rna_model.model.parameters():
    param.requires_grad = False
for param in model.rna_model.model.norm.parameters():
    param.requires_grad = True
for param in model.rna_model.model.performer.net.layers[-2].parameters():
    param.requires_grad = True
for param in model.rna_model.model.to_out.parameters():
    param.requires_grad = True

model = model.to(device)



model = DDP(model, device_ids=[local_rank], output_device=local_rank)

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

rna_loss_fn = nn.MSELoss().to(local_rank)
atac_loss_fn = nn.MSELoss().to(local_rank)
discrim_loss_fn = nn.CrossEntropyLoss().to(local_rank)
dist.barrier()
trigger_times = 0
flag_loss = 10
max_acc = -1
for i in range(1, args.epoch+1):
    train_loader.sampler.set_epoch(i)
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    cum_acc = 0.0
    ProgressBar = tqdm(train_loader, desc=f"Epoch {i}")
    for index, data in enumerate(ProgressBar, 0):
        # if index > 5:
        #     continue
        # ProgressBar.set_description("Epoch %d" % i)
        index += 1
        atac, rna, _ = data[0].to(device), data[1].to(device), data[2].to(device)
        atac, rna, labels = create_pairs(atac, rna, fake_ratio=0.5)
        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                r_atac, r_rna, logits = model(atac,rna)
                with torch.no_grad():
                    rna_int = rna.round()
                rna_loss = rna_loss_fn(r_rna, rna_int)
                atac_loss = atac_loss_fn(r_atac, atac)
                discrim_loss = discrim_loss_fn(logits, labels)
                loss = rna_loss + atac_loss + discrim_loss
                loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            r_atac, r_rna, logits = model(atac, rna)
            with torch.no_grad():
                rna_int = rna.round()
            rna_loss = rna_loss_fn(r_rna, rna_int)
            atac_loss = atac_loss_fn(r_atac, atac)
            discrim_loss = discrim_loss_fn(logits, labels)
            loss = rna_loss + atac_loss + discrim_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e4))

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

    if i % args.VALIDATE_EVERY == 0:
        model.eval()
        running_loss = 0.0
        predictions = []
        truths = []
        dist.barrier()
        with torch.no_grad():
            for index, data_v in enumerate(val_loader):
                index += 1
                atac_v, rna_v, _ = data_v[0].to(device), data_v[1].to(device), data_v[2].to(device)
                atac_v, rna_v, labels = create_pairs(atac_v, rna_v, fake_ratio=0.5)
                r_atac, r_rna, logits = model(atac_v, rna_v)
                rna_loss = rna_loss_fn(r_rna, torch.round(rna_v))
                atac_loss = atac_loss_fn(r_atac, atac_v)
                discrim_loss = discrim_loss_fn(logits, labels)
                loss = rna_loss + atac_loss + discrim_loss
                running_loss += loss.item()
            val_loss = running_loss / index
            if is_master:
                print(f'   Validate Loss: {val_loss:.6f}  ==')
                save_ckpt(i, model, epoch_loss, model_name, ckpt_dir)
            # if flag_loss > val_loss:
            #     flag_loss = val_loss
            #     torch.save(
            #         {
            #             'model_state_dict': model.state_dict(),
            #         },
            #         ckpt_path
            #     )
    del predictions, truths

