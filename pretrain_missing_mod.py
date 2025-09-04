import torch
import torch.nn as nn
from model.read_data import read_data
import argparse
from model.multifoundation import  scMomer
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
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--grad_acc", type=int, default=1, help='Number of gradient accumulation.')
parser.add_argument("--local_rank", type=int, default=0, help='Local process rank.')
parser.add_argument("--seed", type=int, default=2024, help='Random seed.')
parser.add_argument("--data_path", type=str, default=None, help='Path of data.')
parser.add_argument("--mod", type=str, default='multimodal', help='Used modalities: rna, atac, or multimodal.')
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=3, help='Batch size of the model.')
parser.add_argument("--epoch", type=int, default=101, help='Epoch of the training.')
parser.add_argument("--VALIDATE_EVERY", type=int, default=1, help='VALIDATE_EVERY')
parser.add_argument("--projection_dim", type=int, default=128)
parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--model_path", type=str, default=None, help='Path of saved model.')
parser.add_argument("--ckpt_dir", type=str, default='/home/lyh/project/sc/main/model/saved_model/', help='Directory of checkpoint to save.')
parser.add_argument("--dir_train", type=str, default='/home/lyh/project/sc/main/model/saved_model/train_atac_emb.npy')
parser.add_argument("--dir_val", type=str, default='/home/lyh/project/sc/main/model/saved_model/val_atac_emb.npy')
parser.add_argument("--model_name", type=str, default='pretrain_missing_mod')


args = parser.parse_args()
rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.lr

PATIENCE = 10
UNASSIGN_THRES = 0.0


model_name = args.model_name
dist.init_process_group(backend='nccl')
local_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
device_id = local_rank % torch.cuda.device_count()
device = torch.device(device_id)
ckpt_dir = args.ckpt_dir
seed_all(SEED)
dir_train = args.dir_train
dir_val = args.dir_val
train_latent = np.load(dir_train)
val_latent = np.load(dir_val)
train_data, val_data, label_dict = read_data(args.mod, args.data_path, latent_train=train_latent, latent_val=val_latent)

train_sampler = DistributedSampler(train_data)
val_sampler = DistributedSampler(val_data)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, sampler=val_sampler)


# atac_config = MAEConfig(feature_size=train_loader.dataset.atac.shape[-1])

d_rna = Reconstruct_net(train_loader.dataset.rna.shape[-1],args.projection_dim)
d_atac = Reconstruct_net(train_loader.dataset.atac.shape[-1],args.projection_dim)
encoder = Encoder(args.projection_dim, 800)
model = scMomer(
    args,
    atac_config=None,
    rna_decoder=d_rna,
    atac_decoder=d_atac,
    encoder = encoder,
)


path = args.model_path
ckpt = torch.load(path, map_location=device)

new_state_dict = {}
for key in ckpt['model_state_dict']:
    new_state_dict[key[7:]] = ckpt['model_state_dict'][key]

model.load_state_dict(new_state_dict, strict=False)


for param in model.parameters():
    param.requires_grad = False

for param in model.encoder.parameters():
    param.requires_grad = True

for param in model.ATAC.parameters():
    param.requires_grad = True

for param in model.RNA.parameters():
    param.requires_grad = True

for param in model.to_out.parameters():
    param.requires_grad = True


model = model.to(device)

model = DDP(model, device_ids=[local_rank], output_device=local_rank)
optimizer = Adam(model.parameters(), lr=args.lr)

scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)

loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)
rna_loss_fn = nn.MSELoss().to(local_rank)
atac_loss_fn = nn.MSELoss().to(local_rank)
latent_loss_fn = nn.MSELoss().to(local_rank)
dist.barrier()
trigger_times = 0

for i in range(1, args.epoch+1):
    train_loader.sampler.set_epoch(i)
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    cum_acc = 0.0
    ProgressBar = tqdm(train_loader)
    for index, data in enumerate(ProgressBar, 0):
        ProgressBar.set_description("Epoch %d" % i)
        index += 1
        atac, rna, labels, latent_real = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
        if index % GRADIENT_ACCUMULATION != 0:
            with model.no_sync():
                r_atac, r_rna, latent = model(rna_values = rna, mode='one')
                with torch.no_grad():
                    rna_int = rna.round()
                rna_loss = rna_loss_fn(r_rna, rna_int)
                atac_loss = atac_loss_fn(r_atac, atac)
                latent_loss = latent_loss_fn(latent, latent_real)
                loss = rna_loss+atac_loss + latent_loss
                loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            r_atac, r_rna, latent = model(rna_values = rna, mode='one')
            with torch.no_grad():
                rna_int = rna.round()
            rna_loss = rna_loss_fn(r_rna, rna_int)
            atac_loss = atac_loss_fn(r_atac, atac)
            latent_loss = latent_loss_fn(latent, latent_real)
            loss = rna_loss + atac_loss + latent_loss
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
                atac_v, rna_v, labels_v, latent_real = data_v[0].to(device), data_v[1].to(device), data_v[2].to(device), data_v[3].to(device)
                r_atac, r_rna, latent = model(rna_values = rna, mode='one')
                rna_loss = rna_loss_fn(r_rna, torch.round(rna_v))
                atac_loss = atac_loss_fn(r_atac, atac_v)
                latent_loss = latent_loss_fn(latent, latent_real)
                loss = rna_loss + atac_loss + latent_loss
                running_loss += loss.item()
            val_loss = running_loss / index
            if is_master:
                print(f'   Validate Loss: {val_loss:.6f}  ==')
                save_ckpt(i, model, val_loss, model_name, ckpt_dir)
    del predictions, truths

