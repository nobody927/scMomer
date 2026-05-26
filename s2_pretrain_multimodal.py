import torch
import os
import torch.nn as nn
from model.read_data import prepare_train_val_data
import argparse
from model.scmomer import scMomer
from model.atac_encoder import MAEConfig
from torch.utils.data import Subset, DataLoader, Dataset
from torch.optim import Adam, SGD, AdamW
from tqdm import tqdm
import numpy as np
from utils import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from functools import reduce
import torch.nn as nn
parser = argparse.ArgumentParser()
parser.add_argument("--grad_acc", type=int, default=5, help='Number of gradient accumulation.')
parser.add_argument("--local_rank", type=int, default=0, help='Local process rank.')
parser.add_argument("--seed", type=int, default=2024, help='Random seed.')
parser.add_argument("--data_path", type=str, required=True,
                    help='Path of data.')
parser.add_argument("--mod", type=str, default='multimodal', help='Used modalities: rna, atac, or multimodal.')
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=3, help='Batch size of the model.')
parser.add_argument("--epoch", type=int, default=40, help='Epoch of the training.')
parser.add_argument("--VALIDATE_EVERY", type=int, default=1, help='VALIDATE_EVERY')
parser.add_argument("--projection_dim", type=int, default=128)
parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--rna_model_path", type=str, required=True)
parser.add_argument("--atac_model_path", type=str, required=True)
parser.add_argument("--model_name", type=str, default='pretrain_multimodel')
parser.add_argument("--ckpt_dir", type=str, default='./saved_model/',
                    help='Directory of checkpoint to save.')
parser.add_argument("--train_ratio", type=float, default=0.9, help='Ratio of training data.')
args = parser.parse_args()
def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    return local_rank, rank, world_size
local_rank, rank, world_size = setup_ddp()


SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
PATIENCE = 3
UNASSIGN_THRES = 0.0

model_name = args.model_name
is_master = dist.get_rank() == 0

device = torch.device(f'cuda:{local_rank}')
ckpt_dir = args.ckpt_dir
seed_all(SEED)

train_data, val_data, label_dict = prepare_train_val_data(
    data_path=args.data_path,
    train_ratio=args.train_ratio,
    return_mod='multi',
    cell_type_col='rna:cell_type'
)

train_sampler = DistributedSampler(train_data)
val_sampler = DistributedSampler(val_data)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, sampler=val_sampler)



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(64, 2)

    def forward(self, cell_emb):
        out1 = self.linear1(cell_emb)
        out1 = self.dropout1(out1)
        out1 = self.relu(out1)
        out2 = self.linear2(out1)
        return out2

def create_pairs(atac, rna, cell_type, fake_ratio=0.25):
    """
    Create paired samples:
      - Real pairs: (atac[i], rna[i]) with label=0, for all i in batch
      - Fake pairs: (atac[i], rna[j]) with label=1, where j != i and cell_type[j] != cell_type[i]
        If no valid j exists for a selected i, skip generating that fake pair (STRICT mode).

    Returns:
      combined_atac, combined_rna, combined_labels (shuffled)
    """
    batch_size = atac.size(0)
    num_fake_target = int(batch_size * fake_ratio)
    device = atac.device

    if cell_type.device != device:
        cell_type = cell_type.to(device)

    # real paired data (label=0)
    real_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    if num_fake_target > 0:
        # randomly choose indices to attempt creating fake ATAC samples
        fake_idx_atac_candidates = torch.randperm(batch_size, device=device)[:num_fake_target]
        all_indices = torch.arange(batch_size, device=device)

        kept_atac_idx = []
        kept_rna_idx = []

        for idx in fake_idx_atac_candidates:
            # exclude same index (avoid true pair)
            mask = all_indices != idx
            # STRICT: also exclude pairs that share the same cell type
            mask = mask & (cell_type != cell_type[idx])

            available_indices = all_indices[mask]

            # STRICT: if nothing left after filtering, skip this fake pair
            if available_indices.numel() == 0:
                continue

            # choose a random index from available ones (on same device)
            selected = torch.randint(0, available_indices.numel(), (1,), device=device)
            kept_atac_idx.append(idx.item())
            kept_rna_idx.append(available_indices[selected].item())

        if len(kept_atac_idx) > 0:
            fake_idx_atac = torch.tensor(kept_atac_idx, device=device, dtype=torch.long)
            fake_idx_rna = torch.tensor(kept_rna_idx, device=device, dtype=torch.long)

            # construct fake pairs (label=1)
            fake_atac = atac[fake_idx_atac]
            fake_rna = rna[fake_idx_rna]
            fake_labels = torch.ones(len(fake_idx_atac), dtype=torch.long, device=device)

            # concat all
            combined_atac = torch.cat([atac, fake_atac], dim=0)
            combined_rna = torch.cat([rna, fake_rna], dim=0)
            combined_labels = torch.cat([real_labels, fake_labels], dim=0)
        else:
            # no valid fake pairs could be created
            combined_atac = atac
            combined_rna = rna
            combined_labels = real_labels
    else:
        combined_atac = atac
        combined_rna = rna
        combined_labels = real_labels

    # shuffle order
    perm = torch.randperm(combined_atac.size(0), device=device)
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
rna_modal = torch.load(path, map_location='cpu')

path = args.atac_model_path
atac_model = torch.load(path, map_location='cpu')

new_state_dict_rna = {}
prefix_rna = 'rna_model.model.'
for key in rna_modal['model_state_dict']:
    # key[7:] strips the 'module.' prefix from DDP pretraining
    new_state_dict_rna[prefix_rna + key[7:]] = rna_modal['model_state_dict'][key]

missing_keys_r, unexpected_keys_r = model.load_state_dict(new_state_dict_rna, strict=False)

# Filter: only check if RNA-specific parameters are missing
real_missing_rna = [k for k in missing_keys_r if k.startswith('rna_model.')]

print("\n--- 🧬 RNA weight loading report ---")
print(f"⚠️ RNA module truly missing parameters: {len(real_missing_rna)}")
if len(real_missing_rna) > 0:
    print("Missing examples:", real_missing_rna[:5])

print(f"❓ RNA unmatched parameters (unexpected): {len(unexpected_keys_r)}")
if len(unexpected_keys_r) > 0:
    print("Unmatched examples (check prefix):", unexpected_keys_r[:5])

# ================= 2. Load ATAC =================
new_state_dict_atac = {}
prefix_atac = 'atac_model.'
for key in atac_model['model_state_dict']:
    new_state_dict_atac[prefix_atac + key[7:]] = atac_model['model_state_dict'][key]

missing_keys_a, unexpected_keys_a = model.load_state_dict(new_state_dict_atac, strict=False)

# Filter: only check if ATAC-specific parameters are missing
real_missing_atac = [k for k in missing_keys_a if k.startswith('atac_model.')]

print("\n--- 🧫 ATAC weight loading report ---")
print(f"⚠️ ATAC module truly missing parameters: {len(real_missing_atac)}")
if len(real_missing_atac) > 0:
    print("Missing examples:", real_missing_atac[:5])

print(f"❓ ATAC unmatched parameters (unexpected): {len(unexpected_keys_a)}")
if len(unexpected_keys_a) > 0:
    print("Unmatched examples (check prefix):", unexpected_keys_a[:5])



for param in model.rna_model.model.parameters():
    param.requires_grad = False
for param in model.rna_model.model.norm.parameters():
    param.requires_grad = True
for param in model.rna_model.model.performer.net.layers[-2:].parameters():
    param.requires_grad = True
for param in model.rna_model.model.to_out.parameters():
    param.requires_grad = True

for param in model.atac_model.parameters():
    param.requires_grad = False
for param in model.atac_model.layernorm.parameters():
    param.requires_grad = True
for param in model.atac_model.encoder.layer[-2:].parameters():
    param.requires_grad = True
for param in model.atac_projection.parameters():
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
def weighted_reconstruction_loss(pred_log, target_log, zero_weight=0.1, non_zero_weight=1.0, zero_threshold=0):
    # Use torch.abs instead of **2 to implement weighted L1
    l1_loss = torch.abs(pred_log - target_log)
    zero_mask = (target_log < zero_threshold).float()
    weights = zero_weight * zero_mask + non_zero_weight * (1 - zero_mask)
    weighted_loss = (l1_loss * weights).sum() / weights.sum()
    return weighted_loss


rna_loss_fn = nn.MSELoss()
pos_weight = torch.tensor([5.0]).to(device)
atac_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
discrim_loss_fn = nn.CrossEntropyLoss()
dist.barrier()


max_loss = float('inf')


for i in range(1, args.epoch + 1):
    dist.barrier()
    train_loader.sampler.set_epoch(i)
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    cum_acc = 0.0
    total_batches = len(train_loader)
    for index, data in enumerate(train_loader):
        index += 1
        # 1. Get raw data (atac_raw is already binarized to 0/1)
        atac_raw = data[0].float().to(device, non_blocking=True)
        rna = data[1].to(device, non_blocking=True)
        cell_type = data[2]

        # 2. Create real/fake pairs (atac input to model is still 0/1)
        atac, rna, labels = create_pairs(atac_raw, rna, cell_type, fake_ratio=0.5)

        # 3. Forward pass
        r_atac, r_rna, logits = model(atac, rna, reconstruct=True)

        # 4. Filter real samples (only real samples compute reconstruction loss)
        real_mask = (labels == 0)

        # Extract predictions for real samples
        real_r_atac = r_atac[real_mask]
        real_r_rna = r_rna[real_mask]

        # 5. Extract 0/1 targets for real samples (use atac directly, already binarized)
        real_target_atac = atac[real_mask]
        real_target_rna = rna[real_mask]

        # 6. Compute reconstruction loss
        atac_loss = atac_loss_fn(real_r_atac, real_target_atac)
        rna_loss = rna_loss_fn(real_r_rna, real_target_rna)
        discrim_loss = discrim_loss_fn(logits, labels)
        raw_loss = rna_loss + atac_loss + discrim_loss
        loss = raw_loss / GRADIENT_ACCUMULATION

        # Fix: ensure the last incomplete batch also gets a step update
        is_update_step = (index % GRADIENT_ACCUMULATION == 0) or (index == total_batches)

        if not is_update_step:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e5)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        running_loss += raw_loss.item()

    epoch_loss = running_loss / index
    epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)

    if is_master:
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f}  ==')

    scheduler.step()

    # ================= Validation =================
    if i % args.VALIDATE_EVERY == 0:
        model.eval()
        running_loss = 0.0
        dist.barrier()

        with torch.no_grad():

            for index, data_v in enumerate(val_loader):
                index += 1

                # 1. Get data (already binarized to 0/1)
                atac_v = data_v[0].float().to(device, non_blocking=True)
                rna_v = data_v[1].to(device, non_blocking=True)

                # 2. Forward pass
                r_atac_v, r_rna_v, _ = model(atac_v, rna_v, reconstruct=True)

                # 3. Compute reconstruction loss (use atac_v directly as target)
                atac_loss = atac_loss_fn(r_atac_v, atac_v)
                rna_loss = rna_loss_fn(r_rna_v, rna_v)
                # 4. Aggregate validation loss
                raw_loss = rna_loss + atac_loss
                running_loss += raw_loss.item()

            val_loss = running_loss / index
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
            stop_flag = torch.tensor([0], dtype=torch.int, device=device)

        if is_master:

            print(f'    ==  Epoch: {i} | validate Loss: {val_loss:.6f}  ==')
            if val_loss < max_loss:
                max_loss = val_loss
                patient = 0
                save_ckpt(i, model, val_loss, model_name, ckpt_dir)
                print(f'    == model saved at epoch: {i} ==')
            else:
                patient += 1
                if patient > 3:
                    stop_flag += 1

        # Master broadcasts the stop flag to all GPUs
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)

        # All GPUs check the flag; if > 0, all break together
        if stop_flag.item() > 0:
            if is_master:
                print(f' 🛑 Early stopping triggered at epoch {i}. All GPUs stopping gracefully.')
            break


