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
import scanpy as sc
import pickle as pkl
# from read_latent_feature import read_latent
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
import pandas as pd
from utils import save_best_ckpt

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument("--grad_acc", type=int, default=1, help='Number of gradient accumulation.')
parser.add_argument("--local_rank", type=int, default=0, help='Local process rank.')
parser.add_argument("--seed", type=int, default=2024, help='Random seed.')
parser.add_argument("--data_path", type=str, default='/home/lyh/project/data/5fold/', help='Path of data.')
parser.add_argument("--mod", type=str, default='one', help='Used modalities: rna, atac, or multimodal.')
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=3, help='Batch size of the model.')
parser.add_argument("--epoch", type=int, default=20, help='Epoch of the training.')
parser.add_argument("--VALIDATE_EVERY", type=int, default=1, help='VALIDATE_EVERY')
parser.add_argument("--projection_dim", type=int, default=128)
parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--pretrained_model_path", type=str, default=None)
parser.add_argument("--ckpt_dir", type=str, default='/home/lyh/project/sc/main/model/saved_model/')
parser.add_argument("--patience", type=int, default=5, help='Patience step.')
parser.add_argument("--model", type=str, default='scMomer_cell_annotation')

rank = int(os.environ["RANK"])
local_rank = 0

dist.init_process_group(backend='nccl')
is_master = dist.get_rank() == 0

local_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
device_id = local_rank % torch.cuda.device_count()
device = torch.device(device_id)

class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > (5)] = 5
        full_seq = torch.from_numpy(full_seq).to(device)
        seq_label = self.label[index]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]



class Classifier(nn.Module):
    def __init__(self,class_num):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(128,64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(64, class_num)
    def forward(self, feature):
        out1 = self.linear1(feature)
        out1 = self.dropout1(out1)
        out1 = self.relu(out1)
        out2 = self.linear2(out1)
        out2 = self.dropout2(out2)
        return out2


def data_load(dataset, fold, args, name):
    path = args.data_path
    train_data_path = path + "train_fold" + str(fold) + "_" + dataset
    test_data_path = path + "test_fold" + str(fold) + "_" + dataset
    train_data = sc.read_h5ad(train_data_path)
    test_data = sc.read_h5ad(test_data_path)
    train_label_dict, train_label = np.unique(np.array(train_data.obs['cell_type']),
                                  return_inverse=True)  # celltype Convert strings categorical to integrate categorical, and label_dict[label] can be restored
    test_label_dict, test_label = np.unique(np.array(test_data.obs['cell_type']),
                                  return_inverse=True)
    train_label = torch.from_numpy(train_label)
    test_label = torch.from_numpy(test_label)

    train_data = train_data.X
    test_data = test_data.X

    split_train_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    for index_train, index_val in split_train_val.split(train_data, train_label):
        data_train, label_train = train_data[index_train], train_label[index_train]
        data_val, label_val = train_data[index_val], train_label[index_val]
        train_dataset = SCDataset(data_train, label_train)
        val_dataset = SCDataset(data_val, label_val)

    test_dataset = SCDataset(test_data, test_label)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle= False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, shuffle= False)
    return train_loader, val_loader, test_loader, test_label_dict, train_label_dict

def model_init(args):
    encoder = Encoder(args.projection_dim, 800)
    model = scMomer(
        args,
        atac_config=None,
        rna_decoder=None,
        atac_decoder=None,
        encoder=encoder,
    )
    path = args.pretrained_model_path
    ckpt = torch.load(path, map_location=device)
    new_state_dict = {}
    for key in ckpt['model_state_dict']:
        new_state_dict[key[7:]] = ckpt['model_state_dict'][key]

    model.load_state_dict(new_state_dict, strict=False)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.to_out.parameters():
        param.requires_grad = True
    for param in model.rna_model.model.performer.net.layers[-2].parameters():
        param.requires_grad = True
    model.sub_task = Classifier(class_num=train_label_dict.shape[0])
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model

def training(args, model, optimizer, train_loader, val_loader, fold, name):
    loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)
    dist.barrier()
    trigger_times = 0
    max_acc = -1

    for i in range(1, args.epoch + 1):
        train_loader.sampler.set_epoch(i)
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        cum_acc = 0.0
        truths = []
        predictions = []
        ProgressBar = tqdm(train_loader)
        for index, data in enumerate(ProgressBar, 0):
            ProgressBar.set_description("Epoch %d" % i)
            index += 1
            rna, labels = data[0].to(device), data[1].to(device)

            if index % args.grad_acc != 0:
                with model.no_sync():
                    logits = model(None, rna, reconstruct=False, mode=args.mod)
                    loss = loss_fn(logits, labels)
                    loss.backward()
            if index % args.grad_acc == 0:
                logits = model(None, rna, reconstruct=False, mode=args.mod)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e5))
                optimizer.step()
                optimizer.zero_grad()
            truths.append(labels)
            running_loss += loss.item()
            ProgressBar.set_postfix(loss=loss.item())
            softmax = nn.Softmax(dim=-1)
            final_prob = softmax(logits)
            final = final_prob.argmax(dim=-1)
            pred_num = labels.size(0)
            correct_num = torch.eq(final, labels).sum(dim=-1)
            final[np.amax(np.array(final_prob.cpu().detach().numpy()), axis=-1) < 0] = -1
            predictions.append(final)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        predictions = distributed_concat(torch.cat(predictions, dim=0), len(train_loader.dataset), world_size)
        truths = distributed_concat(torch.cat(truths, dim=0), len(train_loader.dataset), world_size)
        no_drop = predictions != -1
        predictions = np.array((predictions[no_drop]).cpu())
        truths = np.array((truths[no_drop]).cpu())
        f1 = f1_score(truths, predictions, average='macro')
        epoch_loss = running_loss / index
        epoch_acc = cum_acc / index
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
        if is_master:
            print(f'    ==  Dataset: {name} | Fold: {fold} Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}% | F1 Score: {f1:.6f}   ==')
            # print(confusion_matrix(truths, predictions))
            # print(classification_report(truths, predictions, labels=range(len(label_dict)), zero_division=0,
            #                             target_names=label_dict.tolist(), digits=len(label_dict)))
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
                    rna_v, labels_v = data_v[0].to(device), data_v[1].to(device)
                    logits = model(None, rna_v, reconstruct=False, mode=args.mod)
                    loss = loss_fn(logits, labels_v)
                    running_loss += loss.item()
                    softmax = nn.Softmax(dim=-1)
                    final_prob = softmax(logits)
                    final = final_prob.argmax(dim=-1)
                    final[np.amax(np.array(final_prob.cpu()), axis=-1) < 0] = -1
                    predictions.append(final)
                    truths.append(labels_v)
                del data_v, labels_v, logits, final_prob, final
                predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_loader.dataset), world_size)
                truths = distributed_concat(torch.cat(truths, dim=0), len(val_loader.dataset), world_size)
                no_drop = predictions != -1
                predictions = np.array((predictions[no_drop]).cpu())
                truths = np.array((truths[no_drop]).cpu())
                cur_acc = accuracy_score(truths, predictions)
                f1 = f1_score(truths, predictions, average='macro')
                val_loss = running_loss / index
                val_loss = get_reduced(val_loss, local_rank, 0, world_size)
                if is_master:
                    print(f'    ==  Dataset: {name} | Fold: {fold} Epoch: {i} | Validation Loss: {val_loss:.6f}| Accuracy: {cur_acc:6.4f}% | F1 Score: {f1:.6f}  ==')
                    # print(confusion_matrix(truths, predictions))
                    # print(classification_report(truths, predictions, labels=range(len(label_dict)), zero_division=0,
                    #                             target_names=label_dict.tolist(), digits=len(label_dict)))
                if cur_acc > max_acc:
                    max_acc = cur_acc
                    trigger_times = 0
                    save_best_ckpt(model, val_loss, name, args.ckpt_dir, fold, args.model)
                    print("saved model.")
                else:
                    trigger_times += 1
                    if trigger_times > args.patience:
                        break
        del predictions, truths

def test(args, model, test_loader, fold, name):
    ckpt = torch.load(f'{args.ckpt_dir}{args.model}_{name}_{str(fold)}_best.pth', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)
    model.eval()
    running_loss = 0.0
    predictions = []
    truths = []
    dist.barrier()

    with torch.no_grad():
        for index, data_v in enumerate(test_loader):
            index += 1
            rna_v, labels_v = data_v[0].to(device), data_v[1].to(device)
            logits = model(None, rna_v, reconstruct=False, mode=args.mod)
            loss = loss_fn(logits, labels_v)
            running_loss += loss.item()
            softmax = nn.Softmax(dim=-1)
            final_prob = softmax(logits)
            final = final_prob.argmax(dim=-1)
            final[np.amax(np.array(final_prob.cpu()), axis=-1) < 0] = -1
            predictions.append(final)
            truths.append(labels_v)

        del data_v, labels_v, logits, final_prob, final
        predictions = distributed_concat(torch.cat(predictions, dim=0), len(test_loader.dataset), world_size)
        truths = distributed_concat(torch.cat(truths, dim=0), len(test_loader.dataset), world_size)
        no_drop = predictions != -1
        predictions = np.array((predictions[no_drop]).cpu())
        truths = np.array((truths[no_drop]).cpu())
        cur_acc = accuracy_score(truths, predictions)
        f1 = f1_score(truths, predictions, average='macro')
        val_loss = running_loss / index
        val_loss = get_reduced(val_loss, local_rank, 0, world_size)
        if is_master:
            print(f'    ==  Dataset: {name} |  Fold: {fold} Test Loss: {val_loss:.6f} | F1 Score: {f1:.6f}  | ACC: {cur_acc:.6f} ==')
            print(confusion_matrix(truths, predictions))
            print(classification_report(truths, predictions, labels=range(len(train_label_dict)), zero_division=0,
                                        target_names=train_label_dict.tolist(), digits=len(train_label_dict)))

if __name__ == '__main__':
    args = parser.parse_args()
    name = args.dataset
    for fold in range(1,6):
        seed_all(args.seed)
        train_loader, val_loader, test_loader, test_label_dict, train_label_dict = data_load(name, fold, args, name[:-5])
        model = model_init(args)
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
        training(args, model, optimizer, train_loader, val_loader, fold, name[:-5])
        test(args, model, test_loader, fold, name[:-5])


