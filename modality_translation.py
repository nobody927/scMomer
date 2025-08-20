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
import scanpy as sc
import pickle as pkl
# from read_latent_feature import read_latent
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument("--grad_acc", type=int, default=1, help='Number of gradient accumulation.')
parser.add_argument("--local_rank", type=int, default=0, help='Local process rank.')
parser.add_argument("--seed", type=int, default=2024, help='Random seed.')
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--dataset", type=str, default="Xin.h5ad")
parser.add_argument("--batch_size", type=int, default=2, help='Batch size of the model.')
parser.add_argument("--epoch", type=int, default=20, help='Epoch of the training.')
parser.add_argument("--VALIDATE_EVERY", type=int, default=1, help='VALIDATE_EVERY')
parser.add_argument("--projection_dim", type=int, default=128)
parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--rna_model_path", type=str, default='/home/lyh/project/sc/main/model/saved_model/missing_30_best.pth')
parser.add_argument("--ckpt_dir", type=str, default='/home/lyh/project/sc/main/model/saved_model/')
parser.add_argument("--model_name", type=str, default='Xin')
parser.add_argument("--patience", type=int, default=3, help='Patience step.')
parser.add_argument("--model", type=str, default='my_turn_all')
# parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')


rank = int(os.environ["RANK"])
local_rank = 0
is_master = local_rank == 0



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
        # rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > (5)] = 5
        full_seq = torch.from_numpy(full_seq).to(device)
        # full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        seq_label = self.label[index].toarray()[0]
        seq_label = torch.from_numpy(seq_label).to(device)
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
        # out = torch.cat([atac,rna],1)
        out1 = self.linear1(feature)
        out1 = self.dropout1(out1)
        out1 = self.relu(out1)
        out2 = self.linear2(out1)
        out2 = self.dropout2(out2)
        return out2

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        return x

def data_load(dataset, fold, args, name, path = "/home/lyh/project/data/5fold/new/"):

    train_data_path_rna = path + "train_rna_fold" + str(fold) + "_" + dataset
    test_data_path_rna = path + "test_rna_fold" + str(fold) + "_" + dataset
    train_data_path_atac = path + "train_atac_fold" + str(fold) + "_" + dataset
    test_data_path_atac = path + "test_atac_fold" + str(fold) + "_" + dataset

    train_data_rna = sc.read_h5ad(train_data_path_rna)
    test_data_rna = sc.read_h5ad(test_data_path_rna)
    train_data_atac = sc.read_h5ad(train_data_path_atac)
    test_data_atac = sc.read_h5ad(test_data_path_atac)

    train_label_dict, train_label = np.unique(np.array(train_data_rna.obs['cell_type']),
                                  return_inverse=True)  # celltype Convert strings categorical to integrate categorical, and label_dict[label] can be restored
    test_label_dict, test_label = np.unique(np.array(test_data_rna.obs['cell_type']),
                                  return_inverse=True)
    # store the label dict and label for prediction
    # with open('/home/lyh/project/data/label/'+args.model+ '_'+name+'_'+str(fold)+'_label_dict', 'wb') as fp:
    #     pkl.dump(test_label_dict, fp)
    # with open('/home/lyh/project/data/label/'+args.model_name+'_'+str(fold)+'_label', 'wb') as fp:
    #     pkl.dump(test_label, fp)
    train_label = torch.from_numpy(train_label)
    test_label = torch.from_numpy(test_label)

    train_data_rna = train_data_rna.X
    test_data_rna = test_data_rna.X
    train_data_atac = train_data_atac.X
    test_data_atac = test_data_atac.X


    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    for index_train, index_val in sss.split(train_data_rna, train_label):
        data_train, label_train = train_data_rna[index_train], train_data_atac[index_train]
        data_val, label_val = train_data_rna[index_val], train_data_atac[index_val]
        train_dataset = SCDataset(data_train, label_train)
        val_dataset = SCDataset(data_val, label_val)

    test_dataset = SCDataset(test_data_rna, test_data_atac)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=False, shuffle= False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle= False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, shuffle= False)

    return train_loader, val_loader, test_loader, test_label_dict, train_label_dict

def model_init(args):
    encoder = Encoder(args.projection_dim, 800)
    atac_config = MAEConfig(feature_size=train_loader.dataset.label.shape[-1])
    d_rna = Reconstruct_net(train_loader.dataset.data.shape[-1], args.projection_dim)
    d_atac = Reconstruct_net(train_loader.dataset.label.shape[-1], args.projection_dim)
    model = scPlex(
        args,
        atac_config=atac_config,
        rna_decoder=d_rna,
        atac_decoder=d_atac,
        encoder=encoder,
        # rna_config=rna_config,
    )
    path = args.rna_model_path
    ckpt = torch.load(path, map_location=device)
    # model.load_state_dict(ckpt['model_state_dict'])

    new_state_dict = {}

    for key in ckpt['model_state_dict']:
        new_state_dict[key[7:]] = ckpt['model_state_dict'][key]
    new_state_dict.pop('atac_model.embeddings.patch_embeddings.projection.weight')
    new_state_dict.pop('ATAC.output_mean.0.weight')
    new_state_dict.pop('ATAC.output_mean.0.bias')
    model.load_state_dict(new_state_dict, strict=False)

    for param in model.rna_model.model.performer.parameters():
        param.requires_grad = False
    for param in model.rna_model.model.performer.net.layers[-2].parameters():
        param.requires_grad = True
    for param in model.rna_model.model.performer.net.layers[-1].parameters():
        param.requires_grad = True


    model.sub_task = Classifier(class_num=train_label_dict.shape[0])

    model = model.to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    return model

def training(args, model, optimizer, train_loader, val_loader, fold, name):
    loss_fn1 = nn.MSELoss().to(local_rank)
    loss_fn2 = nn.MSELoss().to(local_rank)
    dist.barrier()
    trigger_times = 0
    min_loss = 100

    for i in range(1, args.epoch + 1):
        train_loader.sampler.set_epoch(i)
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        ProgressBar = tqdm(train_loader)
        # for index, data in enumerate(train_loader):
        #     rna_v, labels_v = data_v[0].to(device), data_v[1].to(device)
        for index, data in enumerate(ProgressBar, 0):
            # if index > 10:
            #     continue
            ProgressBar.set_description("Epoch %d" % i)
            index += 1
            rna, labels = data[0].to(device), data[1].to(device)

            if index % args.grad_acc != 0:
                with model.no_sync():
                    logits, RNA, _ = model(None, rna)
                    loss1 = loss_fn1(logits, labels)
                    loss2 = loss_fn2(RNA, rna)
                    loss = loss1+loss2
                    loss.backward()
            if index % args.grad_acc == 0:
                logits, RNA, _ = model(None, rna)
                loss1 = loss_fn1(logits, labels)
                loss2 = loss_fn2(RNA, rna)
                loss = loss1 + loss2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e5))
                optimizer.step()
                optimizer.zero_grad()
            # truths.append(labels)
            running_loss += loss.item()
            ProgressBar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / index
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        if is_master:
            print(f'    ==  Dataset: {name} | Fold: {fold} Epoch: {i} | Training Loss: {epoch_loss:.6f} ')
        dist.barrier()
        scheduler.step()

        if i % args.VALIDATE_EVERY == 0:
            model.eval()
            running_loss = 0.0

            dist.barrier()
            with torch.no_grad():
                for index, data_v in enumerate(val_loader):
                    index += 1
                    rna_v, labels_v = data_v[0].to(device), data_v[1].to(device)

                    logits, RNA, _ = model(None, rna_v)
                    loss1 = loss_fn1(logits, labels_v)
                    loss2 = loss_fn2(RNA, rna_v)
                    loss = loss1+loss2

                    running_loss += loss.item()


                del data_v, labels_v, logits,
                val_loss = running_loss / index
                val_loss = get_reduced(val_loss, local_rank, 0, world_size)
                if is_master:
                    print(f'    ==  Dataset: {name} | Fold: {fold} Epoch: {i} | Validation Loss: {val_loss:.6f} ==')
                    if val_loss < min_loss:
                        min_loss = val_loss
                        trigger_times = 0
                        save_best_ckpt(model, val_loss, name, args.ckpt_dir, fold, args.model)
                        print("saved model.")
                    else:
                        trigger_times += 1
                        if trigger_times > args.patience:
                            break


def test(args, model, test_loader, fold, name):

    ckpt = torch.load(f'{args.ckpt_dir}{args.model}_{name}_{str(fold)}_best.pth', map_location=device)

    model.load_state_dict(ckpt['model_state_dict'])

    loss_fn1 = nn.MSELoss().to(local_rank)
    loss_fn2 = nn.MSELoss().to(local_rank)
    model.eval()
    running_loss = 0.0
    predictions = []
    truths = []
    dist.barrier()
    embeddings_dict = {
        'cell_emb': [],
        'atac_emb': []
    }
    with torch.no_grad():
        for index, data_v in enumerate(test_loader):
            index += 1
            rna_v, labels_v = data_v[0].to(device), data_v[1].to(device)
            logits, RNA, cell = model(None, rna_v)
            loss1 = loss_fn1(logits, labels_v)
            loss2 = loss_fn2(RNA, rna_v)
            loss = loss1 + loss2

            atac_emb_np = logits.cpu().detach().numpy()
            rna_emb_np = RNA.cpu().detach().numpy()
            cell_emb_np = cell.cpu().detach().numpy()
            # 保存到字典中

            embeddings_dict['cell_emb'].append(cell_emb_np)
            embeddings_dict['atac_emb'].append(atac_emb_np)



            running_loss += loss.item()


        def gather_embeddings(embeddings):
            gathered_embeddings = [torch.zeros_like(embeddings) for _ in range(world_size)]
            dist.all_gather(gathered_embeddings, embeddings)
            return torch.cat(gathered_embeddings, dim=0)

        for key in embeddings_dict.keys():
            embeddings_dict[key] = torch.tensor(np.concatenate(embeddings_dict[key], axis=0)).to(device)
            # embeddings_dict[key] = gather_embeddings(embeddings_dict[key]).cpu().numpy()
            embeddings_dict[key] = distributed_concat(embeddings_dict[key], len(test_loader.dataset), world_size).cpu().numpy()
        del data_v, labels_v, logits

        val_loss = running_loss / index
        val_loss = get_reduced(val_loss, local_rank, 0, world_size)
        if is_master:
            print(f'    ==  Dataset: {name} |  Fold: {fold} Test Loss: {val_loss:.6f}')
            save_pa = '/home/lyh/project/data/label/' +args.model+ '_' + name + '_' + str(fold) + '_emb_turn_all_5_cell'
            np.savez(save_pa, rna_emb=embeddings_dict['cell_emb'],
                     atac_emb=embeddings_dict['atac_emb'])
            # a = 1







if __name__ == '__main__':
    # save_pa = "/home/lyh/project/data/label/my_Xin.h5ad_1_emb.npy.npz"

    # # 加载 .npz 文件
    # data = np.load(save_pa)
    #
    # # 访问保存的数组
    # cell_emb = data['cell_emb']
    args = parser.parse_args()
    # datas = ['Xin.h5ad', 'Baron_Human.h5ad', 'Muraro.h5ad', 'Segerstolpe.h5ad']
    # datas = ['Muraro.h5ad']
    name = "preprocessed_data.h5mu"

    for fold in range(5,6):
        seed_all(args.seed)
        train_loader, val_loader, test_loader, label_dict, train_label_dict = data_load(name, fold, args, name[:-5])
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
        test(args, model, test_loader, fold, name[:-5])
        #
        # training(args, model, optimizer, train_loader, val_loader, fold, name[:-5])
        # test(args, model, test_loader, fold, name[:-5])


