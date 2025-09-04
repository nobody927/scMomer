import torch
import torch.nn as nn
import argparse
from model.multifoundation import scMomer
from torch.utils.data import Subset, DataLoader, Dataset
import numpy as np
from utils import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import scanpy as sc
import pickle as pkl



os.environ['CUDA_VISIBLE_DEVICES'] = '1'


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0, help='Local process rank.')
parser.add_argument("--seed", type=int, default=2024, help='Random seed.')
parser.add_argument("--data_path", type=str, default='/home/lyh/project/data/Zheng68K.h5ad', help='Path of data.')
parser.add_argument("--batch_size", type=int, default=3, help='Batch size of the model.')
parser.add_argument("--projection_dim", type=int, default=128)
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--ckpt_dir", type=str, default='/home/lyh/project/sc/main/model/saved_model/')
parser.add_argument("--model_name", type=str, default='scMomer_cell_annotation', help='Name of the model.')


class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):

        full_seq = self.data[index].toarray()[0]
        full_seq[full_seq > (args.bin_num)] = args.bin_num
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


def data_load(dataset, args, name):
    path = args.data_path
    test_data_path = path
    test_data = sc.read_h5ad(test_data_path)

    test_label_dict, test_label = np.unique(np.array(test_data.obs['cell_type']),
                                  return_inverse=True)
    # store the label dict and label for prediction
    with open('/home/lyh/project/data/label/'+args.model+ '_'+name+'_'+'_label_dict', 'wb') as fp:
        pkl.dump(test_label_dict, fp)
    with open('/home/lyh/project/data/label/'+args.model+ '_'+name+'_'+'_label', 'wb') as fp:
        pkl.dump(test_label, fp)
    test_label = torch.from_numpy(test_label)
    test_data = test_data.X
    test_dataset = SCDataset(test_data, test_label)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, shuffle= False)
    return test_loader, test_label_dict

def model_init(args):
    encoder = Encoder(args.projection_dim, 800)
    model = scMomer(
        args,
        atac_config=None,
        rna_decoder=None,
        atac_decoder=None,
        encoder=encoder,
    )
    model.sub_task = Classifier(class_num=test_label_dict.shape[0])
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model


def test(args, model, test_loader, name):
    ckpt = torch.load(f'{args.ckpt_dir}{args.model}_{name}_best.pth', map_location=device)
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
            print(f'    ==  Dataset: {name} | Test Loss: {val_loss:.6f} | F1 Score: {f1:.6f}  | ACC: {cur_acc:.6f} ==')
            print(confusion_matrix(truths, predictions))
            print(classification_report(truths, predictions, labels=range(len(test_label_dict)), zero_division=0,
                                        target_names=test_label_dict.tolist(), digits=len(test_label_dict)))


if __name__ == '__main__':
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = 0
    dist.init_process_group(backend='nccl')
    is_master = dist.get_rank() == 0
    local_rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
    device_id = local_rank % torch.cuda.device_count()
    device = torch.device(device_id)
    CLASS = args.bin_num + 2
    name = args.dataset
    seed_all(args.seed)
    test_loader, test_label_dict = data_load(name, args, name[:-5])
    model = model_init(args)
    test(args, model, test_loader, name[:-5])



