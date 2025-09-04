from read_data import *
from DeepCDR import DeepCDR
import torch
from torch.utils.data import Subset, DataLoader, Dataset
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from torch_geometric.data import Data, Batch
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default='/home/lyh/project/data/drug_response/', help='Path of model save.')
parser.add_argument("--name", type=str, default='scMomer', help='Model name.')
parser.add_argument("--binary", type=bool, default=False, help='Binary classification or regression task.')
parser.add_argument("--emb_path", type=str, default='/home/lyh/project/data/drug_response/gene_emb.npy', help='Path of gene embedding.')


args = parser.parse_args()
name = args.name
binary = args.binary
save_path = args.save_path


def custom_collate_fn(batch):
    """
    高效的 collate_fn，适用于 PyTorch Geometric 和多模态数据。
    batch: 列表，每项是 (drug_graph, gene_expr, meth, mut, label)
    """
    drug_graphs, gene_exps, meths, muts, labels = zip(*batch)

    # 利用 torch.stack 合并 batch（假设数据已经是 tensor）
    gene_exps = torch.stack(gene_exps, dim=0)
    meths = torch.stack(meths, dim=0)
    muts = torch.stack(muts, dim=0)
    labels = torch.stack(labels, dim=0)

    # 批处理图数据
    drug_batch = Batch.from_data_list(drug_graphs)

    return drug_batch, gene_exps, meths, muts, labels

mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = drug_response_data_gen(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,IC50_thred_file, use_thred=binary)

feature = np.load(args.emb_path)
gexpr_feature = pd.DataFrame(feature,index=gexpr_feature.index)

data_train_idx,data_test_idx = DataSplit(data_idx)
data_train_idx,data_val_idx = DataSplit(data_train_idx, 0.9)

train_dataset, cancer_type_train_list = FeatureExtract(
    data_train_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)
val_dataset, cancer_type_val_list = FeatureExtract(
    data_val_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)
test_dataset, cancer_type_test_list = FeatureExtract(
    data_test_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,collate_fn=custom_collate_fn)


model = DeepCDR(drug_input_dim=75, mutation_dim=mutation_feature.shape[-1], gexpr_dim=gexpr_feature.shape[-1], methy_dim=methylation_feature.shape[-1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
if binary:
    loss_function = nn.BCELoss()

else:
    loss_function = nn.MSELoss()

model.to(device=device)

def safe_pearsonr(y_pred, y_true):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if valid_mask.sum() < 2:
        return 0.0  # 或者 float('nan')
    return pearsonr(y_pred[valid_mask], y_true[valid_mask])[0]

def train(train_loader,val_loader):
    best = -1
    for epoch in range(30):
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        if binary:
            auc_label = []
            auc_score = []
        else:
            all_outputs = []
            all_labels = []
        ProgressBar = tqdm(train_loader)
        ProgressBar.set_description("Epoch %d" % epoch)
        for i, data in enumerate(ProgressBar, 0):
            drug, gene_expression, methylation, mutation, labels = data[0], data[1], data[2], data[3], data[4].to(device)
            drug = drug.to(device)
            gene_expression = gene_expression.to(device)
            methylation = methylation.to(device)
            mutation = mutation.to(device)
            # batch_data = convert_to_pyg_batch(drug_features, drug_adjs)
            # compute output
            outputs = model(drug, gene_expression, methylation, mutation)
            if binary:
                loss = loss_function(outputs.to(device), labels.to(device))
                # measure accuracy and record loss
                for k in range(len(labels)):
                    auc_label.append(labels.cpu().numpy()[k])
                    auc_score.append(outputs.data.cpu().numpy()[k])
                predicted = (outputs > 0.5).long()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            else:
                loss = loss_function(outputs.to(device), labels.to(device))
                all_outputs.append(outputs.view(-1))
                all_labels.append(labels.view(-1))

            ProgressBar.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
        if binary:
            precision, recall, thresholds = precision_recall_curve(auc_label, auc_score)
            AUPRC = auc(recall, precision)
            # F1 = sklearn.metrics.f1_score(labels, y_pred, labels=None, pos_label=1, average='weighted')
            AUROC = roc_auc_score(auc_label, auc_score)
            # 100 * correct / total, running_loss / (i + 1), AUROC, AUPRC
            print(f"train loss: {running_loss / (i + 1):.4f}, acc: {correct / total:.4f}, AUROC: {AUROC:.4f}")
        else:
            all_outputs = torch.cat(all_outputs).detach().cpu().numpy()
            all_labels = torch.cat(all_labels).detach().cpu().numpy()
            corr_coefficient = safe_pearsonr(all_outputs, all_labels)
            print(f"train loss: {running_loss / (i + 1):.4f}, PCC: {corr_coefficient}")
        model.eval()
        running_loss = 0
        correct = 0
        total = 0
        if binary:
            auc_label = []
            auc_score = []
        else:
            all_outputs = []
            all_labels = []
        ProgressBar = tqdm(val_loader)
        ProgressBar.set_description("Epoch %d Validation" % epoch)
        with torch.no_grad():
            for i, data in enumerate(ProgressBar, 0):
                drug, gene_expression, methylation, mutation, labels = data[0], data[1], data[2], data[3], data[4].to(
                    device)
                drug = drug.to(device)

                gene_expression = gene_expression.to(device)
                methylation = methylation.to(device)
                mutation = mutation.to(device)
                # compute output
                outputs = model(drug, gene_expression, methylation, mutation)
                if binary:
                    loss = loss_function(outputs.to(device), labels.to(device))
                    # measure accuracy and record loss
                    for k in range(len(labels)):
                        auc_label.append(labels.cpu().numpy()[k])
                        auc_score.append(outputs.data.cpu().numpy()[k])
                    predicted = (outputs > 0.5).long()
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                else:
                    loss = loss_function(outputs.to(device), labels.to(device))
                    all_outputs.append(outputs.view(-1))
                    all_labels.append(labels.view(-1))

                ProgressBar.set_postfix(loss=loss.item())
                running_loss += loss.item()
            if binary:
                precision, recall, thresholds = precision_recall_curve(auc_label, auc_score)
                AUPRC = auc(recall, precision)
                AUROC = roc_auc_score(auc_label, auc_score)
                # 100 * correct / total, running_loss / (i + 1), AUROC, AUPRC
                print(f"train loss: {running_loss / (i + 1):.4f}, acc: {correct / total:.4f}, AUROC: {AUROC:.4f}")
                if AUROC > best:
                    best = AUROC
                    torch.save(model.state_dict(), save_path+name+'_drug_response.pth')

            else:
                all_outputs = torch.cat(all_outputs).cpu().numpy()
                all_labels = torch.cat(all_labels).cpu().numpy()
                corr_coefficient = safe_pearsonr(all_outputs, all_labels)
                print(f"val loss: {running_loss / (i + 1):.4f}, PCC: {corr_coefficient}")
                if corr_coefficient > best:
                    best = corr_coefficient
                    torch.save(model.state_dict(), save_path+name+'_drug_response.pth')

    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    if binary:
        auc_label = []
        auc_score = []
        all_outputs = []
        all_labels = []
    else:
        all_outputs = []
        all_labels = []
    model.load_state_dict(torch.load(save_path+name+'_drug_response.pth'))
    ProgressBar = tqdm(test_loader)
    ProgressBar.set_description("Test")
    with torch.no_grad():
        for i, data in enumerate(ProgressBar, 0):
            drug, gene_expression, methylation, mutation, labels = data[0], data[1], data[2], data[3], data[4].to(
                device)
            drug = drug.to(device)
            gene_expression = gene_expression.to(device)
            methylation = methylation.to(device)
            mutation = mutation.to(device)
            outputs = model(drug, gene_expression, methylation, mutation)
            if binary:
                loss = loss_function(outputs.to(device), labels.to(device))
                # measure accuracy and record loss
                for k in range(len(labels)):
                    auc_label.append(labels.cpu().numpy()[k])
                    auc_score.append(outputs.data.cpu().numpy()[k])
                predicted = (outputs > 0.5).long()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_outputs.append(outputs.view(-1))
                all_labels.append(labels.view(-1))
            else:
                loss = loss_function(outputs.to(device), labels.to(device))
                all_outputs.append(outputs.view(-1))
                all_labels.append(labels.view(-1))

            ProgressBar.set_postfix(loss=loss.item())
            running_loss += loss.item()
        if binary:
            all_outputs = torch.cat(all_outputs).cpu().numpy()
            all_labels = torch.cat(all_labels).cpu().numpy()
            np.save('/home/lyh/project/data/drug_response/'+name+'_random_split_all_outputs.npy', all_outputs)
            np.save('/home/lyh/project/data/drug_response/drug_all_labels_binary.npy', all_labels)
            precision, recall, thresholds = precision_recall_curve(auc_label, auc_score)
            AUPRC = auc(recall, precision)
            # F1 = sklearn.metrics.f1_score(labels, y_pred, labels=None, pos_label=1, average='weighted')
            AUROC = roc_auc_score(auc_label, auc_score)
            # 100 * correct / total, running_loss / (i + 1), AUROC, AUPRC
            print(f"train loss: {running_loss / (i + 1):.4f}, acc: {correct / total:.4f}, AUROC: {AUROC:.4f}")
        else:
            all_outputs = torch.cat(all_outputs).cpu().numpy()
            all_labels = torch.cat(all_labels).cpu().numpy()
            corr_coefficient = safe_pearsonr(all_outputs, all_labels)
            from scipy.stats import spearmanr
            import matplotlib.pyplot as plt
            import seaborn as sns
            np.save('/home/lyh/project/data/drug_response/'+name+'_random_split_all_outputs.npy', all_outputs)
            np.save('/home/lyh/project/data/drug_response/drug_all_labels_binary.npy', all_labels)
            # y_pred, y_true 都是 numpy.ndarray
            rho, _ = spearmanr(all_outputs, all_labels)
            print(f"test loss: {running_loss / (i + 1):.4f}, PCC: {corr_coefficient},SCC{rho}")

            sns.set(style="white", font_scale=1.2)
            g = sns.jointplot(x=all_labels, y=all_outputs, kind='scatter', color='tomato', edgecolor=None, s=10, alpha=0.6)

            # 回归线
            sns.regplot(x=all_labels, y=all_outputs, scatter=False, ax=g.ax_joint, line_kws={"color": "steelblue", "linewidth": 2})
            g.ax_joint.tick_params(axis='both', which='major', length=6, width=1, direction='in', top=True, right=True)
            g.ax_joint.tick_params(axis='both', which='minor', length=3, width=1, direction='in', top=True, right=True)

            # 文本标注
            n = len(all_labels)
            g.ax_joint.text(
                0.05, 0.95,
                f"$corr_coefficient$ = {corr_coefficient:.3f}\n$n$ = {n}",
                transform=g.ax_joint.transAxes,
                fontsize=12,
                verticalalignment='top'
            )

            # 标签
            g.set_axis_labels("Observed", "Pred")

            plt.tight_layout()
            plt.show()


train(train_loader,val_loader)