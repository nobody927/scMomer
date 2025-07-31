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
import pickle
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
binary = False
name = 'my'


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

def safe_pearsonr(y_pred, y_true):
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    valid_mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if valid_mask.sum() < 2:
        return 0.0  # 或者 float('nan')
    return pearsonr(y_pred[valid_mask], y_true[valid_mask])[0]


def train(train_loader,val_loader,test_loader,drug_name):
    best = -1
    for epoch in range(20):
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
            # drug_features = drug[0][0].clone().detach().requires_grad_(True).to(device)
            # drug_adjs = drug[0][1].clone().detach().requires_grad_(True).to(device)
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
                    auc_score.append(outputs.data.cpu().numpy()[k][1])
                predicted = torch.argmax(outputs, 1)
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
                # drug_features = drug[0][0].clone().detach().requires_grad_(True).to(device)
                # drug_adjs = drug[0][1].clone().detach().requires_grad_(True).to(device)
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
                        auc_score.append(outputs.data.cpu().numpy()[k][1])
                    predicted = torch.argmax(outputs, 1)
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
                # F1 = sklearn.metrics.f1_score(labels, y_pred, labels=None, pos_label=1, average='weighted')
                AUROC = roc_auc_score(auc_label, auc_score)
                # 100 * correct / total, running_loss / (i + 1), AUROC, AUPRC
                print(f"train loss: {running_loss / (i + 1):.4f}, acc: {correct / total:.4f}, AUROC: {AUROC:.4f}")
            else:
                all_outputs = torch.cat(all_outputs).cpu().numpy()
                all_labels = torch.cat(all_labels).cpu().numpy()
                corr_coefficient = safe_pearsonr(all_outputs, all_labels)
                print(f"val loss: {running_loss / (i + 1):.4f}, PCC: {corr_coefficient}")
                if corr_coefficient > best:
                    best = corr_coefficient
                    torch.save(model.state_dict(), '/home/lyh/project/data/drug_response/'+name+ '_'+drug_name+'_model.pth')
    print(best)

def test(test_loader, drug_name):
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
    model.load_state_dict(torch.load('/home/lyh/project/data/drug_response/' + name + '_' + drug_name + '_model.pth'))
    ProgressBar = tqdm(test_loader)
    ProgressBar.set_description("Test")
    with torch.no_grad():
        for i, data in enumerate(ProgressBar, 0):

            drug, gene_expression, methylation, mutation, labels = data[0], data[1], data[2], data[3], data[4].to(device)
            drug = drug.to(device)
            # drug_features = drug[0][0].clone().detach().requires_grad_(True).to(device)
            # drug_adjs = drug[0][1].clone().detach().requires_grad_(True).to(device)
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
                    auc_score.append(outputs.data.cpu().numpy()[k][1])
                predicted = torch.argmax(outputs, 1)
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
            # F1 = sklearn.metrics.f1_score(labels, y_pred, labels=None, pos_label=1, average='weighted')
            AUROC = roc_auc_score(auc_label, auc_score)
            # 100 * correct / total, running_loss / (i + 1), AUROC, AUPRC
            print(f"train loss: {running_loss / (i + 1):.4f}, acc: {correct / total:.4f}, AUROC: {AUROC:.4f}")
        else:
            all_outputs = torch.cat(all_outputs).cpu().numpy()
            all_labels = torch.cat(all_labels).cpu().numpy()

            from scipy.stats import spearmanr
            corr_coefficient = safe_pearsonr(all_outputs, all_labels)
            rho, _ = spearmanr(all_outputs, all_labels)
            print(f"Test loss: {running_loss / (i + 1):.4f}, PCC: {corr_coefficient},SCC{rho}")
            return corr_coefficient, rho, running_loss / (i + 1)


if __name__ == '__main__':

    mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = drug_response_data_gen(
        Drug_info_file, Cell_line_info_file, Drug_feature_file, Gene_expression_file, IC50_thred_file, use_thred=binary)

    feature = np.load('/home/lyh/project/data/drug_response/gene_emb.npy')
    gexpr_feature = pd.DataFrame(feature, index=gexpr_feature.index)

    results_pcc = {}
    results_scc = {}
    results_loss = {}
    unique_names = sorted(set(item[1] for item in data_idx))
    for drug in unique_names:
        file_path = '/home/lyh/project/data/drug_response/' + name + '_' + drug + '_model.pth'
        if os.path.exists(file_path):

            data_test_idx = [item for item in data_idx if item[1] == drug]
            test_dataset, cancer_type_test_list = FeatureExtract(
                data_test_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)
            model = DeepCDR(drug_input_dim=75, mutation_dim=mutation_feature.shape[-1],
                            gexpr_dim=gexpr_feature.shape[-1],
                            methy_dim=methylation_feature.shape[-1], regression=True)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            if binary:
                loss_function = nn.CrossEntropyLoss()

            else:
                loss_function = nn.MSELoss()
            model.to(device=device)
            pcc, scc, loss = test(test_loader, drug)
            results_pcc[drug] = pcc
            results_scc[drug] = scc
            results_loss[drug] = loss
        else:
            data_train_idx = [item for item in data_idx if item[1] != drug]
            data_test_idx = [item for item in data_idx if item[1] == drug]
            data_train_idx, data_val_idx = DataSplit(data_train_idx)
            train_dataset, cancer_type_train_list = FeatureExtract(
                data_train_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)
            val_dataset, cancer_type_val_list = FeatureExtract(
                data_val_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)
            test_dataset, cancer_type_test_list = FeatureExtract(
                data_test_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn,drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)

            model = DeepCDR(drug_input_dim=75, mutation_dim=mutation_feature.shape[-1], gexpr_dim=gexpr_feature.shape[-1],
                            methy_dim=methylation_feature.shape[-1], regression=True)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            if binary:
                loss_function = nn.CrossEntropyLoss()

            else:
                loss_function = nn.MSELoss()

            model.to(device=device)
            train(train_loader,val_loader,test_loader,drug)
            pcc, scc, loss = test(test_loader,drug)
            results_pcc[drug]=pcc
            results_scc[drug]=scc
            results_loss[drug]=loss
    with open('/home/lyh/project/data/drug_response/'+name+'_blink_pcc.pkl', 'wb') as file:
        pickle.dump(results_pcc, file)
    with open('/home/lyh/project/data/drug_response/'+name+'_blink_scc.pkl', 'wb') as file:
        pickle.dump(results_scc, file)
    with open('/home/lyh/project/data/drug_response/'+name+'_blink_loss.pkl', 'wb') as file:
        pickle.dump(results_loss, file)

