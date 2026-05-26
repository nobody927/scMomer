"""
Drug response prediction using scMomer cell embeddings + GCN drug encoder.

Two modes:
  1) Pre-computed embeddings (fast):  --emb_path cell_emb.npz
  2) On-the-fly inference:            --checkpoint pretrained.pth --panglao_path panglao.h5ad

Architecture:
  Cell embedding (128-dim) → Linear(128→100, trainable)  → concat (200-dim) → CNN → output
  Drug graph → GCN → 100-dim       ──────────────────────↗

Usage:
    # Mode 1: pre-computed embeddings (fast)
    python train_drug.py --data_dir ./data --emb_path cell_emb.npz --outdir ./output_drug

    # Mode 2: on-the-fly scMomer inference
    python train_drug.py --data_dir ./data --checkpoint pretrained.pth \
        --panglao_path panglao_10000.h5ad --outdir ./output_drug
"""

import os, sys, argparse, types, logging, copy
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from model.scmomer import scMomer
from model.read_data import (prepare_drug_data, parse_drug_response, split_drug_data,
                              load_drug_features, calculate_graph_feat,
                              collate_drug_batch, MAX_ATOMS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ====================================================================
# Student Encoder
# ====================================================================
class StudentEncoder(nn.Module):
    def __init__(self, input_dim=16906, output_dim=128, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, rna_value):
        return self.encoder(rna_value)


# ====================================================================
# Dataset with pre-computed embeddings
# ====================================================================
class DrugResponseEmbDataset(Dataset):
    """Drug response dataset with pre-computed cell embeddings."""

    def __init__(self, data_idx, drug_feature, emb_dict):
        self.data_idx = data_idx
        n = len(data_idx)
        nb_drug_feat = next(iter(drug_feature.values()))[0].shape[1]
        emb_dim = next(iter(emb_dict.values())).shape[0]

        self.feat_mats = np.zeros((n, MAX_ATOMS, nb_drug_feat), dtype='float32')
        self.adj_mats = np.zeros((n, MAX_ATOMS, MAX_ATOMS), dtype='float32')
        self.cell_embs = np.zeros((n, emb_dim), dtype='float32')
        self.targets = np.zeros(n, dtype='float32')

        missing = 0
        for idx, (cell_line, pubchem_id, value, _) in enumerate(data_idx):
            feat_mat, adj_list, _ = drug_feature[pubchem_id]
            feat, adj = calculate_graph_feat(feat_mat, adj_list)
            self.feat_mats[idx] = feat
            self.adj_mats[idx] = adj
            if cell_line in emb_dict:
                self.cell_embs[idx] = emb_dict[cell_line]
            else:
                missing += 1
            self.targets[idx] = value
        if missing > 0:
            logger.warning("Missing embeddings for %d samples (filled 0)", missing)

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        return self.feat_mats[idx], self.adj_mats[idx], self.cell_embs[idx], self.targets[idx]


def collate_emb_batch(batch):
    feat_mats, adj_mats, cell_embs, targets = zip(*batch)
    feat_mats = np.stack(feat_mats)
    adj_mats = np.stack(adj_mats)
    data_list = []
    for i in range(feat_mats.shape[0]):
        x = torch.from_numpy(feat_mats[i])
        adj = torch.from_numpy(adj_mats[i])
        edge_index, _ = dense_to_sparse(adj)
        data_list.append(PyGData(x=x, edge_index=edge_index))
    drug_batch = PyGBatch.from_data_list(data_list)
    emb_tensor = torch.tensor(np.stack(cell_embs), dtype=torch.float32)
    target_tensor = torch.tensor(np.stack(targets), dtype=torch.float32)
    return drug_batch, emb_tensor, target_tensor


# ====================================================================
# Drug Response Model (DeepCDR-style GCN + CNN fusion)
# ====================================================================
class DrugResponseModel(nn.Module):
    def __init__(self, drug_input_dim, units_list=None, regression=True):
        super().__init__()
        if units_list is None:
            units_list = [256, 256, 256]
        self.regression = regression

        self.gcn_layers = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()
        in_dim = drug_input_dim
        for out_dim in units_list:
            self.gcn_layers.append(GCNConv(in_dim, out_dim))
            self.gcn_bns.append(nn.BatchNorm1d(out_dim))
            in_dim = out_dim
        self.gcn_final = GCNConv(in_dim, 100)
        self.gcn_bn_final = nn.BatchNorm1d(100)

        self.fusion_fc = nn.Sequential(nn.Linear(200, 300), nn.Tanh(), nn.Dropout(0.1))
        self.conv = nn.Sequential(
            nn.Conv1d(1, 30, 150), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(30, 10, 5), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(10, 5, 5), nn.ReLU(), nn.MaxPool1d(3),
        )
        self.final = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(30, 1))

    def forward(self, drug_data, cell_embed):
        x, edge_index, batch = drug_data.x, drug_data.edge_index, drug_data.batch
        for i, gcn in enumerate(self.gcn_layers):
            x = F.dropout(F.relu(self.gcn_bns[i](gcn(x, edge_index))), 0.1, training=self.training)
        x = F.dropout(F.relu(self.gcn_bn_final(self.gcn_final(x, edge_index))), 0.1, training=self.training)
        x = self.fusion_fc(torch.cat([global_max_pool(x, batch), cell_embed], 1))
        x = self.final(self.conv(x.unsqueeze(1)))
        return x.squeeze(-1) if self.regression else torch.sigmoid(x).squeeze(-1)


# ====================================================================
# Model Loading
# ====================================================================
def load_scmomer(checkpoint_path, rna_dim, projection_dim=128, device="cpu"):
    args = types.SimpleNamespace(projection_dim=projection_dim, normalize=True)
    encoder = StudentEncoder(input_dim=rna_dim, output_dim=projection_dim)
    model = scMomer(args, atac_config=None, rna_decoder=None,
                    atac_decoder=None, encoder=encoder)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    clean_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    model.eval()
    return model


# ====================================================================
# Metrics
# ====================================================================
def compute_metrics(preds, truths, classification=False):
    preds = np.array(preds)
    truths = np.array(truths)
    if classification:
        from sklearn.metrics import roc_auc_score, accuracy_score
        try:
            auc = roc_auc_score(truths, preds)
        except ValueError:
            auc = 0.0
        acc = accuracy_score(truths.astype(int), (preds > 0.5).astype(int))
        return {"AUROC": auc, "ACC": acc}
    else:
        mask = ~(np.isnan(preds) | np.isnan(truths))
        if mask.sum() < 2:
            return {"PCC": 0.0}
        pcc, _ = stats.pearsonr(preds[mask], truths[mask])
        return {"PCC": pcc}


# ====================================================================
# Training
# ====================================================================
def train(drug_model, projection, train_loader, val_loader,
          lr, max_epochs, patience, device, classification, save_path,
          scmomer=None):
    """Train drug response model. scmomer is None when using pre-computed embeddings."""
    trainable_params = list(projection.parameters()) + list(drug_model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.BCELoss() if classification else nn.MSELoss()

    best_score = -float('inf')
    best_state = None
    wait = 0

    for epoch in range(1, max_epochs + 1):
        drug_model.train()
        projection.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} [train]", leave=False)

        for drug_batch, emb_or_gexpr, targets in pbar:
            drug_batch = drug_batch.to(device)
            emb_or_gexpr = emb_or_gexpr.to(device)
            targets = targets.to(device)

            if scmomer is not None:
                with torch.no_grad():
                    cell_emb = scmomer.get_cell_embedding(emb_or_gexpr)
            else:
                cell_emb = emb_or_gexpr

            cell_proj = projection(cell_emb)
            preds = drug_model(drug_batch, cell_proj)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 4.0)
            optimizer.step()

            train_loss += loss.item() * emb_or_gexpr.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)

        train_loss /= len(train_loader.dataset)

        val_metrics = evaluate(drug_model, projection, val_loader, device, classification, scmomer)
        val_score = val_metrics["AUROC"] if classification else val_metrics["PCC"]
        scheduler.step(val_score)

        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        logger.info("  Epoch %3d  train_loss=%.4f  val_%s  lr=%.2e  %s",
                     epoch, train_loss, metric_str,
                     optimizer.param_groups[0]['lr'],
                     "*" if val_score > best_score else "")

        if val_score > best_score:
            best_score = val_score
            best_state = {
                'drug_model': copy.deepcopy(drug_model.state_dict()),
                'projection': copy.deepcopy(projection.state_dict()),
                'epoch': epoch, 'best_score': best_score,
            }
            wait = 0
            torch.save(best_state, save_path)
            logger.info("  Model saved (score=%.4f)", best_score)
        else:
            wait += 1
            if wait >= patience:
                logger.info("  Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        drug_model.load_state_dict(best_state['drug_model'])
        projection.load_state_dict(best_state['projection'])
    logger.info("Best val score: %.4f", best_score)
    return drug_model, projection


def evaluate(drug_model, projection, loader, device, classification, scmomer=None):
    drug_model.eval()
    projection.eval()
    all_preds, all_truths = [], []

    with torch.no_grad():
        for drug_batch, emb_or_gexpr, targets in loader:
            drug_batch = drug_batch.to(device)
            emb_or_gexpr = emb_or_gexpr.to(device)

            if scmomer is not None:
                cell_emb = scmomer.get_cell_embedding(emb_or_gexpr)
            else:
                cell_emb = emb_or_gexpr

            cell_proj = projection(cell_emb)
            preds = drug_model(drug_batch, cell_proj)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_truths.extend(targets.numpy().tolist())

    return compute_metrics(all_preds, all_truths, classification)


# ====================================================================
# Main
# ====================================================================
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    p.add_argument("--data_dir", required=True, help="Directory with drug response data files")
    p.add_argument("--emb_path", default='./cell_emb.npz',
                   help="Pre-computed cell embeddings .npz (Mode 1). "
                        "If not provided, use --checkpoint + --panglao_path (Mode 2).")
    p.add_argument("--panglao_path", default=None,
                   help="Path to panglao_10000.h5ad (required for Mode 2)")
    p.add_argument("--checkpoint", default=None,
                   help="Pretrained scMomer checkpoint (required for Mode 2)")

    # Task
    p.add_argument("--classification", action="store_true", default=True,
                   help="Classification mode (auto-loads ${data_dir}/IC50_thred.txt)")

    # Training
    p.add_argument("--projection_dim", type=int, default=128)
    p.add_argument("--outdir", default="./output_drug/")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, default=0)
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Data paths ----
    drug_info_file = os.path.join(args.data_dir, "1.Drug_listMon Jun 24 09_00_55 2019.csv")
    cell_line_info_file = os.path.join(args.data_dir, "Cell_lines_annotations_20181226.txt")
    cancer_response_file = os.path.join(args.data_dir, "GDSC_IC50.csv")
    drug_feature_dir = os.path.join(args.data_dir, "drug_graph_feat")
    ic50_thred_file = os.path.join(args.data_dir, "IC50_thred.txt") if args.classification else None

    # ---- Load drug features & parse data ----
    logger.info("Loading drug response data ...")
    drug_feature = load_drug_features(drug_feature_dir)
    data_idx = parse_drug_response(
        drug_info_file, cell_line_info_file,
        cancer_response_file, drug_feature_dir,
        classification=args.classification, ic50_thred_file=ic50_thred_file)
    data_train, data_val, data_test = split_drug_data(
        data_idx, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)

    # ---- Mode 1: pre-computed embeddings ----
    if args.emb_path is not None:
        logger.info("Mode 1: using pre-computed embeddings from %s", args.emb_path)
        emb_data = np.load(args.emb_path)
        cell_ids = emb_data['cell_ids']
        embeddings = emb_data['embeddings']
        emb_dict = {cell_ids[i]: embeddings[i] for i in range(len(cell_ids))}
        logger.info("Loaded %d cell embeddings, dim=%d", len(emb_dict), embeddings.shape[1])

        train_dataset = DrugResponseEmbDataset(data_train, drug_feature, emb_dict)
        val_dataset = DrugResponseEmbDataset(data_val, drug_feature, emb_dict)
        test_dataset = DrugResponseEmbDataset(data_test, drug_feature, emb_dict)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_emb_batch)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_emb_batch)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_emb_batch)

        emb_dim = embeddings.shape[1]
        scmomer = None

    # ---- Mode 2: on-the-fly scMomer inference ----
    else:
        assert args.checkpoint is not None and args.panglao_path is not None, \
            "Mode 2 requires --checkpoint and --panglao_path"

        logger.info("Mode 2: on-the-fly scMomer inference")
        gexpr_file = os.path.join(args.data_dir,
                                  "genomic_expression_561celllines_697genes_demap_features.csv")

        train_dataset, val_dataset, test_dataset = prepare_drug_data(
            drug_info_file=drug_info_file,
            cell_line_info_file=cell_line_info_file,
            cancer_response_file=cancer_response_file,
            drug_feature_dir=drug_feature_dir,
            gexpr_file=gexpr_file,
            panglao_path=args.panglao_path,
            val_ratio=args.val_ratio, test_ratio=args.test_ratio,
            seed=args.seed, classification=args.classification,
            ic50_thred_file=ic50_thred_file)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_drug_batch)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_drug_batch)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_drug_batch)

        rna_dim = train_dataset.gexpr_data.shape[1]
        scmomer = load_scmomer(args.checkpoint, rna_dim, args.projection_dim, device)
        emb_dim = args.projection_dim

    # ---- Build trainable components ----
    projection = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.Tanh(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 100),
            nn.ReLU()
        ).to(device)
    drug_input_dim = next(iter(drug_feature.values()))[0].shape[1]
    drug_model = DrugResponseModel(
        drug_input_dim=drug_input_dim,
        regression=not args.classification,
    ).to(device)

    logger.info("Projection params: %d", sum(p.numel() for p in projection.parameters()))
    logger.info("Drug model params: %d", sum(p.numel() for p in drug_model.parameters()))

    # ---- Train ----
    save_path = os.path.join(args.outdir, "drug_response_model.pth")
    drug_model, projection = train(
        drug_model, projection, train_loader, val_loader,
        lr=args.lr, max_epochs=args.max_epochs,
        patience=args.patience, device=device,
        classification=args.classification,
        save_path=save_path, scmomer=scmomer)

    # ---- Test ----
    logger.info("Evaluating on test set ...")
    test_metrics = evaluate(drug_model, projection, test_loader, device,
                            args.classification, scmomer)
    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
    logger.info("Test results: %s", metric_str)

    print(f"\n{'=' * 50}")
    print(f"  Test results: {metric_str}")
    print(f"  Model saved to: {save_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
