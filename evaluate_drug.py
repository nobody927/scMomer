"""
Evaluate a trained drug response model on the test set.

Two modes:
  1) Pre-computed embeddings:  --emb_path cell_emb.npz
  2) On-the-fly inference:     --checkpoint pretrained.pth --panglao_path panglao.h5ad

Usage:
    python evaluate_drug.py --data_dir ./data --emb_path cell_emb.npz \
        --drug_checkpoint output_drug/drug_response_model.pth

    python evaluate_drug.py --data_dir ./data --checkpoint pretrained.pth \
        --panglao_path panglao_10000.h5ad --drug_checkpoint output_drug/drug_response_model.pth
"""

import os, sys, argparse, types, logging
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool

from model.scmomer import scMomer
from model.read_data import (prepare_drug_data, parse_drug_response, split_drug_data,
                              load_drug_features, align_gene_expression)
from train_drug import (DrugResponseModel, DrugResponseEmbDataset, collate_emb_batch,
                         load_scmomer, compute_metrics, StudentEncoder)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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

    return compute_metrics(all_preds, all_truths, classification), all_preds, all_truths


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    p.add_argument("--data_dir", required=True, help="Directory with drug response data files")
    p.add_argument("--emb_path", default='./cell_emb.npz', help="Pre-computed cell embeddings .npz")
    p.add_argument("--panglao_path", default=None, help="Path to panglao_10000.h5ad (Mode 2)")
    p.add_argument("--checkpoint", default=None, help="Pretrained scMomer checkpoint (Mode 2)")

    # Models
    p.add_argument("--drug_checkpoint", default='./output_drug/drug_response_model.pth', help="Trained drug response model (.pth)")
    p.add_argument("--projection_dim", type=int, default=128)

    # Task
    p.add_argument("--classification", action="store_true", default=True)

    # Eval
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
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
    drug_feature = load_drug_features(drug_feature_dir)
    data_idx = parse_drug_response(
        drug_info_file, cell_line_info_file,
        cancer_response_file, drug_feature_dir,
        classification=args.classification, ic50_thred_file=ic50_thred_file)
    _, _, data_test = split_drug_data(data_idx, seed=args.seed)

    # ---- Mode 1: pre-computed embeddings ----
    if args.emb_path is not None:
        logger.info("Mode 1: using pre-computed embeddings from %s", args.emb_path)
        emb_data = np.load(args.emb_path)
        cell_ids = emb_data['cell_ids']
        embeddings = emb_data['embeddings']
        emb_dict = {cell_ids[i]: embeddings[i] for i in range(len(cell_ids))}
        logger.info("Loaded %d cell embeddings, dim=%d", len(emb_dict), embeddings.shape[1])

        test_dataset = DrugResponseEmbDataset(data_test, drug_feature, emb_dict)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_emb_batch)
        emb_dim = embeddings.shape[1]
        scmomer = None

    # ---- Mode 2: on-the-fly scMomer inference ----
    else:
        assert args.checkpoint is not None and args.panglao_path is not None, \
            "Mode 2 requires --checkpoint and --panglao_path"

        logger.info("Mode 2: on-the-fly scMomer inference")
        import pandas as pd
        gexpr_file = os.path.join(args.data_dir,
                                  "genomic_expression_561celllines_697genes_demap_features.csv")

        _, _, test_dataset = prepare_drug_data(
            drug_info_file=drug_info_file,
            cell_line_info_file=cell_line_info_file,
            cancer_response_file=cancer_response_file,
            drug_feature_dir=drug_feature_dir,
            gexpr_file=gexpr_file,
            panglao_path=args.panglao_path,
            seed=args.seed, classification=args.classification,
            ic50_thred_file=ic50_thred_file)

        from train_drug import collate_drug_batch
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_drug_batch)

        rna_dim = test_dataset.gexpr_data.shape[1]
        scmomer = load_scmomer(args.checkpoint, rna_dim, args.projection_dim, device)
        emb_dim = args.projection_dim

    # ---- Load trained drug model ----
    drug_input_dim = next(iter(drug_feature.values()))[0].shape[1]
    drug_model = DrugResponseModel(drug_input_dim=drug_input_dim,
                                   regression=not args.classification)
    projection = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.Tanh(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 100),
            nn.ReLU()
        )
    logger.info("Loading drug model: %s", args.drug_checkpoint)
    ckpt = torch.load(args.drug_checkpoint, map_location='cpu')
    drug_model.load_state_dict(ckpt['drug_model'])
    projection.load_state_dict(ckpt['projection'])
    drug_model = drug_model.to(device).eval()
    projection = projection.to(device).eval()

    # ---- Evaluate ----
    logger.info("Evaluating on test set ...")
    metrics, preds, truths = evaluate(
        drug_model, projection, test_loader, device, args.classification, scmomer)

    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info("Test results: %s", metric_str)

    print(f"\n{'=' * 50}")
    print(f"  Test results: {metric_str}")
    print(f"  Evaluated: {len(preds)} samples")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
