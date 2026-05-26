"""
One-time extraction of cell line embeddings using scMomer.

Loads aligned gene expression for all cell lines, runs scMomer.get_cell_embedding(),
saves the 128-dim embeddings as .npy for downstream drug response training.

Usage:
    python extract_cell_emb.py --data_dir ./data --panglao_path panglao_10000.h5ad \
        --checkpoint pretrained_multimodal.pth --output cell_emb.npz
"""

import os, sys, argparse, types, logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from tqdm import tqdm

from model.scmomer import scMomer
from model.read_data import load_drug_info, load_cell_line_info, align_gene_expression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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


def load_scmomer(checkpoint_path, rna_dim, projection_dim=128, device="cpu"):
    args = types.SimpleNamespace(projection_dim=projection_dim, normalize=True)
    encoder = StudentEncoder(input_dim=rna_dim, output_dim=projection_dim)
    model = scMomer(args, atac_config=None, rna_decoder=None,
                    atac_decoder=None, encoder=encoder)

    logger.info("Loading scMomer checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    clean_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model = model.to(device)
    model.eval()
    return model


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", required=True,
                   help="Directory with genomic_expression_561celllines_697genes_demap_features.csv "
                        "and Cell_lines_annotations.txt")
    p.add_argument("--panglao_path", required=True, help="Path to panglao_10000.h5ad")
    p.add_argument("--checkpoint", required=True, help="Pretrained scMomer checkpoint (.pth)")
    p.add_argument("--projection_dim", type=int, default=128)
    p.add_argument("--output", default="cell_emb.npz", help="Output .npz file")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=int, default=1)
    return p


def main():
    args = build_parser().parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Load gene expression ----
    gexpr_file = os.path.join(args.data_dir,
                              "genomic_expression_561celllines_697genes_demap_features.csv")
    logger.info("Loading gene expression: %s", gexpr_file)
    gexpr_df = pd.read_csv(gexpr_file, sep=',', header=0, index_col=[0])

    # ---- Align to model gene space ----
    logger.info("Aligning to panglao gene space ...")
    gexpr_aligned = align_gene_expression(gexpr_df, args.panglao_path)

    cell_ids = list(gexpr_aligned.index)
    n_cells = len(cell_ids)
    rna_dim = gexpr_aligned.shape[1]
    logger.info("Cell lines: %d, genes: %d", n_cells, rna_dim)

    # ---- Load scMomer ----
    model = load_scmomer(args.checkpoint, rna_dim, args.projection_dim, device)

    # ---- Extract embeddings ----
    logger.info("Extracting embeddings ...")
    all_embs = np.zeros((n_cells, args.projection_dim), dtype=np.float32)
    gexpr_vals = gexpr_aligned.values.astype(np.float32)

    for start in tqdm(range(0, n_cells, args.batch_size), desc="Extracting"):
        end = min(start + args.batch_size, n_cells)
        batch = torch.tensor(gexpr_vals[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            emb = model.get_cell_embedding(batch)
        all_embs[start:end] = emb.cpu().numpy()

    # ---- Save ----
    np.savez(args.output, cell_ids=np.array(cell_ids), embeddings=all_embs)
    logger.info("Saved: %s  (%d cells x %d dim)", args.output, n_cells, args.projection_dim)


if __name__ == "__main__":
    main()
