"""
Preprocess paired multimodal (RNA + ATAC) data for scMomer.

Processing steps:
  RNA:  filter cells (<200 genes) -> filter genes (<1% cells) -> align to reference
        gene space -> normalize (10k) -> log1p (base 2)
  ATAC: filter features (<1% cells) -> filter cells (<1000 peaks) -> keep raw counts

Cells are intersected at the end to ensure RNA and ATAC share the same cells.

Usage:
    python data_process_scmomer.py \
        --input data.h5mu \
        --ref panglao_reference.h5ad \
        --output processed.h5mu
"""

import argparse
import numpy as np
import anndata as ad
import scanpy as sc
import muon as mu
from scipy import sparse


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="Input h5mu file (raw multimodal data)")
    p.add_argument("--ref", required=True,
                   help="Reference h5ad for RNA gene space alignment (e.g. panglao)")
    p.add_argument("--output", required=True, help="Output h5mu path")
    p.add_argument("--min_genes", type=int, default=200,
                   help="Min genes per cell for RNA cell filtering")
    p.add_argument("--min_peaks", type=int, default=1000,
                   help="Min peaks per cell for ATAC cell filtering")
    p.add_argument("--min_cells_pct", type=float, default=0.01,
                   help="Min fraction of cells a feature must be detected in (for both RNA and ATAC)")
    p.add_argument("--target_sum", type=float, default=1e4,
                   help="Normalization target sum for RNA")
    p.add_argument("--log_base", type=float, default=2,
                   help="Log1p base for RNA (0 = skip log)")
    return p


def align_gene_space(rna, ref_adata):
    """
    Align RNA data to reference gene space using O(n) dict lookup.
    Only keeps genes present in both RNA and reference.
    """
    ref_genes = ref_adata.var_names.tolist()
    rna_genes = rna.var_names.tolist()

    # O(1) lookup
    rna_gene_to_idx = {g: i for i, g in enumerate(rna_genes)}

    col_indices = []
    row_indices = []
    data_values = []

    X_rna = rna.X
    if sparse.issparse(X_rna) and not sparse.isspmatrix_csr(X_rna):
        X_rna = X_rna.tocsr()

    matched = 0
    for col_idx, gene in enumerate(ref_genes):
        if gene in rna_gene_to_idx:
            rna_col_idx = rna_gene_to_idx[gene]
            col_data = X_rna[:, rna_col_idx]
            if sparse.issparse(col_data):
                col_data = col_data.toarray().flatten()
            nonzero_mask = col_data != 0
            nonzero_rows = np.where(nonzero_mask)[0]
            if len(nonzero_rows) > 0:
                row_indices.append(nonzero_rows)
                col_indices.append(np.full(len(nonzero_rows), col_idx))
                data_values.append(col_data[nonzero_rows])
            matched += 1

    print(f"  Gene alignment: {matched} / {len(ref_genes)} genes matched")

    if row_indices:
        all_rows = np.concatenate(row_indices)
        all_cols = np.concatenate(col_indices)
        all_data = np.concatenate(data_values)
        counts = sparse.coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(rna.n_obs, len(ref_genes)),
            dtype=np.float32
        ).tocsr()
    else:
        counts = sparse.csr_matrix((rna.n_obs, len(ref_genes)), dtype=np.float32)

    aligned = ad.AnnData(X=counts)
    aligned.var_names = ref_genes
    aligned.obs_names = rna.obs_names
    aligned.obs = rna.obs.copy()
    aligned.uns = ref_adata.uns.copy()
    return aligned


def main():
    args = build_parser().parse_args()

    # ==========================================
    # 1. Load data
    # ==========================================
    print("1. Loading data...")
    mdata = mu.read(args.input, backed=False)
    rna = mdata.mod["rna"].copy()
    atac = mdata.mod["atac"].copy()
    print(f"   RNA: {rna.shape}")
    print(f"   ATAC: {atac.shape}")

    # ==========================================
    # 2. RNA processing
    # ==========================================
    print("2. RNA processing...")

    # 2a. Cell filtering: remove cells with < min_genes
    print(f"   Filtering cells: min_genes={args.min_genes}")
    sc.pp.filter_cells(rna, min_genes=args.min_genes)
    print(f"   After cell filter: {rna.shape}")

    # 2b. Gene filtering: remove genes detected in < min_cells_pct of cells
    min_cells = int(args.min_cells_pct * rna.n_obs)
    print(f"   Filtering genes: min_cells={min_cells} ({args.min_cells_pct*100:.1f}%)")
    sc.pp.filter_genes(rna, min_cells=min_cells)
    print(f"   After gene filter: {rna.shape}")

    # 2c. Align to reference gene space
    print("3. Aligning RNA to reference gene space...")
    ref_adata = sc.read_h5ad(args.ref)
    rna = align_gene_space(rna, ref_adata)

    # 2d. Normalize + log1p
    print(f"   Normalizing: target_sum={args.target_sum}")
    sc.pp.normalize_total(rna, target_sum=args.target_sum)
    if args.log_base > 0:
        print(f"   Log1p: base={args.log_base}")
        sc.pp.log1p(rna, base=args.log_base)

    # ==========================================
    # 3. ATAC processing
    # ==========================================
    print("4. ATAC processing...")

    # 3a. Feature filtering: remove features accessible in < min_cells_pct of cells
    min_cells_atac = int(args.min_cells_pct * atac.n_obs)
    print(f"   Filtering features: min_cells={min_cells_atac} ({args.min_cells_pct*100:.1f}%)")
    sc.pp.filter_genes(atac, min_cells=min_cells_atac)
    print(f"   After feature filter: {atac.shape}")

    # 3b. Cell filtering: remove cells with < min_peaks
    print(f"   Filtering cells: min_peaks={args.min_peaks}")
    if 'n_genes_by_counts' not in atac.obs:
        atac.obs['n_peaks_by_counts'] = np.array((atac.X > 0).sum(axis=1)).flatten()
    sc.pp.filter_cells(atac, min_genes=args.min_peaks)
    print(f"   After cell filter: {atac.shape}")

    # No normalization for ATAC (kept as raw counts, binarized later in read_data.py)

    # ==========================================
    # 4. Intersect cells
    # ==========================================
    print("5. Intersecting cells...")
    common_cells = list(set(rna.obs_names) & set(atac.obs_names))
    common_cells.sort()
    print(f"   Common cells: {len(common_cells)}")

    rna = rna[common_cells, :].copy()
    atac = atac[common_cells, :].copy()
    print(f"   Final RNA: {rna.shape}")
    print(f"   Final ATAC: {atac.shape}")

    # ==========================================
    # 5. Save
    # ==========================================
    print("6. Saving...")
    mdata_processed = mu.MuData({'rna': rna, 'atac': atac})
    mdata_processed.write(args.output)
    print(f"   Saved to: {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
