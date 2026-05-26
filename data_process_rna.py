"""
Align query h5ad data to reference gene space and preprocess.

Usage:
    python preprocess.py \
        --ref /path/to/panglao_10000.h5ad \
        --query /path/to/new_data.h5ad \
        --output /path/to/output.h5ad \
        --min_genes 200
"""

import argparse
import numpy as np
import anndata as ad
import scanpy as sc
from scipy import sparse


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ref", required=True, help="Reference h5ad (defines the gene space)")
    p.add_argument("--query", required=True, help="Query h5ad to align")
    p.add_argument("--output", required=True, help="Output h5ad path")
    p.add_argument("--min_genes", type=int, default=200, help="Min genes per cell filter")
    p.add_argument("--min_cells", type=int, default=0, help="Min cells per gene filter (0 = skip)")
    p.add_argument("--target_sum", type=float, default=1e4, help="Normalization target sum")
    p.add_argument("--log_base", type=float, default=2, help="Log1p base (0 = skip log)")
    return p


def main():
    args = build_parser().parse_args()

    # Load data
    print(f"Loading reference: {args.ref}")
    ref_adata = sc.read_h5ad(args.ref)
    print(f"  Reference shape: {ref_adata.shape}")

    print(f"Loading query: {args.query}")
    query_adata = sc.read_h5ad(args.query)
    print(f"  Query shape: {query_adata.shape}")

    ref_genes = ref_adata.var_names.tolist()
    query_genes = query_adata.var_names.tolist()

    # Build gene -> index mapping for query (O(1) lookup)
    query_gene_to_idx = {g: i for i, g in enumerate(query_genes)}

    # Find overlapping genes and build sparse matrix efficiently
    col_indices = []
    row_indices = []
    data_values = []

    X_query = query_adata.X
    # Convert to CSR for efficient row slicing
    if sparse.issparse(X_query) and not sparse.isspmatrix_csr(X_query):
        X_query = X_query.tocsr()

    matched = 0
    for col_idx, gene in enumerate(ref_genes):
        if gene in query_gene_to_idx:
            query_col_idx = query_gene_to_idx[gene]
            col_data = X_query[:, query_col_idx]
            if sparse.issparse(col_data):
                col_data = col_data.toarray().flatten()
            nonzero_mask = col_data != 0
            nonzero_rows = np.where(nonzero_mask)[0]
            if len(nonzero_rows) > 0:
                row_indices.append(nonzero_rows)
                col_indices.append(np.full(len(nonzero_rows), col_idx))
                data_values.append(col_data[nonzero_rows])
            matched += 1

    print(f"Matched genes: {matched} / {len(ref_genes)}")

    # Build sparse matrix in COO format
    if row_indices:
        all_rows = np.concatenate(row_indices)
        all_cols = np.concatenate(col_indices)
        all_data = np.concatenate(data_values)
        counts = sparse.coo_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(query_adata.n_obs, len(ref_genes)),
            dtype=np.float32
        ).tocsr()
    else:
        counts = sparse.csr_matrix((query_adata.n_obs, len(ref_genes)), dtype=np.float32)

    # Create aligned AnnData
    new = ad.AnnData(X=counts)
    new.var_names = ref_genes
    new.obs_names = query_adata.obs_names
    new.obs = query_adata.obs
    new.uns = ref_adata.uns

    # Preprocessing
    print(f"Filtering cells: min_genes={args.min_genes}")
    sc.pp.filter_cells(new, min_genes=args.min_genes)
    print(f"  After cell filter: {new.shape}")

    if args.min_cells > 0:
        print(f"Filtering genes: min_cells={args.min_cells}")
        sc.pp.filter_genes(new, min_cells=args.min_cells)
        print(f"  After gene filter: {new.shape}")

    print(f"Normalizing: target_sum={args.target_sum}")
    sc.pp.normalize_total(new, target_sum=args.target_sum)

    if args.log_base > 0:
        print(f"Log1p: base={args.log_base}")
        sc.pp.log1p(new, base=args.log_base)

    print(f"Saving: {args.output}")
    new.write(args.output)
    print(f"Done. Final shape: {new.shape}")


if __name__ == "__main__":
    main()
