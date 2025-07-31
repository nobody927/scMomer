import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from scipy import sparse
import muon as mu


panglao = sc.read_h5ad('./data/panglao_10000.h5ad')
mdata = mu.read('/home/lyh/project/sc/main/model/data/fetal.h5mu', backed=False)
rna = mdata.mod["rna"]
counts = sparse.lil_matrix((rna.X.shape[0],panglao.X.shape[1]),dtype=np.float32)
ref = panglao.var_names.tolist()
obj = rna.var_names.tolist()

for i in range(len(ref)):
    if ref[i] in obj:
        loc = obj.index(ref[i])
        counts[:,i] = rna.X[:,loc]

counts = counts.tocsr()
new = ad.AnnData(X=counts)
new.var_names = ref
new.obs_names = rna.obs_names
new.obs = rna.obs
new.uns = panglao.uns

# sc.pp.filter_cells(new, min_genes=200)
# sc.pp.normalize_total(new, target_sum=1e4)
# sc.pp.log1p(new, base=2)
mdata_new = mu.MuData(mdata.mod)
mdata_new1 = mu.MuData({
    'rna': new,
    'atac': mdata_new.mod["atac"],# 确保new是anndata兼容的AnnData对象
})

mdata_new1.write("./data/preprocessed_fetal_new.h5mu")

a = 1