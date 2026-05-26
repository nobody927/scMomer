import torch
import argparse
import os
import types
import warnings
import logging
import numpy as np
import scipy.sparse
import anndata as ad
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from model.scmomer import scMomer
from utils import seed_all
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class StudentEncoder(nn.Module):
    """Shallow MLP: RNA -> ATAC representation (1 hidden layer)"""

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


class CellDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if scipy.sparse.issparse(x):
            x = np.asarray(x.todense(), dtype=np.float32).flatten()
        else:
            x = np.asarray(x, dtype=np.float32).flatten()
        return torch.from_numpy(x), self.y[idx]


def evaluate_classification(features, labels, seed=2024):
    warnings.filterwarnings("ignore", category=UserWarning)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed)
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    acc_scores = cross_val_score(clf, features, labels, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(clf, features, labels, cv=cv, scoring='f1_macro')

    return {
        "accuracy_mean": acc_scores.mean(),
        "accuracy_std": acc_scores.std(),
        "accuracy_folds": acc_scores.tolist(),
        "f1_mean": f1_scores.mean(),
        "f1_std": f1_scores.std(),
        "f1_folds": f1_scores.tolist()
    }


def load_model(checkpoint_path, rna_dim, device="cpu"):
    """Load a complete trained model from a single checkpoint."""
    args = types.SimpleNamespace(projection_dim=128, normalize=True)
    encoder = StudentEncoder(input_dim=rna_dim, output_dim=128)

    model = scMomer(
        args,
        atac_config=None,
        rna_decoder=None,
        atac_decoder=None,
        encoder=encoder,
    )

    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    clean_sd = {}
    for k, v in sd.items():
        clean_sd[k[7:] if k.startswith('module.') else k] = v

    model.load_state_dict(clean_sd, strict=False)
    logger.info("  Loaded %d keys", len(clean_sd))

    model = model.to(device)
    model.eval()
    return model


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate scMomer embeddings and save representations.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--h5ad", required=True, help="Input h5ad file (RNA, already processed)")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    p.add_argument("--label_col", default="cell_type", help="Column in obs for labels")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--projection_dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", type=int, default=1)
    p.add_argument("--save_dir", default="./eval_output/", help="Directory to save results")
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)

    # ========================================================
    # 1. Load data from h5ad
    # ========================================================
    logger.info("Loading: %s", args.h5ad)
    adata = ad.read_h5ad(args.h5ad)
    logger.info("Shape: %s", adata.shape)

    label_col = args.label_col
    if label_col not in adata.obs.columns:
        candidates = ["cell_type", "CellType", "celltype", "cell_type_major",
                       "cell_type_minor", "cluster", "annotation", "labels"]
        for c in candidates:
            if c in adata.obs.columns:
                label_col = c
                break
    logger.info("Label column: '%s'", label_col)

    le = LabelEncoder()
    labels = le.fit_transform(adata.obs[label_col].astype(str).values)
    n_classes = len(le.classes_)
    logger.info("Classes: %d  Samples: %d", n_classes, len(labels))

    X = adata.X
    rna_dim = adata.shape[1]

    dataset = CellDataset(X, labels)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ========================================================
    # 2. Load model
    # ========================================================
    logger.info("Loading model...")
    model = load_model(args.checkpoint, rna_dim, device=device)

    # ========================================================
    # 3. Extract embeddings
    # ========================================================
    logger.info("Extracting embeddings...")
    all_embs = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Extracting"):
            x = x.to(device)
            emb = model.get_cell_embedding(x)
            all_embs.append(emb.cpu().numpy())
            all_labels.extend(y.numpy())

    all_embs = np.vstack(all_embs)
    all_labels = np.array(all_labels)

    # ========================================================
    # 4. Evaluate
    # ========================================================
    logger.info("Evaluating...")
    num_classes = len(np.unique(all_labels))

    kmeans = KMeans(n_clusters=num_classes, random_state=args.seed, n_init=10)
    pred_clusters = kmeans.fit_predict(all_embs)
    ari = adjusted_rand_score(all_labels, pred_clusters)
    nmi = normalized_mutual_info_score(all_labels, pred_clusters)
    ami = adjusted_mutual_info_score(all_labels, pred_clusters)

    cls_result = evaluate_classification(all_embs, all_labels, seed=args.seed)

    print(f"\n{'=' * 70}")
    print("Evaluation Results")
    print(f"{'=' * 70}")
    print(f"  ARI: {ari:.4f} | NMI: {nmi:.4f} | AMI: {ami:.4f}")
    print(f"  Accuracy: {cls_result['accuracy_mean']:.4f}+/-{cls_result['accuracy_std']:.4f}")
    print(f"            Folds: [{', '.join([f'{x:.4f}' for x in cls_result['accuracy_folds']])}]")
    print(f"  Macro F1: {cls_result['f1_mean']:.4f}+/-{cls_result['f1_std']:.4f}")
    print(f"            Folds: [{', '.join([f'{x:.4f}' for x in cls_result['f1_folds']])}]")

    # ========================================================
    # 5. Save representations to h5ad
    # ========================================================
    emb_adata = ad.AnnData(
        X=all_embs.astype(np.float32),
        obs=adata.obs.copy(),
    )
    emb_adata.obs['label'] = all_labels
    emb_adata.obs['label_name'] = le.inverse_transform(all_labels)
    emb_adata.uns['metrics'] = {
        'ari': ari,
        'nmi': nmi,
        'ami': ami,
        'accuracy_mean': cls_result['accuracy_mean'],
        'accuracy_std': cls_result['accuracy_std'],
        'f1_mean': cls_result['f1_mean'],
        'f1_std': cls_result['f1_std'],
    }

    emb_path = os.path.join(args.save_dir, "representations.h5ad")
    emb_adata.write_h5ad(emb_path)
    logger.info("Representations saved to: %s", emb_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
