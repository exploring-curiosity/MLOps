#!/usr/bin/env python
import os
import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description="Retrain Embedding MLP from MLflow Model Registry")
parser.add_argument("--emb_model_name",        type=str,   default="PannsMLP_PrimaryLabel", help="Registry name for Embedding MLP")

parser.add_argument(
    "--epochs", "-e",
    type=int,
    default=30,
    help="number of retraining epochs"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size"
)
parser.add_argument(
    "--lr",
    type=int,
    default=1e-3,
    help="Learning rate"
)
parser.add_argument(
    "--weight_decay",
    type=int,
    default=1e-4,
    help="Weight decay"
)
args = parser.parse_args()

birdclef_base_dir = os.getenv("BIRDCLEF_BASE_DIR", "/mnt/data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ---------------------- Load base model from MLflow Registry ----------------------
client     = MlflowClient()
model_name = args.emb_model_name

# fetch all versions registered under this name
all_versions = client.search_model_versions(f"name = '{model_name}'")
if not all_versions:
    print(f"No registered model found for '{model_name}'. Exiting.")
    sys.exit(0)

# pick the highest version number
latest = max(all_versions, key=lambda mv: int(mv.version))
version = latest.version

# load model via MLflow's models:/ URI (fetches from MinIO)
model_uri = f"models:/{model_name}/{version}"
print(f"Loading model from registry URI: {model_uri}")
model     = mlflow.pytorch.load_model(model_uri).to(DEVICE)

# ---------------------- MLflow Setup ----------------------
mlflow.set_experiment(f"{model_name}_Retrain")
if mlflow.active_run():
    mlflow.end_run()
run = mlflow.start_run(log_system_metrics=True)
print(f"MLFLOW_RUN_ID={run.info.run_id}")

mlflow.log_param("base_model_name",    model_name)
mlflow.log_param("base_model_version", version)

# log GPU/CPU info
gpu_info = next(
    (
        subprocess.run(cmd, capture_output=True, text=True).stdout
        for cmd in ["nvidia-smi", "rocm-smi"]
        if subprocess.run(f"command -v {cmd}", shell=True, capture_output=True).returncode == 0
    ),
    "No GPU found."
)
mlflow.log_text(gpu_info, "gpu-info.txt")

# ---------------------- Hyperparameters ----------------------
EPOCHS       = args.epochs
BATCH_SIZE     = args.batch_size
LR             = args.lr
WEIGHT_DECAY   = args.weight_decay 
HIDDEN_DIMS  = [2048, 1024, 512]
DROPOUT      = 0.5
MIXUP_ALPHA  = 0.4
FOCAL_GAMMA  = 2.0
BEST_CKPT    = "best_emb_mlp_retrain.pth"

mlflow.log_params({
    "batch_size":   BATCH_SIZE,
    "lr":           LR,
    "weight_decay": WEIGHT_DECAY,
    "epochs":       EPOCHS,
    "hidden_dims":  HIDDEN_DIMS,
    "dropout":      DROPOUT,
    "mixup_alpha":  MIXUP_ALPHA,
    "focal_gamma":  FOCAL_GAMMA
})

# ---------------------- Dataset Class ----------------------
class EmbeddingDataset(Dataset):
    def __init__(self, manifest, meta_csv, base, classes, key="embedding"):
        m = pd.read_csv(manifest)
        m["emb_path"] = (
            m["emb_path"].astype(str)
             .str.lstrip(os.sep)
             .apply(lambda p: os.path.join(base, "embeddings", p))
        )
        meta = pd.read_csv(meta_csv, usecols=["filename","secondary_labels"])
        meta["rid"]  = meta.filename.str.replace(r"\.ogg$","",regex=True)
        meta["secs"] = meta.secondary_labels.fillna("").str.split()
        sec_map = dict(zip(meta.rid, meta.secs))

        self.rows   = []
        self.idx_map = {c:i for i,c in enumerate(classes)}
        self.num_cls = len(classes)
        self.key     = key

        for _, row in tqdm(m.iterrows(), total=len(m), desc="Building dataset"):
            rid         = row.chunk_id.split("_chk")[0]
            labs        = [row.primary_label] + sec_map.get(rid, [])
            primary_cls = row.primary_label
            self.rows.append((row.emb_path, labs, primary_cls))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, labs, primary_cls = self.rows[i]
        arr = np.load(path)[self.key]                # (n_windows, emb_dim)
        x   = arr.mean(axis=0).astype(np.float32)    # (emb_dim,)
        y   = np.zeros(self.num_cls, dtype=np.float32)
        for c in labs:
            idx = self.idx_map.get(c)
            if idx is not None:
                y[idx] = 1.0
        prim_idx = self.idx_map.get(primary_cls, -1)
        return x, y, prim_idx

# ---------------------- Mixup ----------------------
def mixup(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha>0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

# ---------------------- Data Loading (VAL instead of TRAIN) ----------------------
TAXONOMY_CSV = os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "taxonomy.csv")
VAL_MAN      = os.path.join(birdclef_base_dir, "Features", "manifest_val.csv")
TEST_MAN     = os.path.join(birdclef_base_dir, "Features", "manifest_test.csv")
TRAIN_META   = os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "train.csv")
FEATURE_BASE = os.path.join(birdclef_base_dir, "Features")

tax_df     = pd.read_csv(TAXONOMY_CSV)
CLASSES    = sorted(tax_df["primary_label"].astype(str).tolist())
NUM_CLASSES= len(CLASSES)

train_ds = EmbeddingDataset(VAL_MAN, TRAIN_META, FEATURE_BASE, CLASSES)
test_ds  = EmbeddingDataset(TEST_MAN, TRAIN_META, FEATURE_BASE, CLASSES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)

# ---------------------- Loss, Optimizer, Scheduler ----------------------
# recompute pos_weight on VAL set for focal loss
counts = np.zeros(NUM_CLASSES, dtype=int)
for _, labs, _ in train_ds.rows:
    for c in labs:
        idx = train_ds.idx_map.get(c)
        if idx is not None:
            counts[idx] += 1
n   = len(train_ds)
neg = n - counts
pw  = np.ones(NUM_CLASSES, dtype=np.float32)
mask= counts>0
pw[mask] = neg[mask]/counts[mask]
pos_weight = torch.from_numpy(pw).to(DEVICE)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p_t = torch.exp(-bce)
        return ((1 - p_t)**self.gamma * bce).mean()

criterion = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(
    optimizer,
    max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,
    div_factor=10
)
scaler    = GradScaler()

# ---------------------- Training Loop ----------------------
best_f1, best_ap = 0.0, 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    train_bar = tqdm(train_loader, desc=f"[{epoch}/{EPOCHS}] Train", unit="batch")
    running_loss, total = 0.0, 0
    for xb, yb, _ in train_bar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        xb_m, ya, yb_m, lam = mixup(xb, yb)
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            logits = model(xb_m)
            loss   = lam*criterion(logits, ya) + (1-lam)*criterion(logits, yb_m)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = xb.size(0)
        running_loss += loss.item()*bs
        total       += bs
        train_bar.set_postfix({"loss": f"{running_loss/total:.4f}"})
    train_loss = running_loss/total

    model.eval()
    all_scores, all_tgts, all_prims = [], [], []
    val_loss, total = 0.0, 0
    with torch.no_grad():
        for xb, yb, prim_idx in tqdm(test_loader, desc=f"[{epoch}/{EPOCHS}] Eval ", unit="batch"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with autocast(device_type="cuda"):
                logits = model(xb)
                val_loss += criterion(logits, yb).item()*xb.size(0)
                scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_tgts.append(yb.cpu().numpy())
            all_prims.extend(prim_idx.tolist())
            total += xb.size(0)

    val_loss /= total
    scores   = np.vstack(all_scores)
    tgts     = np.vstack(all_tgts)
    prims    = np.array(all_prims, dtype=int)

    thresholds = np.full(NUM_CLASSES, 0.5, dtype=np.float32)
    for i in range(NUM_CLASSES):
        y_true = tgts[:, i]
        if 0 < y_true.sum() < len(y_true):
            prec, rec, th = precision_recall_curve(y_true, scores[:, i])
            f1_vals = 2 * prec * rec / (prec + rec + 1e-8)
            best    = np.nanargmax(f1_vals[:-1])
            thresholds[i] = th[best]

    preds     = (scores >= thresholds).astype(int)
    micro_f1  = f1_score(tgts, preds, average="micro", zero_division=0)
    micro_ap  = average_precision_score(tgts, scores, average="micro")
    top1      = scores.argmax(axis=1)
    primary_acc = (top1 == prims).mean()

    if micro_f1 > best_f1:
        best_f1, best_ap = micro_f1, micro_ap
        torch.save(model.state_dict(), BEST_CKPT)
        mlflow.log_artifact(BEST_CKPT, artifact_path="model")

    mlflow.log_metrics({
        "train_loss":   train_loss,
        "val_loss":     val_loss,
        "micro_f1":     micro_f1,
        "micro_ap":     micro_ap,
        "primary_acc":  primary_acc
    }, step=epoch)

    print(f"→ Epoch {epoch}/{EPOCHS}  F1={micro_f1:.4f}  AP={micro_ap:.4f}  PrimAcc={primary_acc:.4f}")

mlflow.log_metric("best_micro_f1", best_f1)
mlflow.log_metric("best_micro_ap", best_ap)

LOCAL_MODEL_DIR = "Panns_Emb_MLP_model_retrain"
mlflow.pytorch.save_model(model, LOCAL_MODEL_DIR)
mlflow.log_artifacts(LOCAL_MODEL_DIR, artifact_path="panns_emb_mlp_model_retrain")

mlflow.end_run()
