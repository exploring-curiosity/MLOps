import os
import sys
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import argparse

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description="Retrain ResNet50_MelAug from MLflow Registry using VAL manifest")
parser.add_argument("--res_model_name",        type=str,   default="ResNet50_MelAug",      help="Registry name for ResNet50")
parser.add_argument(
    "--epochs", "-e",
    type=int,
    default=20,
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
    default=1e-4,
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# ---------------------- MLflow Setup ----------------------
mlflow.set_experiment("ResNet50_MelAug_Retrain")
if mlflow.active_run():
    mlflow.end_run()
run = mlflow.start_run(log_system_metrics=True)
print(f"MLFLOW_RUN_ID={run.info.run_id}")

# ---------------------- Load base model from Registry ----------------------
model_name = args.res_model_name
client = MlflowClient()
all_versions = client.search_model_versions(f"name = '{model_name}'")
if not all_versions:
    print(f"No registered model '{model_name}' found. Exiting.")
    sys.exit(0)
latest = max(all_versions, key=lambda mv: int(mv.version))
model_uri = f"models:/{model_name}/{latest.version}"
print(f"Loading model from registry URI: {model_uri}")
model = mlflow.pytorch.load_model(model_uri).to(DEVICE)
mlflow.log_param("base_model_name", model_name)
mlflow.log_param("base_model_version", latest.version)

# log GPU/CPU info
gpu_info = next(
    (subprocess.run(cmd, capture_output=True, text=True).stdout
     for cmd in ["nvidia-smi","rocm-smi"]
     if subprocess.run(f"command -v {cmd}", shell=True, capture_output=True).returncode==0),
    "No GPU found."
)
mlflow.log_text(gpu_info, "gpu-info.txt")

# ---------------------- Hyperparameters ----------------------
BATCH_SIZE     = args.batch_size
LR             = args.lr
WEIGHT_DECAY   = args.weight_decay 
EPOCHS         = args.epochs
SAVE_EPOCH_CK  = False
BEST_CKPT      = "best_resnet50_retrain.pth"

mlflow.log_params({
    "batch_size":    BATCH_SIZE,
    "lr":            LR,
    "weight_decay":  WEIGHT_DECAY,
    "epochs":        EPOCHS,
    "model":         "resnet50_scratch"
})

# ---------------------- Dataset Class ----------------------
class MelAugDataset(Dataset):
    def __init__(self, manifest_csv, meta_csv, feature_base, classes, key="mel"):
        m_df = pd.read_csv(manifest_csv)
        m_df["mel_path"] = (
            m_df["mel_aug_path"].astype(str)
                .str.lstrip(os.sep)
                .apply(lambda p: os.path.join(feature_base, "mel_aug", p))
        )
        meta = pd.read_csv(meta_csv, usecols=["filename","secondary_labels"])
        meta["rid"]     = meta.filename.str.replace(r"\.ogg$","",regex=True)
        meta["sec_list"]= meta.secondary_labels.fillna("").str.split()
        sec_map = dict(zip(meta.rid, meta.sec_list))

        self.rows   = []
        self.idx_map = {c:i for i,c in enumerate(classes)}
        self.num_cls = len(classes)
        self.key     = key

        for _, row in tqdm(m_df.iterrows(), total=len(m_df), desc="Building dataset"):
            rid      = row.chunk_id.split("_chk")[0]
            labs     = [row.primary_label] + sec_map.get(rid, [])
            labs     = [l for l in labs if l in self.idx_map]
            prim_idx = self.idx_map[row.primary_label]
            self.rows.append((row.mel_path, labs, prim_idx))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, labs, prim_idx = self.rows[i]
        npz  = np.load(path)
        arr  = npz[self.key]
        x    = torch.from_numpy(arr).unsqueeze(0).float()
        y    = torch.zeros(self.num_cls, dtype=torch.float32)
        for l in labs:
            y[self.idx_map[l]] = 1.0
        return x, y, prim_idx

# ---------------------- Data Loading (VAL instead of TRAIN) ----------------------
TAXONOMY_CSV    = os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "taxonomy.csv")
VAL_MANIFEST    = os.path.join(birdclef_base_dir, "Features", "manifest_val.csv")
TEST_MANIFEST   = os.path.join(birdclef_base_dir, "Features", "manifest_test.csv")
TRAIN_CSV       = os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "train.csv")
FEATURE_BASE    = os.path.join(birdclef_base_dir, "Features")

tax_df      = pd.read_csv(TAXONOMY_CSV)
CLASSES     = sorted(tax_df["primary_label"].astype(str).tolist())
NUM_CLASSES = len(CLASSES)

train_ds = MelAugDataset(VAL_MANIFEST, TRAIN_CSV, FEATURE_BASE, CLASSES)
test_ds  = MelAugDataset(TEST_MANIFEST, TRAIN_CSV, FEATURE_BASE, CLASSES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ---------------------- Loss, Optimizer, Scheduler ----------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(
    optimizer,
    max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,
    div_factor=10
)
scaler = GradScaler()

# ---------------------- Retraining Loop ----------------------
best_f1, best_ap, best_acc = 0.0, 0.0, 0.0
thresholds = np.full(NUM_CLASSES, 0.5, dtype=np.float32)

for epoch in range(1, EPOCHS+1):
    # — Train —
    model.train()
    run_loss, total = 0.0, 0
    for xb, yb, _ in tqdm(train_loader, desc=f"[{epoch}/{EPOCHS}] Train", unit="batch"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            logits = model(xb)
            loss   = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = xb.size(0)
        run_loss += loss.item()*bs
        total    += bs
    mlflow.log_metric("train_loss", run_loss/total, step=epoch)

    # — Eval —
    model.eval()
    all_scores, all_tgts, all_prims = [], [], []
    val_loss, total = 0.0, 0
    with torch.no_grad():
        for xb, yb, prim_idx in tqdm(test_loader, desc=f"[{epoch}/{EPOCHS}] Eval ", unit="batch"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with autocast(device_type="cuda"):
                logits = model(xb)
                val_loss += criterion(logits, yb).item()*xb.size(0)
                scores   = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_tgts.append(yb.cpu().numpy())
            all_prims.extend(prim_idx.tolist())
            total += xb.size(0)
    mlflow.log_metric("val_loss", val_loss/total, step=epoch)

    scores = np.vstack(all_scores)
    tgts   = np.vstack(all_tgts)
    prims  = np.array(all_prims, dtype=int)

    for i in range(NUM_CLASSES):
        y_true = tgts[:, i]
        if 0 < y_true.sum() < len(y_true):
            prec, rec, th = precision_recall_curve(y_true, scores[:, i])
            f1_vals = 2*prec*rec/(prec+rec+1e-8)
            best    = np.nanargmax(f1_vals[:-1])
            thresholds[i] = th[best]

    preds      = (scores >= thresholds).astype(int)
    micro_f1   = f1_score(tgts, preds, average="micro", zero_division=0)
    micro_ap   = average_precision_score(tgts, scores, average="micro")
    top1       = scores.argmax(axis=1)
    primary_acc= (top1 == prims).mean()

    mlflow.log_metrics({
        "micro_f1": micro_f1,
        "micro_ap": micro_ap,
        "primary_acc": primary_acc
    }, step=epoch)

    if micro_f1 > best_f1:
        best_f1, best_ap, best_acc = micro_f1, micro_ap, primary_acc
        torch.save(model.state_dict(), BEST_CKPT)
        mlflow.log_artifact(BEST_CKPT, artifact_path="model")

    print(f"→ Epoch {epoch}/{EPOCHS}  F1={micro_f1:.4f}  AP={micro_ap:.4f}  PrimAcc={primary_acc:.4f}")

mlflow.log_metric("best_micro_f1", best_f1)
mlflow.log_metric("best_micro_ap", best_ap)
mlflow.log_metric("best_primary_acc", best_acc)

LOCAL_MODEL_DIR = "ResNet50_model_retrain"
mlflow.pytorch.save_model(model, LOCAL_MODEL_DIR)
mlflow.log_artifacts(LOCAL_MODEL_DIR, artifact_path="resnet50_model_retrain")

mlflow.end_run()
