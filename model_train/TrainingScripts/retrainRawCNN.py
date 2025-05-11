import os
import sys
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
import shutil

import argparse

parser = argparse.ArgumentParser(description="Retrain RawCNN from MLflow Registry using VAL manifest")
parser.add_argument("--raw_model_name",        type=str,   default="RawAudioCNN",          help="Registry name for RawAudioCNN")

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
    default=3e-3,
    help="Learning rate"
)
parser.add_argument(
    "--weight_decay",
    type=int,
    default=1e-2,
    help="Weight decay"
)
args = parser.parse_args()

birdclef_base_dir = os.getenv("BIRDCLEF_BASE_DIR", "/mnt/data")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# ---------------------- MLflow Setup ----------------------
mlflow.set_experiment("RawAudioCNN_Retrain")
if mlflow.active_run():
    mlflow.end_run()
run = mlflow.start_run(log_system_metrics=True)
print(f"MLFLOW_RUN_ID={run.info.run_id}")

# ---------------------- Load base model from Registry ----------------------
model_name = args.raw_model_name
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
EPOCHS       = args.epochs
SAVE_CKPT    = False
BEST_CKPT    = "best_rawcnn_retrain.pth"

mlflow.log_params({
    "batch_size":   BATCH_SIZE,
    "lr":           LR,
    "weight_decay": WEIGHT_DECAY,
    "epochs":       EPOCHS,
    "model":        "RawAudioCNN"
})

# ---------------------- Dataset Class ----------------------
class RawAudioDataset(Dataset):
    def __init__(self, manifest_csv, meta_csv, base, classes,
                 sr=32000, dur=10.0):
        m = pd.read_csv(manifest_csv)
        m["path"] = (
            m["audio_path"].astype(str)
             .str.lstrip(os.sep)
             .apply(lambda p: os.path.join(base, "denoised", p))
        )
        meta = pd.read_csv(meta_csv, usecols=["filename","secondary_labels"])
        meta["rid"]  = meta.filename.str.replace(r"\.ogg$","",regex=True)
        meta["secs"] = meta.secondary_labels.fillna("").str.split()
        sec_map      = dict(zip(meta.rid, meta.secs))

        self.rows    = []
        self.idx_map = {c:i for i,c in enumerate(classes)}
        self.wav_len = int(sr * dur)
        for _, r in m.iterrows():
            rid  = r.chunk_id.split("_chk")[0]
            labs = [r.primary_label] + sec_map.get(rid, [])
            labs = [l for l in labs if l in self.idx_map]
            prim = self.idx_map[r.primary_label]
            self.rows.append((r.path, labs, prim))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, labs, prim = self.rows[i]
        wav, _ = torchaudio.load(path)
        wav     = wav.squeeze(0)
        if wav.size(0) < self.wav_len:
            wav = F.pad(wav, (0, self.wav_len-wav.size(0)))
        else:
            wav = wav[:self.wav_len]
        wav = (wav - wav.mean()) / wav.std().clamp_min(1e-6)
        y = torch.zeros(len(self.idx_map), dtype=torch.float32)
        for l in labs:
            y[self.idx_map[l]] = 1.0
        return wav, y, prim

train_ds = RawAudioDataset(
    os.path.join(birdclef_base_dir, "Features", "manifest_val.csv"),
    os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "train.csv"),
    os.path.join(birdclef_base_dir, "Features"),
    sorted(pd.read_csv(os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "taxonomy.csv"))["primary_label"].astype(str).tolist())
)
test_ds = RawAudioDataset(
    os.path.join(birdclef_base_dir, "Features", "manifest_test.csv"),
    os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "train.csv"),
    os.path.join(birdclef_base_dir, "Features"),
    train_ds.idx_map.keys()
)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True
)

# ---------------------- Loss, Optimizer, Scheduler ----------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS, pct_start=0.1,
    div_factor=10, final_div_factor=100
)
scaler    = GradScaler()

# ---------------------- Retraining Loop ----------------------
best_f1 = best_ap = best_acc = 0.0
thresholds = np.full(len(train_ds.idx_map), 0.5, dtype=np.float32)

for epoch in range(1, EPOCHS+1):
    model.train()
    run_loss = total = 0
    for wav, yb, _ in tqdm(train_loader, desc=f"[{epoch}/{EPOCHS}] Train"):
        wav, yb = wav.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            logits = model(wav)
            loss   = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        run_loss += loss.item() * wav.size(0)
        total    += wav.size(0)
    mlflow.log_metric("train_loss", run_loss/total, step=epoch)

    model.eval()
    val_loss = total = 0
    all_scores, all_tgts, all_prims = [], [], []
    with torch.no_grad():
        for wav, yb, prim in tqdm(test_loader, desc=f"[{epoch}/{EPOCHS}] Test"):
            wav, yb = wav.to(DEVICE), yb.to(DEVICE)
            with autocast(device_type="cuda"):
                logits = model(wav)
                loss   = criterion(logits, yb)
            val_loss += loss.item() * wav.size(0)
            total    += wav.size(0)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_tgts.append(yb.cpu().numpy())
            all_prims.append(prim.numpy())
    mlflow.log_metric("val_loss", val_loss/total, step=epoch)

    scores = np.vstack(all_scores)
    tgts   = np.vstack(all_tgts)
    prims  = np.concatenate(all_prims)

    for i in range(scores.shape[1]):
        y_true, y_score = tgts[:,i], scores[:,i]
        if 0 < y_true.sum() < len(y_true):
            prec, rec, th = precision_recall_curve(y_true, y_score)
            f1s = 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-8)
            if f1s.size>0:
                thresholds[i] = th[np.nanargmax(f1s)]

    preds    = (scores>=thresholds).astype(int)
    micro_f1 = f1_score(tgts, preds, average="micro", zero_division=0)
    micro_ap = average_precision_score(tgts, scores, average="micro")
    prim_acc = (scores.argmax(axis=1)==prims).mean()

    mlflow.log_metrics({
        "micro_f1":  micro_f1,
        "micro_ap":  micro_ap,
        "prim_acc":  prim_acc
    }, step=epoch)

    if micro_f1 > best_f1:
        best_f1, best_ap, best_acc = micro_f1, micro_ap, prim_acc
        torch.save(model.state_dict(), BEST_CKPT)
        mlflow.log_artifact(BEST_CKPT, artifact_path="model")

    print(f"→ Epoch {epoch}/{EPOCHS}  F1={micro_f1:.4f}  AP={micro_ap:.4f}  PrimAcc={prim_acc:.4f}")

mlflow.log_metric("best_micro_f1", best_f1)
mlflow.log_metric("best_micro_ap", best_ap)
mlflow.log_metric("best_prim_acc", best_acc)

LOCAL_MODEL_DIR = "RawAudioCNN_model_retrain"
if os.path.isdir(LOCAL_MODEL_DIR):
    shutil.rmtree(LOCAL_MODEL_DIR)
mlflow.pytorch.save_model(model, LOCAL_MODEL_DIR)
mlflow.log_artifacts(LOCAL_MODEL_DIR, artifact_path="RawAudioCNN_model_retrain")

mlflow.end_run()
