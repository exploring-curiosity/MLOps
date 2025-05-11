import os
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
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve

import argparse
import shutil

parser = argparse.ArgumentParser(description="Train RawCNN with configurable epochs")
parser.add_argument(
    "--epochs", "-e",
    type=int,
    default=20,
    help="number of training epochs"
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

mlflow.set_experiment("RawAudioCNN")
if mlflow.active_run():
    mlflow.end_run()
run = mlflow.start_run(log_system_metrics=True)

print(f"MLFLOW_RUN_ID={run.info.run_id}")

# log GPU/CPU info
gpu_info = next(
    (subprocess.run(cmd, capture_output=True, text=True).stdout 
     for cmd in ["nvidia-smi","rocm-smi"]
     if subprocess.run(f"command -v {cmd}", shell=True,
                       capture_output=True).returncode==0),
    "No GPU found."
)
mlflow.log_text(gpu_info, "gpu-info.txt")



BATCH_SIZE     = args.batch_size
LR             = args.lr
WEIGHT_DECAY   = args.weight_decay 
EPOCHS       = args.epochs
SAVE_CKPT    = False
BEST_CKPT    = "best_rawcnn.pth"

TAXONOMY_CSV    = os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "taxonomy.csv")
TRAIN_MAN       = os.path.join(birdclef_base_dir, "Features", "manifest_train.csv")
TEST_MAN        = os.path.join(birdclef_base_dir, "Features", "manifest_test.csv")
TRAIN_META      = os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "train.csv")
FEATURE_BASE    = os.path.join(birdclef_base_dir, "Features")

tax_df  = pd.read_csv(TAXONOMY_CSV)
CLASSES = sorted(tax_df["primary_label"].astype(str).tolist())
NUM_CLS = len(CLASSES)

mlflow.log_params({
    "model":        "RawAudioCNN",
    "batch_size":   BATCH_SIZE,
    "lr":           LR,
    "weight_decay": WEIGHT_DECAY,
    "epochs":       EPOCHS,
    "input":        "raw_waveform",
})

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
        wav, _ = torchaudio.load(path)    # (1, samples)
        wav     = wav.squeeze(0)          # (samples,)
        if wav.size(0) < self.wav_len:
            wav = F.pad(wav, (0, self.wav_len-wav.size(0)))
        else:
            wav = wav[:self.wav_len]
        # normalize per‐sample
        wav = (wav - wav.mean()) / wav.std().clamp_min(1e-6)
        # label vector
        y = torch.zeros(NUM_CLS, dtype=torch.float32)
        for l in labs:
            y[self.idx_map[l]] = 1.0
        return wav, y, prim
    
train_ds = RawAudioDataset(TRAIN_MAN, TRAIN_META, FEATURE_BASE, CLASSES)
test_ds  = RawAudioDataset(TEST_MAN,  TRAIN_META, FEATURE_BASE, CLASSES)

train_loader = DataLoader(train_ds,
    batch_size=BATCH_SIZE, shuffle=True,  num_workers=16, pin_memory=True)
test_loader  = DataLoader(test_ds,
    batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

class RawAudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # initial downsample
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7)
        self.bn1   = nn.BatchNorm1d(16)
        self.pool  = nn.MaxPool1d(4)
        # deeper layers
        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7)
        self.bn2   = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7)
        self.bn3   = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64,128,kernel_size=15, stride=2,padding=7)
        self.bn4   = nn.BatchNorm1d(128)
        # global pooling & head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc          = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, T] → [B,1,T]
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x).squeeze(-1)  # [B,128]
        return self.fc(x)
    

model     = RawAudioCNN(NUM_CLS).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS, pct_start=0.1,
    div_factor=10, final_div_factor=100
)
scaler    = GradScaler()


best_f1 = best_ap = best_acc = 0.0
thresholds = np.full(NUM_CLS, 0.5, dtype=np.float32)

for epoch in range(1, EPOCHS+1):
    # — Train —
    model.train()
    run_loss = total = 0
    train_bar = tqdm(train_loader, desc=f"[{epoch}/{EPOCHS}] Train", unit="batch")
    for wav, yb, prim in train_bar:
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

        bs = wav.size(0)
        run_loss += loss.item()*bs
        total    += bs
        train_bar.set_postfix({"loss": f"{run_loss/total:.4f}"})

    train_loss = run_loss/total

    # — Eval —
    model.eval()
    val_loss = total = 0
    all_scores, all_tgts, all_prims = [], [], []
    eval_bar = tqdm(test_loader, desc=f"[{epoch}/{EPOCHS}] Test ", unit="batch")
    with torch.no_grad():
        for wav, yb, prim in eval_bar:
            wav, yb = wav.to(DEVICE), yb.to(DEVICE)
            with autocast(device_type="cuda"):
                logits = model(wav)
                loss   = criterion(logits, yb)
            bs = wav.size(0)
            val_loss += loss.item()*bs
            total    += bs
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_tgts.append(yb.cpu().numpy())
            all_prims.append(prim.numpy())
            eval_bar.set_postfix({"loss": f"{val_loss/total:.4f}"})

    val_loss = val_loss/total
    scores   = np.vstack(all_scores)
    tgts     = np.vstack(all_tgts)
    prims    = np.concatenate(all_prims, axis=0)

    # threshold calibration
    for i in range(NUM_CLS):
        y_true, y_score = tgts[:,i], scores[:,i]
        if 0<y_true.sum()<len(y_true):
            prec, rec, th = precision_recall_curve(y_true, y_score)
            f1s = 2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-8)
            if f1s.size>0:
                thresholds[i] = th[np.nanargmax(f1s)]

    preds    = (scores>=thresholds).astype(int)
    micro_f1 = f1_score(tgts, preds, average="micro", zero_division=0)
    micro_ap = average_precision_score(tgts, scores, average="micro")
    prim_acc = (scores.argmax(axis=1)==prims).mean()

    # checkpoint best
    if micro_f1>best_f1:
        best_f1, best_ap, best_acc = micro_f1, micro_ap, prim_acc
        torch.save(model.state_dict(), BEST_CKPT)
        mlflow.log_artifact(BEST_CKPT, artifact_path="model")

    # MLflow logging
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "micro_f1":   micro_f1,
        "micro_ap":   micro_ap,
        "prim_acc":   prim_acc
    }, step=epoch)

    print(f"→ Epoch {epoch}/{EPOCHS}  "
          f"F1={micro_f1:.4f}  AP={micro_ap:.4f}  PrimAcc={prim_acc:.4f}")
    
mlflow.log_metric("best_micro_f1", best_f1)
mlflow.log_metric("best_micro_ap", best_ap)
mlflow.log_metric("best_prim_acc", best_acc)

LOCAL_MODEL_DIR = "RawCNN_model"
if os.path.isdir(LOCAL_MODEL_DIR):
    shutil.rmtree(LOCAL_MODEL_DIR)
mlflow.pytorch.save_model(model, LOCAL_MODEL_DIR)
mlflow.log_artifacts(LOCAL_MODEL_DIR, artifact_path="RawCNN_model")

mlflow.end_run()