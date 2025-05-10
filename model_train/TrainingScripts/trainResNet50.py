import os
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

from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
from torch.amp import autocast, GradScaler
import argparse

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description="Train ResNet50 with configurable epochs")
parser.add_argument(
    "--epochs", "-e",
    type=int,
    default=20,
    help="number of training epochs"
)
args = parser.parse_args()

birdclef_base_dir = os.getenv("BIRDCLEF_BASE_DIR", "/mnt/data")


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

mlflow.set_experiment("ResNet50_MelAug")
if mlflow.active_run():
    mlflow.end_run()
mlflow.start_run(log_system_metrics=True)

# log GPU/CPU info once
gpu_info = next(
    (subprocess.run(cmd, capture_output=True, text=True).stdout
        for cmd in ["nvidia-smi","rocm-smi"]
        if subprocess.run(f"command -v {cmd}", shell=True, capture_output=True).returncode == 0),
    "No GPU found."
)
mlflow.log_text(gpu_info, "gpu-info.txt")

BATCH_SIZE     = 64
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
EPOCHS         = args.epochs
SAVE_EPOCH_CK  = False
BEST_CKPT      = "best_resnet50.pt"

TAXONOMY_CSV    = os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "taxonomy.csv")
TRAIN_MANIFEST       = os.path.join(birdclef_base_dir, "Features", "manifest_train.csv")
TEST_MANIFEST        = os.path.join(birdclef_base_dir, "Features", "manifest_test.csv")
TRAIN_CSV      = os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "train.csv")
FEATURE_BASE    = os.path.join(birdclef_base_dir, "Features")

tax_df     = pd.read_csv(TAXONOMY_CSV)
CLASSES    = sorted(tax_df["primary_label"].astype(str).tolist())
NUM_CLASSES= len(CLASSES)

mlflow.log_params({
    "model":         "resnet50_scratch",
    "input":         "mel_aug",
    "num_classes":   NUM_CLASSES,
    "batch_size":    BATCH_SIZE,
    "lr":            LR,
    "weight_decay":  WEIGHT_DECAY,
    "epochs":        EPOCHS,
    "save_epoch_ck": SAVE_EPOCH_CK
})

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

        self.rows = []
        self.idx_map   = {c:i for i,c in enumerate(classes)}
        self.num_cls   = len(classes)
        self.key       = key

        for _, row in tqdm(m_df.iterrows(), total=len(m_df), desc="Building dataset"):
            rid  = row.chunk_id.split("_chk")[0]
            labs = [row.primary_label] + sec_map.get(rid, [])
            labs = [l for l in labs if l in self.idx_map]
            prim_idx = self.idx_map[row.primary_label]
            self.rows.append((row.mel_path, labs, prim_idx))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, labs, prim_idx = self.rows[i]
        npz  = np.load(path)
        arr  = npz[self.key]                   # [n_mels, n_frames]
        x    = torch.from_numpy(arr).unsqueeze(0).float()  # [1,n_mels,n_frames]
        y    = torch.zeros(self.num_cls, dtype=torch.float32)
        for l in labs:
            y[self.idx_map[l]] = 1.0
        return x, y, prim_idx


train_ds = MelAugDataset(TRAIN_MANIFEST, TRAIN_CSV, FEATURE_BASE, CLASSES)
test_ds  = MelAugDataset(TEST_MANIFEST,  TRAIN_CSV, FEATURE_BASE, CLASSES)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)


def get_resnet50_multilabel(num_classes):
    m = resnet50(weights=None)
    # adapt 1‐channel
    m.conv1 = nn.Conv2d(1, m.conv1.out_channels,
                        kernel_size=m.conv1.kernel_size,
                        stride=m.conv1.stride,
                        padding=m.conv1.padding,
                        bias=False)
    m.fc    = nn.Linear(m.fc.in_features, num_classes)
    return m

model     = get_resnet50_multilabel(NUM_CLASSES).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.1,
    div_factor=10
)
scaler = GradScaler()

best_f1, best_ap, best_acc = 0.0, 0.0, 0.0
thresholds = np.full(NUM_CLASSES, 0.5, dtype=np.float32)

for epoch in range(1, EPOCHS+1):
    # — Train —
    model.train()
    run_loss, total = 0.0, 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Train", unit="batch")
    for xb, yb, _ in train_bar:
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
        train_bar.set_postfix({"loss": f"{run_loss/total:.4f}"})
    train_loss = run_loss/total

    # — Eval —
    model.eval()
    all_scores, all_tgts, all_prims = [], [], []
    val_loss, total = 0.0, 0
    with torch.no_grad():
        for xb, yb, prim_idx in tqdm(test_loader, desc=f"Epoch {epoch}/{EPOCHS} Eval", unit="batch"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with autocast(device_type="cuda"):
                logits = model(xb)
                val_loss += criterion(logits, yb).item()*xb.size(0)
                scores   = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores)
            all_tgts.append(yb.cpu().numpy())
            all_prims.extend(prim_idx.tolist())
            total += xb.size(0)

    val_loss /= total
    scores = np.vstack(all_scores)
    tgts   = np.vstack(all_tgts)
    prims  = np.array(all_prims, dtype=int)

    # fast threshold calibration
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

    # checkpoint best
    if micro_f1 > best_f1:
        best_f1, best_ap, best_acc = micro_f1, micro_ap, primary_acc
        torch.save(model.state_dict(), BEST_CKPT)
        mlflow.log_artifact(BEST_CKPT, artifact_path="model")

    # log metrics
    mlflow.log_metrics({
        "train_loss":     train_loss,
        "val_loss":       val_loss,
        "micro_f1":       micro_f1,
        "micro_ap":       micro_ap,
        "primary_acc":    primary_acc
    }, step=epoch)

    print(f"→ Epoch {epoch}/{EPOCHS}  "
          f"F1={micro_f1:.4f}  AP={micro_ap:.4f}  PrimAcc={primary_acc:.4f}")


mlflow.log_metric("best_micro_f1", best_f1)
mlflow.log_metric("best_micro_ap", best_ap)
mlflow.log_metric("best_primary_acc", best_acc)

LOCAL_MODEL_DIR = "ResNet50_model"
mlflow.pytorch.save_model(model, LOCAL_MODEL_DIR)
mlflow.log_artifacts(LOCAL_MODEL_DIR, artifact_path="resnet50_model")

mlflow.end_run()