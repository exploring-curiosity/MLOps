import os
import sys
import subprocess
import argparse
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
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import shutil
import argparse

import timm
from peft import get_peft_model, LoraConfig
from torchvision.models import resnet50


# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description="Train MetaMLP with configurable epochs and frozen base models")
parser.add_argument("--epochs",        "-e", type=int,   default=5,   help="number of training epochs")
parser.add_argument("--batch_size",          type=int,   default=64,   help="Batch size")
parser.add_argument("--lr",                  type=float, default=1e-4, help="Learning rate")
parser.add_argument("--weight_decay",        type=float, default=1e-4, help="Weight decay")
parser.add_argument("--emb_model_name",      type=str,   default="PannsMLP_PrimaryLabel", help="MLflow registry name for Embedding MLP")
parser.add_argument("--raw_model_name",      type=str,   default="RawAudioCNN",          help="MLflow registry name for RawAudioCNN")
parser.add_argument("--res_model_name",      type=str,   default="ResNet50_MelAug",      help="MLflow registry name for ResNet50")
parser.add_argument("--eff_model_name",      type=str,   default="EfficientNetB3_LoRA",  help="MLflow registry name for EfficientNetB3")
args = parser.parse_args()


# ---------------------- Remaining Setup ----------------------
birdclef_base_dir = os.getenv("BIRDCLEF_BASE_DIR", "/mnt/data")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE     = args.batch_size
LR             = args.lr
WEIGHT_DECAY   = args.weight_decay
EPOCHS         = args.epochs
HIDDEN_DIMS    = [1024, 512]
DROPOUT        = 0.3
FOCAL_GAMMA    = 2.0
BEST_CKPT      = "best_meta_mlp.pth"
THRESHOLD_INIT = 0.5


TAXONOMY_CSV    = os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "taxonomy.csv")
TRAIN_MANIFEST  = os.path.join(birdclef_base_dir, "Features", "manifest_train.csv")
TEST_MANIFEST   = os.path.join(birdclef_base_dir, "Features", "manifest_test.csv")
TRAIN_META      = os.path.join(birdclef_base_dir, "Data", "birdclef-2025", "train.csv")
FEATURE_BASE    = os.path.join(birdclef_base_dir, "Features")

# ─── MLflow & SYSTEM METRICS ─────────────────────────────────────────────────
mlflow.set_experiment("MetaMLP_Supervisor")
if mlflow.active_run():
    mlflow.end_run()
run = mlflow.start_run(log_system_metrics=True)

print(f"MLFLOW_RUN_ID={run.info.run_id}")

gpu_info = next(
    (subprocess.run(cmd, capture_output=True, text=True).stdout
     for cmd in ["nvidia-smi","rocm-smi"]
     if subprocess.run(f"command -v {cmd}", shell=True,
                       capture_output=True).returncode==0),
    "No GPU found."
)
mlflow.log_text(gpu_info, "gpu-info.txt")

tax_df   = pd.read_csv(TAXONOMY_CSV)
CLASSES  = sorted(tax_df["primary_label"].astype(str).tolist())
NUM_CLS  = len(CLASSES)
IDX_MAP  = {c:i for i,c in enumerate(CLASSES)}


mlflow.log_params({
    "batch_size":   BATCH_SIZE,
    "lr":           LR,
    "weight_decay": WEIGHT_DECAY,
    "epochs":       EPOCHS,
    "hidden_dims":  HIDDEN_DIMS,
    "dropout":      DROPOUT,
    "focal_gamma":  FOCAL_GAMMA,
    "emb_model":    args.emb_model_name,
    "raw_model":    args.raw_model_name,
    "res_model":    args.res_model_name,
    "eff_model":    args.eff_model_name
})

class EnsembleDataset(Dataset):
    def __init__(self, manifest, meta_csv, base):
        m = pd.read_csv(manifest)
        meta = pd.read_csv(meta_csv, usecols=["filename","secondary_labels"])
        meta["rid"]  = meta.filename.str.replace(r"\.ogg$","",regex=True)
        meta["secs"] = meta.secondary_labels.fillna("").str.split()
        sec_map = dict(zip(meta.rid, meta.secs))

        self.rows = []
        for _,r in m.iterrows():
            rid  = r.chunk_id.split("_chk")[0]
            labs = [r.primary_label] + sec_map.get(rid,[])
            labs = [l for l in labs if l in IDX_MAP]
            prim = IDX_MAP[r.primary_label]

            emb_p = os.path.join(base,"embeddings", r.emb_path.lstrip(os.sep))
            ma_p  = os.path.join(base,"mel_aug",    r.mel_aug_path.lstrip(os.sep))
            m_p   = os.path.join(base,"mel",        r.mel_path.lstrip(os.sep))
            wav_p = os.path.join(base,"denoised",   r.audio_path.lstrip(os.sep))

            yvec = np.zeros(NUM_CLS, np.float32)
            for l in labs:
                yvec[IDX_MAP[l]] = 1.0

            self.rows.append((emb_p, ma_p, m_p, wav_p, yvec, prim))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        emb_p, ma_p, m_p, wav_p, yvec, prim = self.rows[i]

        # 1) Embedding
        emb_arr = np.load(emb_p)["embedding"].mean(axis=0).astype(np.float32)
        emb = torch.from_numpy(emb_arr)

        # 2) Mel‑aug
        ma_arr = np.load(ma_p)["mel"].astype(np.float32)
        ma = torch.from_numpy(ma_arr).unsqueeze(0)  # [1, n_mels, n_frames]

        # 3) Clean mel
        m_arr = np.load(m_p)["mel"].astype(np.float32)
        m = torch.from_numpy(m_arr).unsqueeze(0)    # [1, n_mels, n_frames]

        # 4) Raw waveform
        wav, _ = torchaudio.load(wav_p)   # [1, samples]
        wav = wav.squeeze(0)              # [samples]
        T   = 32000 * 10
        if wav.size(0) < T:
            wav = F.pad(wav, (0, T - wav.size(0)))
        else:
            wav = wav[:T]
        wav = (wav - wav.mean()) / wav.std().clamp_min(1e-6)  # normalize

        # 5) Label vector
        y = torch.from_numpy(yvec)

        return emb, ma, m, wav, y, prim


def ensemble_collate_fn(batch):
    """
    batch: list of tuples (emb, mel_aug, mel, wav, y, prim)
     - emb: [emb_dim]
     - mel_aug, mel: [1, n_mels, n_frames]
     - wav: [T]
     - y: [num_classes]
     - prim: int
    """
    embs, mas, ms, wavs, ys, prims = zip(*batch)

    embs = torch.stack(embs, dim=0)   # [B, emb_dim]
    mas  = torch.stack(mas,  dim=0)   # [B, 1, n_mels, n_frames]
    ms   = torch.stack(ms,   dim=0)   # [B, 1, n_mels, n_frames]
    # wavs is a list of [T], stack into [B, T]
    wavs = torch.stack(wavs, dim=0)   # [B, T]
    ys   = torch.stack(ys,   dim=0)   # [B, num_classes]
    prims = torch.tensor(prims, dtype=torch.long)  # [B]

    return embs, mas, ms, wavs, ys, prims


train_ds = EnsembleDataset(TRAIN_MANIFEST, TRAIN_META, FEATURE_BASE)
test_ds  = EnsembleDataset(TEST_MANIFEST,  TRAIN_META, FEATURE_BASE)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True,
    collate_fn=ensemble_collate_fn
)
test_loader  = DataLoader(
    test_ds,  batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True,
    collate_fn=ensemble_collate_fn
)


def load_and_freeze_model(model_name):
    client   = MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        print(f"No registered model found for '{model_name}'. Exiting.")
        sys.exit(1)
    latest = max(versions, key=lambda mv: int(mv.version))
    uri    = f"models:/{model_name}/{latest.version}"
    print(f"Loading model from registry URI: {uri}")
    model = mlflow.pytorch.load_model(uri).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

# ─── LOAD & FREEZE BASE MODELS ──────────────────────────────────────────────────
emb_model = load_and_freeze_model(args.emb_model_name)
raw_model = load_and_freeze_model(args.raw_model_name)
res_model = load_and_freeze_model(args.res_model_name)
eff_model = load_and_freeze_model(args.eff_model_name)

class MetaMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout):
        super().__init__()
        layers, dims = [], [in_dim]+hidden_dims
        for i in range(len(hidden_dims)):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        layers.append(nn.Linear(dims[-1], NUM_CLS))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

meta_model = MetaMLP(NUM_CLS*4, HIDDEN_DIMS, DROPOUT).to(DEVICE)
mlflow.log_param("meta_in_dim", NUM_CLS*4)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__(); self.gamma=gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        return ((1-p_t)**self.gamma * bce).mean()

criterion = FocalLoss(FOCAL_GAMMA)
optimizer = torch.optim.AdamW(meta_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS, pct_start=0.1, div_factor=10
)
scaler = GradScaler()

best_f1, best_ap, best_acc = 0.0, 0.0, 0.0
thresholds = np.full(NUM_CLS, THRESHOLD_INIT, dtype=np.float32)

for epoch in range(1, EPOCHS+1):
    # — Train —
    meta_model.train()
    train_bar = tqdm(train_loader, desc=f"[{epoch}/{EPOCHS}] Train", unit="batch")
    run_loss = total = 0
    for emb, ma, m, wav, yb, prim in train_bar:
        emb, ma, m, wav, yb = [t.to(DEVICE) for t in (emb, ma, m, wav, yb)]

        # get frozen‑base outputs
        with torch.no_grad():
            p1 = torch.sigmoid(emb_model(emb))
            p2 = torch.sigmoid(res_model(ma))
            p3 = torch.sigmoid(eff_model(m))
            p4 = torch.sigmoid(raw_model(wav))

        feat = torch.cat([p1, p2, p3, p4], dim=1)

        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            logits = meta_model(feat)
            loss   = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        bs = emb.size(0)
        run_loss += loss.item() * bs
        total    += bs

        # <-- update tqdm postfix with latest running avg loss -->
        train_bar.set_postfix({"loss": f"{run_loss/total:.4f}"})

    train_loss = run_loss / total

    # — Eval —
    meta_model.eval()
    eval_bar = tqdm(test_loader, desc=f"[{epoch}/{EPOCHS}] Eval ", unit="batch")
    val_loss = total = 0
    all_scores, all_tgts, all_prims = [], [], []

    with torch.no_grad():
        for emb, ma, m, wav, yb, prim in eval_bar:
            emb, ma, m, wav, yb = [t.to(DEVICE) for t in (emb, ma, m, wav, yb)]
            p1 = torch.sigmoid(emb_model(emb))
            p2 = torch.sigmoid(res_model(ma))
            p3 = torch.sigmoid(eff_model(m))
            p4 = torch.sigmoid(raw_model(wav))
            feat   = torch.cat([p1, p2, p3, p4], dim=1)

            logits = meta_model(feat)
            loss   = criterion(logits, yb)

            bs = emb.size(0)
            val_loss += loss.item() * bs
            total    += bs

            scores = logits.sigmoid().cpu().numpy()
            all_scores.append(scores)
            all_tgts.append(yb.cpu().numpy())
            all_prims.extend(prim.tolist())

            # <-- update tqdm postfix with latest running eval loss -->
            eval_bar.set_postfix({"loss": f"{val_loss/total:.4f}"})

    val_loss = val_loss / total

    scores = np.vstack(all_scores)
    tgts   = np.vstack(all_tgts)
    prims  = np.array(all_prims, dtype=int)

    # calibrate thresholds
    for i in range(NUM_CLS):
        y_true = tgts[:,i]
        if 0<y_true.sum()<len(y_true):
            prec,rec,th = precision_recall_curve(y_true, scores[:,i])
            f1s = 2*prec*rec/(prec+rec+1e-8)
            thresholds[i] = th[np.nanargmax(f1s[:-1])]

    preds     = (scores>=thresholds).astype(int)
    micro_f1  = f1_score(tgts, preds, average="micro", zero_division=0)
    micro_ap  = average_precision_score(tgts, scores, average="micro")
    primary_acc = (scores.argmax(axis=1)==prims).mean()

    # checkpoint best
    if micro_f1>best_f1:
        best_f1,best_ap,best_acc = micro_f1,micro_ap,primary_acc
        torch.save(meta_model.state_dict(), BEST_CKPT)
        mlflow.log_artifact(BEST_CKPT, artifact_path="model")

    mlflow.log_metrics({
        "train_loss":  train_loss,
        "val_loss":    val_loss,
        "micro_f1":    micro_f1,
        "micro_ap":    micro_ap,
        "prim_acc":    primary_acc
    }, step=epoch)

    print(f"→ Epoch {epoch}/{EPOCHS}  "
          f"F1={micro_f1:.4f}  AP={micro_ap:.4f}  PrimAcc={primary_acc:.4f}")
    
mlflow.log_metric("best_micro_f1", best_f1)
mlflow.log_metric("best_micro_ap", best_ap)
mlflow.log_metric("best_primary_acc", best_acc)

LOCAL_MODEL_DIR = "Meta_MLP_model"
if os.path.isdir(LOCAL_MODEL_DIR):
    shutil.rmtree(LOCAL_MODEL_DIR)
mlflow.pytorch.save_model(meta_model, LOCAL_MODEL_DIR)
mlflow.log_artifacts(LOCAL_MODEL_DIR, artifact_path="meta_mlp_model")

mlflow.end_run()