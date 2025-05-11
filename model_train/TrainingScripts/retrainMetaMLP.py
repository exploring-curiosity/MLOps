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

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description="Retrain MetaMLP ensemble head with frozen base models")
parser.add_argument("--epochs",           "-e", type=int,   default=20,                     help="number of retraining epochs")
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
parser.add_argument("--emb_model_name",        type=str,   default="PannsMLP_PrimaryLabel", help="Registry name for Embedding MLP")
parser.add_argument("--raw_model_name",        type=str,   default="RawAudioCNN",          help="Registry name for RawAudioCNN")
parser.add_argument("--res_model_name",        type=str,   default="ResNet50_MelAug",      help="Registry name for ResNet50")
parser.add_argument("--eff_model_name",        type=str,   default="EfficientNetB3_LoRA",  help="Registry name for EfficientNetB3")
parser.add_argument("--meta_model_name",       type=str,   default="MetaMLP_Supervisor",  help="Registry name for MetaMLP head")
args = parser.parse_args()

# ---------------------- Setup ----------------------
BIRDCLEF_BASE_DIR = os.getenv("BIRDCLEF_BASE_DIR", "/mnt/data")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ---------------------- Helper to load & freeze models ----------------------
def load_frozen_model(reg_name):
    client = MlflowClient()
    vers   = client.search_model_versions(f"name = '{reg_name}'")
    if not vers:
        print(f"No registered model '{reg_name}' found. Exiting."); sys.exit(0)
    latest = max(vers, key=lambda mv: int(mv.version))
    uri    = f"models:/{reg_name}/{latest.version}"
    print(f"Loading {reg_name} → {uri}")
    m = mlflow.pytorch.load_model(uri).to(DEVICE)
    m.eval()
    for p in m.parameters(): p.requires_grad = False
    return m

emb_model = load_frozen_model(args.emb_model_name)
raw_model = load_frozen_model(args.raw_model_name)
res_model = load_frozen_model(args.res_model_name)
eff_model = load_frozen_model(args.eff_model_name)

# ---------------------- MLflow Setup ----------------------
mlflow.set_experiment(f"{args.meta_model_name}_Retrain")
if mlflow.active_run(): mlflow.end_run()
run = mlflow.start_run(log_system_metrics=True)
print(f"MLFLOW_RUN_ID={run.info.run_id}")

# log base model parameters
mlflow.log_param("emb_model_name", args.emb_model_name)
mlflow.log_param("raw_model_name", args.raw_model_name)
mlflow.log_param("res_model_name", args.res_model_name)
mlflow.log_param("eff_model_name", args.eff_model_name)
# log meta-mlp model name
mlflow.log_param("meta_model_name", args.meta_model_name)

gpu_info = next(
    (subprocess.run(cmd, capture_output=True, text=True).stdout
     for cmd in ["nvidia-smi","rocm-smi"]
     if subprocess.run(f"command -v {cmd}", shell=True, capture_output=True).returncode==0),
    "No GPU found."
)
mlflow.log_text(gpu_info, "gpu-info.txt")

# ---------------------- Hyperparameters ----------------------
EPOCHS       = args.epochs
BATCH_SIZE     = args.batch_size
LR             = args.lr
WEIGHT_DECAY   = args.weight_decay 
HIDDEN_DIMS  = [1024, 512]
DROPOUT      = 0.3
FOCAL_GAMMA  = 2.0
BEST_CKPT    = "best_meta_mlp.pth"
TH_INIT      = 0.5

mlflow.log_params({
    "epochs":       EPOCHS,
    "batch_size":   BATCH_SIZE,
    "lr":           LR,
    "weight_decay": WEIGHT_DECAY,
    "hidden_dims":  HIDDEN_DIMS,
    "dropout":      DROPOUT,
    "focal_gamma":  FOCAL_GAMMA
})

# ---------------------- Prepare Data ----------------------
TAXONOMY_CSV   = os.path.join(BIRDCLEF_BASE_DIR, "Data", "birdclef-2025", "taxonomy.csv")
TRAIN_MANIFEST = os.path.join(BIRDCLEF_BASE_DIR, "Features", "manifest_val.csv")
TEST_MANIFEST  = os.path.join(BIRDCLEF_BASE_DIR, "Features", "manifest_test.csv")
TRAIN_META     = os.path.join(BIRDCLEF_BASE_DIR, "Data", "birdclef-2025", "train.csv")
FEATURE_BASE   = os.path.join(BIRDCLEF_BASE_DIR, "Features")

tax_df   = pd.read_csv(TAXONOMY_CSV)
CLASSES  = sorted(tax_df["primary_label"].astype(str).tolist())
NUM_CLS  = len(CLASSES)
IDX_MAP  = {c:i for i,c in enumerate(CLASSES)}

class EnsembleDataset(Dataset):
    def __init__(self, manifest, meta_csv, base):
        m = pd.read_csv(manifest)
        meta = pd.read_csv(meta_csv, usecols=["filename","secondary_labels"])
        meta["rid"]  = meta.filename.str.replace(r"\.ogg$","",regex=True)
        meta["secs"] = meta.secondary_labels.fillna("").str.split()
        sec_map = dict(zip(meta.rid, meta.secs))
        self.rows = []
        for _, r in m.iterrows():
            rid  = r.chunk_id.split("_chk")[0]
            labs = [r.primary_label] + sec_map.get(rid,[])
            labs = [l for l in labs if l in IDX_MAP]
            prim = IDX_MAP[r.primary_label]
            emb_p = os.path.join(base,"embeddings", r.emb_path.lstrip(os.sep))
            ma_p  = os.path.join(base,"mel_aug",    r.mel_aug_path.lstrip(os.sep))
            m_p   = os.path.join(base,"mel",        r.mel_path.lstrip(os.sep))
            wav_p = os.path.join(base,"denoised",   r.audio_path.lstrip(os.sep))
            self.rows.append((emb_p, ma_p, m_p, wav_p, labs, prim))

    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        emb_p, ma_p, m_p, wav_p, labs, prim = self.rows[i]
        emb_arr = np.load(emb_p)["embedding"].mean(axis=0).astype(np.float32)
        emb     = torch.from_numpy(emb_arr)
        ma_arr  = np.load(ma_p)["mel"].astype(np.float32); ma = torch.from_numpy(ma_arr).unsqueeze(0)
        m_arr   = np.load(m_p)["mel"].astype(np.float32); m = torch.from_numpy(m_arr).unsqueeze(0)
        wav, _  = torchaudio.load(wav_p); wav = wav.squeeze(0)
        T = 32000*10
        wav = F.pad(wav,(0,T-wav.size(0))) if wav.size(0)<T else wav[:T]
        wav = (wav - wav.mean())/wav.std().clamp_min(1e-6)
        y = torch.zeros(NUM_CLS, dtype=torch.float32)
        for l in labs: y[IDX_MAP[l]] = 1.0
        return emb, ma, m, wav, y, prim

def ensemble_collate_fn(batch):
    embs, mas, ms, wavs, ys, prims = zip(*batch)
    return (torch.stack(embs), torch.stack(mas), torch.stack(ms),
            torch.stack(wavs), torch.stack(ys), torch.tensor(prims))

train_ds = EnsembleDataset(TRAIN_MANIFEST, TRAIN_META, FEATURE_BASE)
test_ds  = EnsembleDataset(TEST_MANIFEST,  TRAIN_META, FEATURE_BASE)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, collate_fn=ensemble_collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, collate_fn=ensemble_collate_fn)

# ---------------------- MetaMLP Head ----------------------
class MetaMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout):
        super().__init__()
        layers, dims = [], [in_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [ nn.Linear(dims[i], dims[i+1]),
                        nn.BatchNorm1d(dims[i+1]), nn.ReLU(), nn.Dropout(dropout) ]
        layers.append(nn.Linear(dims[-1], NUM_CLS))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

meta_model = MetaMLP(NUM_CLS*4, HIDDEN_DIMS, DROPOUT).to(DEVICE)
mlflow.log_param("meta_in_dim", NUM_CLS*4)

# ---------------------- Loss / Opt / Scheduler ----------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma): super().__init__(); self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        return ((1-p_t)**self.gamma * bce).mean()

criterion = FocalLoss(FOCAL_GAMMA)
optimizer = torch.optim.AdamW(meta_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.1, div_factor=10)
scaler    = GradScaler()

# ---------------------- Retraining Loop ----------------------
best_f1 = best_ap = best_acc = 0.0
thresholds = np.full(NUM_CLS, TH_INIT, dtype=np.float32)

for epoch in range(1, EPOCHS+1):
    meta_model.train()
    run_loss = total = 0
    for emb, ma, m, wav, yb, _ in tqdm(train_loader, desc=f"[{epoch}/{EPOCHS}] Train"):
        emb, ma, m, wav, yb = [t.to(DEVICE) for t in (emb,ma,m,wav,yb)]
        with torch.no_grad():
            p1 = torch.sigmoid(emb_model(emb))
            p2 = torch.sigmoid(raw_model(wav))
            p3 = torch.sigmoid(res_model(ma))
            p4 = torch.sigmoid(eff_model(m))
        feat = torch.cat([p1,p2,p3,p4], dim=1)
        optimizer.zero_grad()
        with autocast():
            logits = meta_model(feat)
            loss   = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        run_loss += loss.item()*feat.size(0)
        total    += feat.size(0)
    mlflow.log_metric("train_loss", run_loss/total, step=epoch)

    meta_model.eval()
    val_loss = total = 0
    all_scores, all_tgts, all_prims = [], [], []
    with torch.no_grad():
        for emb, ma, m, wav, yb, prim in tqdm(test_loader, desc=f"[{epoch}/{EPOCHS}] Eval"):
            emb, ma, m, wav, yb = [t.to(DEVICE) for t in (emb,ma,m,wav,yb)]
            p1 = torch.sigmoid(emb_model(emb))
            p2 = torch.sigmoid(raw_model(wav))
            p3 = torch.sigmoid(res_model(ma))
            p4 = torch.sigmoid(eff_model(m))
            feat = torch.cat([p1,p2,p3,p4], dim=1)
            logits = meta_model(feat)
            loss   = criterion(logits, yb)
            val_loss += loss.item()*feat.size(0)
            total    += feat.size(0)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(scores); all_tgts.append(yb.cpu().numpy())
            all_prims.extend(prim.tolist())
    mlflow.log_metric("val_loss", val_loss/total, step=epoch)

    scores = np.vstack(all_scores); tgts = np.vstack(all_tgts); prims = np.array(all_prims)
    for i in range(NUM_CLS):
        y_true, y_score = tgts[:,i], scores[:,i]
        if 0<y_true.sum()<len(y_true):
            prec, rec, th = precision_recall_curve(y_true, y_score)
            f1s = 2*prec*rec/(prec+rec+1e-8)
            thresholds[i] = th[np.nanargmax(f1s[:-1])]
    preds       = (scores>=thresholds).astype(int)
    micro_f1    = f1_score(tgts,preds,average="micro",zero_division=0)
    micro_ap    = average_precision_score(tgts,scores,average="micro")
    primary_acc = (scores.argmax(1)==prims).mean()

    mlflow.log_metrics({
        "micro_f1":    micro_f1,
        "micro_ap":    micro_ap,
        "primary_acc": primary_acc
    }, step=epoch)

    if micro_f1>best_f1:
        best_f1,best_ap,best_acc = micro_f1,micro_ap,primary_acc
        torch.save(meta_model.state_dict(), BEST_CKPT)
        mlflow.log_artifact(BEST_CKPT, artifact_path="model")

    print(f"→ Epoch {epoch}/{EPOCHS}  F1={micro_f1:.4f}  AP={micro_ap:.4f}  PrimAcc={primary_acc:.4f}")

mlflow.log_metric("best_micro_f1", best_f1)
mlflow.log_metric("best_micro_ap", best_ap)
mlflow.log_metric("best_primary_acc", best_acc)

LOCAL_MODEL_DIR = f"{args.meta_model_name}_model_retrain"
mlflow.pytorch.save_model(meta_model, LOCAL_MODEL_DIR)
mlflow.log_artifacts(LOCAL_MODEL_DIR, artifact_path=f"{args.meta_model_name}_model_retrain")

mlflow.end_run()
