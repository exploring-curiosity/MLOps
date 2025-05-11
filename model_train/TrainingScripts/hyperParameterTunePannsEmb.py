import os
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
from torch.amp import autocast, GradScaler
import argparse

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.mlflow import MLflowLoggerCallback

# ──────────────────────────────────────────────────────────────────────────────
# Dataset + model definitions (copied from your training script)
# ──────────────────────────────────────────────────────────────────────────────
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
        self.idx_map= {c:i for i,c in enumerate(classes)}
        self.num_cls= len(classes)
        self.key    = key

        for _, row in m.iterrows():
            rid         = row.chunk_id.split("_chk")[0]
            labs        = [row.primary_label] + sec_map.get(rid, [])
            self.rows.append((row.emb_path, labs, row.primary_label))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, labs, primary_cls = self.rows[i]
        arr = np.load(path)[self.key]
        x   = arr.mean(axis=0).astype(np.float32)
        y   = np.zeros(self.num_cls, dtype=np.float32)
        for c in labs:
            idx = self.idx_map.get(c)
            if idx is not None: y[idx]=1.0
        prim_idx = self.idx_map[primary_cls]
        return x, y, prim_idx

class EmbeddingClassifier(nn.Module):
    def __init__(self, emb_dim, num_cls, hidden_dims, dropout):
        super().__init__()
        layers = []
        dims = [emb_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        layers.append(nn.Linear(hidden_dims[-1], num_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none"
        )
        p_t = torch.exp(-bce)
        return ((1 - p_t)**self.gamma * bce).mean()

def mixup(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha>0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

# ──────────────────────────────────────────────────────────────────────────────
# Trainable function for Ray Tune
# ──────────────────────────────────────────────────────────────────────────────
def train_tune(config):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # paths & classes
    base = os.getenv("BIRDCLEF_BASE_DIR", "/mnt/data")
    TAXONOMY_CSV = os.path.join(base, "Data","birdclef-2025","taxonomy.csv")
    TRAIN_MAN    = os.path.join(base, "Features","manifest_train.csv")
    TEST_MAN     = os.path.join(base, "Features","manifest_test.csv")
    TRAIN_META   = os.path.join(base, "Data","birdclef-2025","train.csv")
    FEATURE_BASE = os.path.join(base, "Features")

    tax_df = pd.read_csv(TAXONOMY_CSV)
    classes = sorted(tax_df["primary_label"].astype(str).tolist())

    # datasets & loaders
    train_ds = EmbeddingDataset(TRAIN_MAN, TRAIN_META, FEATURE_BASE, classes, key="embedding")
    val_ds   = EmbeddingDataset(TEST_MAN,  TRAIN_META, FEATURE_BASE, classes, key="embedding")
    train_loader = DataLoader(train_ds,
                              batch_size=int(config["batch_size"]),
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,
                              batch_size=int(config["batch_size"]),
                              shuffle=False,num_workers=4, pin_memory=True)

    # compute class weights
    counts = np.zeros(len(classes), dtype=int)
    for _, labs, _ in train_ds.rows:
        for c in labs:
            counts[train_ds.idx_map[c]] += 1
    neg = len(train_ds) - counts
    pw  = np.ones(len(classes),dtype=np.float32)
    mask = counts>0
    pw[mask] = neg[mask]/counts[mask]
    pos_weight = torch.tensor(pw, device=device)

    # model, loss, optimizer, scheduler
    sample_x,_,_ = train_ds[0]
    model = EmbeddingClassifier(
        emb_dim=sample_x.shape[0],
        num_cls=len(classes),
        hidden_dims=config["hidden_dims"],
        dropout=config["dropout"]
    ).to(device)

    criterion = FocalLoss(config["focal_gamma"], pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
        pct_start=0.1,
        div_factor=10
    )
    scaler = GradScaler()

    thresholds = np.full(len(classes), 0.5, dtype=np.float32)

    for epoch in range(1, config["epochs"]+1):
        # train
        model.train()
        running_loss = 0.0; total=0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb_m, ya, yb_m, lam = mixup(xb, yb, config["mixup_alpha"])
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                logits = model(xb_m)
                loss   = lam*criterion(logits, ya) + (1-lam)*criterion(logits, yb_m)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            scheduler.step()

            bs = xb.size(0)
            running_loss += loss.item()*bs; total += bs
        train_loss = running_loss/total

        # validation
        model.eval()
        val_loss=0.0; total=0
        all_scores=[]; all_tgts=[]
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(device_type="cuda"):
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item()*xb.size(0)
                    scores   = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(scores)
                all_tgts.append(yb.cpu().numpy())
                total += xb.size(0)
        val_loss /= total

        scores = np.vstack(all_scores)
        tgts   = np.vstack(all_tgts)
        # calibrate thresholds
        for i in range(len(classes)):
            y_true = tgts[:,i]
            if 0<y_true.sum()<len(y_true):
                prec,rec,th = precision_recall_curve(y_true, scores[:,i])
                f1s = 2*prec*rec/(prec+rec+1e-8)
                thresholds[i] = th[np.nanargmax(f1s[:-1])]
        preds    = (scores>=thresholds).astype(int)
        micro_f1 = f1_score(tgts, preds, average="micro", zero_division=0)
        micro_ap = average_precision_score(tgts, scores, average="micro")

        # report metrics
        tune.report({
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "micro_f1":   micro_f1,
            "micro_ap":   micro_ap,
        })

# ──────────────────────────────────────────────────────────────────────────────
# Main: Ray Tune run
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--num_samples", type=int,   default=20)
    args = parser.parse_args()

    # search space
    config = {
        "lr":            tune.loguniform(1e-4, 1e-2),
        "weight_decay":  tune.loguniform(1e-6, 1e-2),
        "batch_size":    tune.choice([32, 64, 128]),
        "hidden_dims":   tune.choice([[2048,1024,512], [1024,512,256]]),
        "dropout":       tune.uniform(0.3, 0.7),
        "mixup_alpha":   tune.uniform(0.0, 0.6),
        "focal_gamma":   tune.choice([1.0, 2.0, 3.0]),
        "epochs":        args.epochs
    }

    scheduler = ASHAScheduler(
        metric="micro_f1", mode="max",
        max_t=args.epochs, grace_period=1, reduction_factor=2
    )

    mlflow_cb = MLflowLoggerCallback(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", None),
        experiment_name="PannsMLP_Hyperparam_Tune"
    )

    tune.run(
        train_tune,
        name="asha_hpt",
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        callbacks=[mlflow_cb],
        resources_per_trial={"cpu": 4, "gpu": 0.25},
    )
