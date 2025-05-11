import os
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from peft import get_peft_model, LoraConfig
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
from torch.amp import autocast, GradScaler


from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.mlflow import MLflowLoggerCallback

import argparse

# ──────────────────────────────────────────────────────────────────────────────
# Dataset + model builder (exactly as your original script, but parameterized)
# ──────────────────────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, manifest_csv, meta_csv, base, classes, key="mel"):
        m = pd.read_csv(manifest_csv)
        m["mel_path"] = (
            m["mel_path"].astype(str)
             .str.lstrip(os.sep)
             .apply(lambda p: os.path.join(base,"mel",p))
        )
        meta = pd.read_csv(meta_csv, usecols=["filename","secondary_labels"])
        meta["rid"]  = meta.filename.str.replace(r"\.ogg$","",regex=True)
        meta["secs"]= meta.secondary_labels.fillna("").str.split()
        sec_map = dict(zip(meta.rid, meta.secs))

        self.rows    = []
        self.idx_map = {c:i for i,c in enumerate(classes)}
        self.num_cls = len(classes)
        self.key     = key

        for _, r in m.iterrows():
            rid     = r.chunk_id.split("_chk")[0]
            labs    = [r.primary_label] + sec_map.get(rid, [])
            labs    = [l for l in labs if l in self.idx_map]
            prim_idx= self.idx_map[r.primary_label]
            self.rows.append((r.mel_path, labs, prim_idx))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, labs, prim_idx = self.rows[i]
        arr = np.load(path)[self.key]                # [n_mels,n_frames]
        x   = torch.from_numpy(arr).unsqueeze(0).float()  # [1,n_mels,n_frames]
        y   = torch.zeros(self.num_cls, dtype=torch.float32)
        for c in labs:
            y[self.idx_map[c]] = 1.0
        return x, y, prim_idx


def build_efficientnetb3_lora(num_classes, lora_r, lora_alpha, lora_dropout):
    TARGET_MODULES  = ["conv_pw","conv_dw","conv_pwl","conv_head"]
    MODULES_TO_SAVE = ["classifier"]

    base = timm.create_model("efficientnet_b3", pretrained=True)
    orig_fwd = base.forward
    def forward_patch(*args, **kwargs):
        x = kwargs.pop("input_ids", args[0])
        return orig_fwd(x)
    base.forward = forward_patch

    stem = base.conv_stem
    base.conv_stem = nn.Conv2d(1, stem.out_channels,
                               kernel_size=stem.kernel_size,
                               stride=stem.stride,
                               padding=stem.padding,
                               bias=False)
    base.classifier = nn.Linear(base.classifier.in_features, num_classes)

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=MODULES_TO_SAVE,
        task_type="FEATURE_EXTRACTION",
        inference_mode=False
    )
    return get_peft_model(base, lora_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# Trainable function for Ray Tune
# ──────────────────────────────────────────────────────────────────────────────
def train_tune(config, checkpoint_dir=None):
    # config contains: lr, weight_decay, batch_size, lora_r, lora_alpha, lora_dropout
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load taxonomy + classes
    base = os.getenv("BIRDCLEF_BASE_DIR", "/mnt/data")
    tax_df = pd.read_csv(os.path.join(base, "Data","birdclef-2025","taxonomy.csv"))
    classes = sorted(tax_df["primary_label"].astype(str).tolist())
    num_classes = len(classes)

    # build datasets & loaders
    train_ds = MelDataset(
        os.path.join(base, "Features","manifest_train.csv"),
        os.path.join(base, "Data","birdclef-2025","train.csv"),
        os.path.join(base, "Features"),
        classes
    )
    val_ds = MelDataset(
        os.path.join(base, "Features","manifest_test.csv"),
        os.path.join(base, "Data","birdclef-2025","train.csv"),
        os.path.join(base, "Features"),
        classes
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True
    )

    # build model
    model = build_efficientnetb3_lora(
        num_classes,
        config["lora_r"],
        config["lora_alpha"],
        config["lora_dropout"]
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
        pct_start=0.1,
        div_factor=10
    )
    scaler = GradScaler()
    thresholds = np.full(num_classes, 0.5, dtype=np.float32)

    # optional mixup
    def mixup(x, y, alpha=0.4):
        lam = np.random.beta(alpha,alpha) if alpha>0 else 1.0
        idx = torch.randperm(x.size(0), device=x.device)
        return lam*x + (1-lam)*x[idx], y, y[idx], lam

    for epoch in range(1, config["epochs"] + 1):
        # train
        model.train()
        running_loss = 0.0
        total = 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb_m, ya, yb_m, lam = mixup(xb, yb)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                logits = model(xb_m)
                loss = lam*criterion(logits, ya) + (1-lam)*criterion(logits, yb_m)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            bs = xb.size(0)
            running_loss += loss.item() * bs
            total += bs
        train_loss = running_loss / total

        # validation
        model.eval()
        val_loss = 0.0
        total = 0
        all_scores = []
        all_tgts   = []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(device_type="cuda"):
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item() * xb.size(0)
                    scores = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(scores)
                all_tgts.append(yb.cpu().numpy())
                total += xb.size(0)
        val_loss = val_loss / total

        scores = np.vstack(all_scores)
        tgts   = np.vstack(all_tgts)

        # calibrate thresholds
        for i in range(num_classes):
            y_true = tgts[:,i]
            if 0 < y_true.sum() < len(y_true):
                prec, rec, th = precision_recall_curve(y_true, scores[:,i])
                f1s = 2*prec*rec/(prec+rec+1e-8)
                thresholds[i] = th[np.nanargmax(f1s[:-1])]

        preds    = (scores >= thresholds).astype(int)
        micro_f1 = f1_score(tgts, preds, average="micro", zero_division=0)
        micro_ap = average_precision_score(tgts, scores, average="micro")

        # report to Tune (and MLflow via callback)
        tune.report({
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "micro_f1":   micro_f1,
            "micro_ap":   micro_ap,
        })


# ──────────────────────────────────────────────────────────────────────────────
# Main: Ray Tune run with ASHA + MLflow
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--num_samples",   type=int,   default=20)
    args = parser.parse_args()

    # define search space
    config = {
        "lr":              tune.loguniform(1e-5, 1e-2),
        "weight_decay":    tune.loguniform(1e-6, 1e-2),
        "batch_size":      tune.choice([32, 64, 128]),
        "lora_r":          tune.choice([4, 8, 12, 16]),
        "lora_alpha":      tune.choice([8, 16, 24, 32]),
        "lora_dropout":    tune.uniform(0.0, 0.5),
        "epochs":          args.epochs,
    }

    # ASHA scheduler
    scheduler = ASHAScheduler(
        metric="micro_f1",
        mode="max",
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2
    )

    # MLflow logger callback
    mlflow_cb = MLflowLoggerCallback(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", None),
        experiment_name="EfficientNetB3_LoRA_tuning"
    )

    tune.run(
        train_tune,
        name="asha_tuning",
        config=config,
        num_samples=args.num_samples,
        scheduler=scheduler,
        callbacks=[mlflow_cb],
        resources_per_trial={"cpu": 4, "gpu": 0.25},
    )
