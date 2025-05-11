#!/usr/bin/env python3

import os
import math
import numpy as np
import pandas as pd
from pathlib import Path

# Base paths
BASE_FEAT = Path("/workspace/tmp_manifests/processed")
DEN_MAN = BASE_FEAT / "denoised" / "manifest.csv"
MEL_MAN = BASE_FEAT / "mel" / "manifest.csv"
EMB_MAN = BASE_FEAT / "embeddings" / "manifest.csv"
AUG_MAN = BASE_FEAT / "mel_aug" / "manifest.csv"
OUT_DIR = BASE_FEAT

# Load and merge
den_df = pd.read_csv(DEN_MAN)[["chunk_id", "audio_path", "primary_label"]]
mel_df = pd.read_csv(MEL_MAN)[["chunk_id", "mel_path"]]
emb_df = pd.read_csv(EMB_MAN)[["chunk_id", "emb_path"]]
aug_df = pd.read_csv(AUG_MAN)[["chunk_id", "mel_aug_path"]]

df = (
    den_df
    .merge(mel_df, on="chunk_id")
    .merge(emb_df, on="chunk_id")
    .merge(aug_df, on="chunk_id")
)
df["recording_id"] = df["chunk_id"].str.split("_chk").str[0]

unique_recs = df[["recording_id", "primary_label"]].drop_duplicates()
rng = np.random.default_rng(42)
seed_per_species = (
    unique_recs.groupby("primary_label")["recording_id"]
    .apply(lambda ids: rng.choice(ids.values, 1)[0])
    .tolist()
)

all_recs = unique_recs["recording_id"].tolist()
remaining = [r for r in all_recs if r not in seed_per_species]

n_total = len(all_recs)
n_train_target = int(round(0.70 * n_total))
n_additional = max(0, n_train_target - len(seed_per_species))
addl_train = rng.choice(remaining, size=n_additional, replace=False).tolist()
train_recs = set(seed_per_species + addl_train)

remaining_after_train = [r for r in all_recs if r not in train_recs]
n_test = int(round(0.10 * n_total))
test_recs = set(rng.choice(remaining_after_train, size=n_test, replace=False).tolist())
val_recs = set(r for r in all_recs if r not in train_recs and r not in test_recs)

df_train = df[df["recording_id"].isin(train_recs)].reset_index(drop=True)
df_test = df[df["recording_id"].isin(test_recs)].reset_index(drop=True)
df_val = df[df["recording_id"].isin(val_recs)].reset_index(drop=True)

for name, split_df in [("train", df_train), ("test", df_test), ("val", df_val)]:
    out = split_df.drop(columns=["recording_id"])
    out.to_csv(OUT_DIR / f"manifest_{name}.csv", index=False)

train_df = df_train.copy()
train_df["recording_id"] = train_df["chunk_id"].str.split("_chk").str[0]
groups = (
    train_df[["recording_id", "primary_label"]]
    .drop_duplicates()
    .groupby("primary_label")["recording_id"]
    .apply(list)
    .to_dict()
)

target = int(np.median([len(v) for v in groups.values()]))
oversampled_recs = []
for sp, recs in groups.items():
    oversampled_recs.extend(recs)
    n_extra = max(0, target - len(recs))
    if n_extra:
        extras = rng.choice(recs, size=n_extra, replace=True)
        oversampled_recs.extend(extras.tolist())

frames = [train_df[train_df["recording_id"] == rec] for rec in oversampled_recs]
oversampled_df = pd.concat(frames, ignore_index=True).drop(columns=["recording_id"])
oversampled_df.to_csv(OUT_DIR / "manifest_train_oversampled.csv", index=False)