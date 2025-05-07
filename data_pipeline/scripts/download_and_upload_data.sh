#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Script: download_and_upload_data.sh
# Purpose:
#   - Download BirdCLEF 2025 dataset using Kaggle CLI
#   - Extract only selected parts of the dataset
#   - Upload the required subsets to object store (via rclone)
# Target: object-persist-project38 (mounted at /mnt/object)
# Requires: rclone configured, .env loaded or exported
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# Step 1: Ensure Kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "[INFO] Installing Kaggle CLI..."
    pip install kaggle
fi

# Step 2: Move kaggle.json into the correct location
mkdir -p ~/.kaggle
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "[ERROR] kaggle.json not found in home directory. Please upload it first."
    exit 1
fi
chmod 600 ~/.kaggle/kaggle.json

# Step 3: Download and unzip BirdCLEF 2025 dataset
mkdir -p ~/Data
cd ~/Data

if [ ! -f birdclef-2025.zip ]; then
    echo "[INFO] Downloading BirdCLEF 2025 dataset..."
    kaggle competitions download -c birdclef-2025
fi

if [ ! -d birdclef-2025 ]; then
    echo "[INFO] Unzipping dataset..."
    unzip -qq birdclef-2025.zip -d birdclef-2025
fi

# Step 4: Upload raw data to object store
echo "[INFO] Uploading train_audio/ to /mnt/object/raw/train_audio/"
rclone copy ~/Data/birdclef-2025/train_audio /mnt/object/raw/train_audio --progress

echo "[INFO] Uploading train.csv and taxonomy.csv to /mnt/object/raw/"
rclone copy ~/Data/birdclef-2025/train.csv /mnt/object/raw/ --progress
rclone copy ~/Data/birdclef-2025/taxonomy.csv /mnt/object/raw/ --progress

# Step 5: Select 10% of most recent train_soundscapes
mkdir -p ~/Data/production_sample
TRAIN_SCAPES_DIR=~/Data/birdclef-2025/train_soundscapes
SAMPLE_DIR=~/Data/production_sample

file_count=$(ls -1 $TRAIN_SCAPES_DIR/*.ogg | wc -l)
percent_count=$((file_count / 10))

echo "[INFO] Sampling $percent_count files from train_soundscapes..."
ls -1 $TRAIN_SCAPES_DIR/*.ogg | sort | tail -n $percent_count | while read f; do
    cp "$f" "$SAMPLE_DIR/"
done

# Step 6: Upload sampled production data
echo "[INFO] Uploading sampled train_soundscapes to /mnt/object/raw/production/train_soundscapes_subset/"
rclone copy $SAMPLE_DIR /mnt/object/raw/production/train_soundscapes_subset --progress

echo "All data uploaded to object store successfully."