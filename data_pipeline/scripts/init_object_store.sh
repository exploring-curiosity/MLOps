#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Script: init_object_store.sh (CHI@UC version)
# Purpose: Provision and mount Chameleon Object Store container
#     using rclone with env variable-based config.
# Target Site: CHI@UC
# Container Name: object-persist-project38
# Mount Point: /mnt/object
# Auth via ENV vars (see .env.example)
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# Step 1: Install rclone if not already installed
if ! command -v rclone &> /dev/null; then
  echo "[INFO] Installing rclone..."
  curl https://rclone.org/install.sh | sudo bash
fi

# Step 2: Ensure /etc/fuse.conf allows non-root mounts
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

# Step 3: Prepare mount point
sudo mkdir -p /mnt/object
sudo chown -R cc:cc /mnt/object

# Step 4: Set up rclone env vars for CHI@UC
export RCLONE_CONFIG_CHI_UC_TYPE=swift
export RCLONE_CONFIG_CHI_UC_USER=019d1a6fa51a4e35975ac31a6a1aec75
export RCLONE_CONFIG_CHI_UC_KEY=dxH98yDYRieIf4RthqBH-bXx3r9Ofzfs_zp5qNV_DbhHFqMTRdvpbQBnuN3mBiObaut1bfcAlZkKDHBYTVAhlg
export RCLONE_CONFIG_CHI_UC_AUTH=https://chi.uc.chameleoncloud.org:5000/v3
export RCLONE_CONFIG_CHI_UC_REGION=CHI@UC

# Step 5: Create the container if it doesn't already exist
echo "[INFO] Creating object container if not exists..."
rclone mkdir chi_uc:object-persist-project38 || true

# Step 6: Mount the object store to /mnt/object
echo "[INFO] Mounting object store 'object-persist-project38'..."
rclone mount chi_uc:object-persist-project38 /mnt/object --allow-other --daemon

# Step 7: Confirm mount worked
echo "[INFO] Verifying mount contents..."
ls /mnt/object || echo "[WARNING] Mount failed or is empty. Check rclone credentials."

echo "Object store 'object-persist-project38' mounted successfully at /mnt/object"