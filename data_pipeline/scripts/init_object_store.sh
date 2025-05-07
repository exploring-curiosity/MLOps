#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# Script: init_object_store.sh
# Purpose: Securely configure and mount object store on CHI@UC using rclone
# Container: object-persist-project38
# Mount Point: /mnt/object
# Depends on: ~/.env file containing required variables
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "[INFO] Loading environment variables..."
if [[ ! -f .env ]]; then
    echo "[ERROR] .env file not found in $(pwd)"
    exit 1
fi

export $(grep -v '^#' .env | xargs)

# Validate required variables
required_vars=(RCLONE_CONFIG_CHI_UC_USER RCLONE_CONFIG_CHI_UC_KEY RCLONE_CONFIG_CHI_UC_AUTH RCLONE_CONFIG_CHI_UC_REGION)
for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "[ERROR] Missing environment variable: $var"
        exit 1
    fi
done

# Step 1: Install rclone if needed
if ! command -v rclone &> /dev/null; then
    echo "[INFO] Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
fi

# Step 2: Enable user mounts
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

# Step 3: Create mount directory
echo "[INFO] Creating /mnt/object..."
sudo mkdir -p /mnt/object
sudo chown -R cc:cc /mnt/object

# Step 4: Setup rclone configuration (in-memory)
export RCLONE_CONFIG_CHI_UC_TYPE=swift
export RCLONE_CONFIG_CHI_UC_USER=$RCLONE_CONFIG_CHI_UC_USER
export RCLONE_CONFIG_CHI_UC_KEY=$RCLONE_CONFIG_CHI_UC_KEY
export RCLONE_CONFIG_CHI_UC_AUTH=$RCLONE_CONFIG_CHI_UC_AUTH
export RCLONE_CONFIG_CHI_UC_REGION=$RCLONE_CONFIG_CHI_UC_REGION

# Step 5: Verify connection
echo "[INFO] Validating remote configuration with rclone..."
if ! rclone lsd chi_uc: &> /dev/null; then
    echo "[ERROR] Failed to access chi_uc remote. Check credentials and project ID."
    exit 1
fi

# Step 6: Create container if it doesn't exist
echo "[INFO] Creating object store container 'object-persist-project38' if not present..."
rclone mkdir chi_uc:object-persist-project38 || true

# Step 7: Mount the container
echo "[INFO] Mounting container at /mnt/object..."
rclone mount chi_uc:object-persist-project38 /mnt/object --allow-other --daemon

# Final Check
sleep 2
if mountpoint -q /mnt/object; then
    echo "[SUCCESS] Object store mounted at /mnt/object"
else
    echo "[ERROR] Mount failed. Check logs and rclone config."
fi
