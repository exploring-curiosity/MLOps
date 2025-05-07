#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Script: init_object_store.sh (Option 2: ENV-based setup)
# Purpose: Provision and mount object store via env config
# Site: CHI@UC | Container: object-persist-project38
# Requires: .env file with RCLONE_CONFIG_CHI_UC_* variables
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# Step 1: Install rclone if not installed
if ! command -v rclone &> /dev/null; then
  echo "[INFO] Installing rclone..."
  curl https://rclone.org/install.sh | sudo bash
fi

# Step 2: Ensure fuse is configured properly
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

# Step 3: Load env variables from .env file
if [ ! -f .env ]; then
  echo "[ERROR] .env file not found in current directory. Exiting."
  exit 1
fi

export $(grep -v '^#' .env | xargs)

# Step 4: Verify required environment variables
required_vars=(
  RCLONE_CONFIG_CHI_UC_TYPE
  RCLONE_CONFIG_CHI_UC_USER
  RCLONE_CONFIG_CHI_UC_KEY
  RCLONE_CONFIG_CHI_UC_AUTH
  RCLONE_CONFIG_CHI_UC_REGION
)

for var in "${required_vars[@]}"; do
  if [ -z "${!var:-}" ]; then
    echo "[ERROR] Missing environment variable: $var"
    exit 1
  fi
done

# Step 5: Create mount point
sudo mkdir -p /mnt/object
sudo chown -R cc:cc /mnt/object

# Step 6: Test rclone config and create container
echo "[INFO] Validating remote configuration with rclone..."
rclone lsd chi_uc: || { echo "[ERROR] Remote validation failed"; exit 1; }

echo "[INFO] Creating container 'object-persist-project38' if it does not exist..."
rclone mkdir chi_uc:object-persist-project38 || true

# Step 7: Mount the object store
rclone mount chi_uc:object-persist-project38 /mnt/object --allow-other --daemon
sleep 2

# Step 8: Confirm mount
echo "[INFO] Listing contents of /mnt/object:"
ls -al /mnt/object

echo "Object store successfully mounted at /mnt/object"