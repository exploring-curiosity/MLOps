#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Script: init_object_store.sh
# Purpose: Create and mount Chameleon Object Store (CHI@UC)
# Auth via ~/.env and writes rclone.conf to avoid rclone service discovery
# Notes: This script avoids uploading credentials by using a .env file
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# Step 0: Load .env file containing sensitive credentials
if [ ! -f .env ]; then
  echo "[ERROR] .env file missing in current directory. Please create it with your credential ID and SECRET."
  exit 1
fi

export $(grep -v '^#' .env | xargs)

# Step 1: Ensure rclone is installed
if ! command -v rclone &> /dev/null; then
  echo "[INFO] Installing rclone..."
  curl https://rclone.org/install.sh | sudo bash
fi

# Step 2: Prepare fuse settings and mount point
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf
sudo mkdir -p /mnt/object
sudo chown -R cc:cc /mnt/object

# Step 3: Dynamically write rclone.conf using credentials from env
mkdir -p ~/.config/rclone

cat > ~/.config/rclone/rclone.conf <<EOF
[chi_uc]
type = swift
user_id = $RCLONE_CONFIG_CHI_UC_USER_ID
user = $RCLONE_CONFIG_CHI_UC_USER
key = $RCLONE_CONFIG_CHI_UC_KEY
auth = https://chi.uc.chameleoncloud.org:5000/v3
region = CHI@UC
auth_version = 3
storage_url = https://objects.chi.uc.chameleoncloud.org/v1/AUTH_$RCLONE_PROJECT_ID
EOF

# Step 4: Test and create container if missing
echo "[INFO] Verifying access to object store..."
rclone lsd chi_uc: || {
  echo "[ERROR] Failed to access chi_uc remote. Check credentials and project ID."; exit 1;
}

# Step 5: Create container if it doesn't exist
rclone mkdir chi_uc:object-persist-project38 || true

# Step 6: Mount object store to /mnt/object
rclone mount chi_uc:object-persist-project38 /mnt/object --allow-other --daemon
sleep 2

# Step 7: Confirm mount worked
echo "[INFO] Listing contents of mounted object store:"
ls -al /mnt/object

echo "Object store mounted successfully at /mnt/object"