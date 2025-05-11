import os
import pandas as pd

# Paths
AUDIO_ROOT = "/Users/vaishnavideshmukh/Documents/MLOps/MLOps_Project/birdclef-2025/train_audio"
TRAIN_CSV = "/Users/vaishnavideshmukh/Documents/MLOps/MLOps_Project/birdclef-2025/train.csv"
OUTPUT_CSV = "data/manifest.csv"

# Load train.csv
df = pd.read_csv(TRAIN_CSV)

# Extract actual filename only (drop the subdir part from train.csv if present)
df["file_only"] = df["filename"].apply(lambda x: os.path.basename(str(x).strip()).lower())

# Recursively gather audio file info
audio_entries = []
for root, _, files in os.walk(AUDIO_ROOT):
    for file in files:
        if file.lower().endswith(".ogg"):
            rel_path = os.path.relpath(os.path.join(root, file), AUDIO_ROOT)
            audio_entries.append((file.lower(), rel_path))

audio_df = pd.DataFrame(audio_entries, columns=["file_only", "rel_path"])

# Merge on extracted filename
merged = pd.merge(df, audio_df, on="file_only", how="inner")

# Optional debug
print(f"[DEBUG] Matched entries: {len(merged)}")

# Final manifest
manifest_df = merged[[
    "rel_path", "primary_label", "secondary_labels",
    "rating", "collection", "latitude", "longitude"
]].rename(columns={"rel_path": "filename"})

# Save manifest
os.makedirs("data", exist_ok=True)
manifest_df.to_csv(OUTPUT_CSV, index=False)
print(f"Manifest saved to {OUTPUT_CSV} with {len(manifest_df)} entries.")
