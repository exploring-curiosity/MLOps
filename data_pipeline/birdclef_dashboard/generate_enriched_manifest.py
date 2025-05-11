import pandas as pd

# Paths
manifest_path = "data/manifest.csv"
taxonomy_path = "/Users/vaishnavideshmukh/Documents/MLOps/MLOps_Project/birdclef-2025/taxonomy.csv"
output_path = "data/manifest_enriched.csv"

# Load both CSVs
manifest = pd.read_csv(manifest_path)
taxonomy = pd.read_csv(taxonomy_path)

# Normalize keys
manifest["primary_label"] = manifest["primary_label"].str.strip().str.lower()
taxonomy["primary_label"] = taxonomy["primary_label"].str.strip().str.lower()

# Merge on primary_label
merged = pd.merge(manifest, taxonomy, on="primary_label", how="left")

# Save enriched manifest
merged.to_csv(output_path, index=False)
print(f"Enriched manifest saved to {output_path} with {len(merged)} entries.")
