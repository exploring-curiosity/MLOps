import os
import csv


def write_predictions_to_csv(predictions, output_dir, base_filename):
    if not predictions or not isinstance(predictions, list):
        print("[Error] predictions must be a non-empty list of tuples")
        return
    try:
        os.makedirs(output_dir, exist_ok=True)
        out_csv = os.path.join(output_dir, f"{base_filename}.csv")
        with open(out_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["chunk_filename", "top_species_label", "confidence_score"])
            for chunk_name, label, confidence in predictions:
                writer.writerow([chunk_name, label, confidence])
        print(f"[Saved] {out_csv}")
    except (PermissionError, OSError) as e:
        print(f"[Error] Failed to write CSV: {e}")
