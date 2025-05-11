import os
import math
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from tqdm import tqdm

CHUNK_SEC = 10
SAMPLE_RATE = 32000
CHUNK_SAMPLES = CHUNK_SEC * SAMPLE_RATE
PROP_DECREASE = 0.9

def denoise_and_chunk(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    for label in sorted(os.listdir(input_dir)):
        label_path = os.path.join(input_dir, label)
        if not os.path.isdir(label_path):
            continue

        out_label_dir = os.path.join(output_dir, label)
        os.makedirs(out_label_dir, exist_ok=True)

        for file in tqdm(os.listdir(label_path), desc=f"{label}"):
            if not file.endswith(".ogg"):
                continue

            chunk_base = os.path.splitext(file)[0]
            file_path = os.path.join(label_path, file)
            y, sr = sf.read(file_path, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

            n_chunks = math.ceil(len(y) / CHUNK_SAMPLES)
            for ci in range(n_chunks):
                seg = y[ci*CHUNK_SAMPLES:(ci+1)*CHUNK_SAMPLES]
                if len(seg) < CHUNK_SAMPLES:
                    seg = np.pad(seg, (0, CHUNK_SAMPLES - len(seg)), mode='constant')

                den = nr.reduce_noise(y=seg, sr=SAMPLE_RATE, stationary=False, prop_decrease=PROP_DECREASE)
                den /= (np.max(np.abs(den)) + 1e-9)

                chunk_id = f"{chunk_base}_chk{ci}"
                rel_path = os.path.join(label, f"{chunk_id}.ogg")
                out_path = os.path.join(out_label_dir, f"{chunk_id}.ogg")
                sf.write(out_path, den, SAMPLE_RATE, format="OGG", subtype="VORBIS")
                manifest.append({"chunk_id": chunk_id, "audio_path": f"/{rel_path}", "primary_label": label})

    manifest_path = os.path.join(output_dir, "manifest.csv")
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"[INFO] Wrote manifest with {len(manifest)} entries to {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with raw train_audio/")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write denoised chunks")
    args = parser.parse_args()

    denoise_and_chunk(args.input_dir, args.output_dir)
