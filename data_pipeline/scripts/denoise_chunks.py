import os
import math
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pydub import AudioSegment
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model

# ─── CONFIG ───────────────────────────────────────────────────────────────
DATA_ROOT = "/workspace/tmp_download/birdclef-2025"
AUDIO_DIR = os.path.join(DATA_ROOT, "train_audio")
CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
OUTPUT_DIR = os.path.join(DATA_ROOT, "denoised_chunks")
SAMPLE_RATE = 32000
CHUNK_DURATION = 10  # in seconds
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION

# ─── LOAD METADATA ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── LOAD DEMUCS MODEL ────────────────────────────────────────────────────
print("[INFO] Loading Demucs model...")
model = get_model(name="htdemucs").cpu()
model.eval()

# ─── DENOISE FUNCTION ─────────────────────────────────────────────────────
def denoise_with_demucs(waveform, sr):
    with torch.no_grad():
        ref = waveform.mean(0)
        normalized = (waveform - ref.mean()) / (ref.std() + 1e-9)
        sources = apply_model(model, normalized.unsqueeze(0), samplerate=sr, split=True, progress=False, num_workers=0)[0]
        denoised = sources[0]  # vocals/stems[0] (cleaned audio)
    return denoised.squeeze().numpy()

# ─── PROCESS AUDIO FILES ──────────────────────────────────────────────────
for _, row in tqdm(df.iterrows(), total=len(df), desc="[INFO] Denoising & Chunking"):
    fname = row["filename"]
    label = row["primary_label"]
    src_fp = os.path.join(AUDIO_DIR, fname)

    if not os.path.exists(src_fp):
        print(f"[WARN] Missing file: {src_fp}")
        continue

    # Read and normalize
    waveform, sr = torchaudio.load(src_fp)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        sr = SAMPLE_RATE

    # Apply denoising
    denoised = denoise_with_demucs(waveform[0], sr)

    # Chunk and save
    base_name = os.path.splitext(os.path.basename(fname))[0]
    label_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(label_dir, exist_ok=True)
    n_chunks = math.ceil(len(denoised) / CHUNK_SAMPLES)

    for i in range(n_chunks):
        start = i * CHUNK_SAMPLES
        end = start + CHUNK_SAMPLES
        chunk = denoised[start:end]
        if len(chunk) < CHUNK_SAMPLES:
            pad = CHUNK_SAMPLES - len(chunk)
            chunk = torch.nn.functional.pad(torch.tensor(chunk), (0, pad)).numpy()

        chunk_path = os.path.join(label_dir, f"{base_name}_chk{i}.ogg")
        sf.write(chunk_path, chunk, sr, format='OGG', subtype='VORBIS')

print("[INFO] Denoising and chunking complete.")
