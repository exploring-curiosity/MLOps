import os
import torchaudio
import numpy as np
from inference_service.feature_extraction_utils import compute_mel, compute_pann_embedding, augment_mel

def process_all_chunks(chunk_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(chunk_dir):
        if not fname.endswith(".ogg"): continue
        path = os.path.join(chunk_dir, fname)
        waveform, sr = torchaudio.load(path)
        mel = compute_mel(waveform, sr)
        pann = compute_pann_embedding(waveform)
        aug = augment_mel(mel)
        np.savez(os.path.join(out_dir, f"{fname}.npz"), mel=mel, pann=pann, aug_mel=aug)
