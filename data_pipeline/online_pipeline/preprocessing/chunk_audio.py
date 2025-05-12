import librosa
import os
import soundfile as sf
from pathlib import Path
import numpy as np

def chunk_audio(file_path, out_dir, chunk_len=5.0, overlap=0.5, sr=32000):
    y, sr = librosa.load(file_path, sr=sr)
    step = chunk_len * (1 - overlap)
    os.makedirs(out_dir, exist_ok=True)
    for i, start in enumerate(np.arange(0, len(y)/sr - chunk_len, step)):
        start_s = int(start * sr)
        end_s = int((start + chunk_len) * sr)
        chunk = y[start_s:end_s]
        fname = os.path.join(out_dir, f"{Path(file_path).stem}_chunk{i}.ogg")
        sf.write(fname, chunk, sr)
    return out_dir
