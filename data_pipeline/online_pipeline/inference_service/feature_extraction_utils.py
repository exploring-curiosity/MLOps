import torchaudio.transforms as T
import torch
import numpy as np

def compute_mel(waveform, sr):
    mel = T.MelSpectrogram(sample_rate=sr, n_mels=128)(waveform)
    return mel.log2().clamp(min=-10).numpy()

def compute_pann_embedding(waveform):
    return np.random.rand(2048)  # Placeholder: replace with actual PANN embedding

def augment_mel(mel):
    mel = torch.tensor(mel).float()
    mel += 0.05 * torch.randn_like(mel)
    return mel.numpy()
