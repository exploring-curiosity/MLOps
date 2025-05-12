import os
import torch
import pytest
import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Constants
FEATURE_BASE = "/mnt/BirdCLEF/birdclef_dataset/features_sampled"
EVAL_MANIFEST = os.path.join(FEATURE_BASE, "manifest_test.csv")
TAXONOMY_CSV = os.path.join(FEATURE_BASE, "taxonomy.csv")
NUM_CLASSES = 206
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def taxonomy():
    df = pd.read_csv(TAXONOMY_CSV)
    return sorted(df["primary_label"].astype(str).tolist())


@pytest.fixture(scope="session")
def model_bundle():
    from model_defs import (
        EmbeddingClassifier, RawAudioCNN, get_resnet50_multilabel,
        build_efficientnetb3_lora, MetaMLP
    )

    # Load emb_dim from sample file
    manifest = pd.read_csv(EVAL_MANIFEST)
    emb_path = os.path.join(FEATURE_BASE, "embeddings", manifest.iloc[0].emb_path.lstrip(os.sep))
    emb_dim = np.load(emb_path)["embedding"].shape[1]

    emb_model = EmbeddingClassifier(emb_dim, NUM_CLASSES).to(DEVICE)
    res_model = get_resnet50_multilabel(NUM_CLASSES).to(DEVICE)
    eff_model = build_efficientnetb3_lora(NUM_CLASSES).to(DEVICE)
    raw_model = RawAudioCNN(NUM_CLASSES).to(DEVICE)
    meta_model = MetaMLP(NUM_CLASSES * 4, [1024, 512], 0.3).to(DEVICE)

    # Load weights
    emb_model.load_state_dict(torch.load("models/best_emb_mlp.pt", map_location=DEVICE))
    res_model.load_state_dict(torch.load("models/best_resnet50.pt", map_location=DEVICE))
    eff_model.load_state_dict(torch.load("models/best_effb3_lora.pt", map_location=DEVICE))
    raw_model.load_state_dict(torch.load("models/best_rawcnn.pt", map_location=DEVICE))
    meta_model.load_state_dict(torch.load("models/best_meta_mlp.pt", map_location=DEVICE))

    # Eval mode
    for m in [emb_model, res_model, eff_model, raw_model, meta_model]:
        m.eval()

    return emb_model, res_model, eff_model, raw_model, meta_model


class BirdCLEFDataset(Dataset):
    def __init__(self, manifest_path, base_dir):
        self.df = pd.read_csv(manifest_path)
        self.base = base_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]

        emb_path = os.path.join(self.base, "embeddings", sample.emb_path.lstrip(os.sep))
        emb = np.load(emb_path)["embedding"].mean(axis=0).astype(np.float32)
        emb = torch.tensor(emb).unsqueeze(0)

        mel_aug_path = os.path.join(self.base, "mel_aug", sample.mel_aug_path.lstrip(os.sep))
        mel_aug = np.load(mel_aug_path)["mel"].astype(np.float32)
        mel_aug = torch.tensor(mel_aug).unsqueeze(0).unsqueeze(0)

        mel_path = os.path.join(self.base, "mel", sample.mel_path.lstrip(os.sep))
        mel = np.load(mel_path)["mel"].astype(np.float32)
        mel = torch.tensor(mel).unsqueeze(0).unsqueeze(0)

        wav_path = os.path.join(self.base, "denoised", sample.audio_path.lstrip(os.sep))
        wav, _ = torchaudio.load(wav_path)
        wav = wav.mean(dim=0)
        if wav.size(0) < 160000:
            wav = F.pad(wav, (0, 160000 - wav.size(0)))
        else:
            wav = wav[:160000]
        wav = (wav - wav.mean()) / wav.std().clamp_min(1e-6)
        wav = wav.unsqueeze(0)

        label = int(sample.label)
        return emb, mel_aug, mel, wav, label


@pytest.fixture(scope="session")
def test_loader():
    dataset = BirdCLEFDataset(EVAL_MANIFEST, FEATURE_BASE)
    return DataLoader(dataset, batch_size=32, shuffle=False)


@pytest.fixture(scope="session")
def predictions(model_bundle, test_loader):
    emb_model, res_model, eff_model, raw_model, meta_model = model_bundle

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for emb, mel_aug, mel, wav, label in test_loader:
            emb = emb.to(DEVICE)
            mel_aug = mel_aug.to(DEVICE)
            mel = mel.to(DEVICE)
            wav = wav.to(DEVICE)

            p1 = torch.sigmoid(emb_model(emb))
            p2 = torch.sigmoid(res_model(mel_aug))
            p3 = torch.sigmoid(eff_model(mel))
            p4 = torch.sigmoid(raw_model(wav))

            feat = torch.cat([p1, p2, p3, p4], dim=1)
            logits = meta_model(feat)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = label.numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    return np.array(all_labels), np.array(all_preds)


def test_accuracy(predictions):
    from sklearn.metrics import accuracy_score
    y_true, y_pred = predictions
    acc = accuracy_score(y_true, y_pred)
    print(f"\nBirdCLEF Model Accuracy: {acc:.4f}")
    assert acc > 0.70, "Accuracy below acceptable threshold"
