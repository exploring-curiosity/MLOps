#!/usr/bin/env python3

import os
import math
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import noisereduce as nr
import torch
from multiprocessing.dummy import Pool as ThreadPool
from tqdm.auto import tqdm
from scipy.signal import butter, filtfilt
from panns_inference import AudioTagging
import shutil

# Constants
PANNS_SR = 32000
CHUNK_SEC = 10
CHUNK_SAMPLES = PANNS_SR * CHUNK_SEC
N_FFT = 2048
HOP_LENGTH = 1024
N_MELS = 64
WINDOW_SEC = 1.0
WIN_SAMPLES = int(WINDOW_SEC * PANNS_SR)
LOWCUT = 2000
HIGHCUT = 8000
BAND_ALPHA = 2.0
PROP_DECREASE = 0.9
STATIONARY_NOISE = False
THRESH = 0.5
CHUNKS_PER_BATCH = 64
SAMPLE_FRAC = 1.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class indices for bird presence detection
BIRD_CLASS_IDXS = [
    14,22,27,28,33,34,35,37,40,
    72,73,80,84,
    *range(98,107),108,
    *range(111,122),
    *range(126,133),
    137,361,442,503
]

# Paths
DATA_ROOT = '/workspace/tmp_features'
AUDIO_DIR = os.path.join(DATA_ROOT, 'denoised')
CSV_PATH = os.path.join(DATA_ROOT, 'train.csv')
DEN_DIR = os.path.join(DATA_ROOT, 'denoised')
MEL_DIR = os.path.join(DATA_ROOT, 'mel')
EMB_DIR = os.path.join(DATA_ROOT, 'embeddings')
MEL_AUG_DIR = os.path.join(DATA_ROOT, 'mel_aug')
OUT_DIR = DATA_ROOT

# Create necessary directories
for d in (MEL_DIR, EMB_DIR, MEL_AUG_DIR):
    os.makedirs(d, exist_ok=True)

def calculate_mel_spectrogram(wave, sr=PANNS_SR):
    m = librosa.feature.melspectrogram(
        y=wave, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    return librosa.power_to_db(m, ref=np.max)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def augment_waveform_with_filter(wave, mask, sr,
                                 lowcut=LOWCUT, highcut=HIGHCUT,
                                 alpha=BAND_ALPHA):
    b, a = butter_bandpass(lowcut, highcut, sr, order=4)
    wave_band = filtfilt(b, a, wave)
    wave_aug  = wave.copy()
    win_len   = int(WINDOW_SEC * sr)
    for i, m in enumerate(mask):
        if m:
            start = i * win_len
            end   = min((i+1) * win_len, len(wave))
            wave_aug[start:end] = wave[start:end] + alpha * wave_band[start:end]
    peak = np.max(np.abs(wave_aug))
    if peak > 1.0:
        wave_aug /= peak
    return wave_aug

def get_per_second_embeddings_and_mask(wave, model, device):
    win_len   = PANNS_SR
    n_windows = math.ceil(len(wave) / win_len)
    embs, mask = [], []
    for w in range(n_windows):
        seg = wave[w*win_len:(w+1)*win_len]
        if len(seg) < win_len:
            seg = np.pad(seg, (0, win_len-len(seg)), mode='constant')
        inp = torch.from_numpy(seg).unsqueeze(0).to(device)
        with torch.no_grad():
            clipwise, emb = model.inference(inp)
        probs  = clipwise.squeeze(0).cpu().numpy() if torch.is_tensor(clipwise) else np.squeeze(clipwise,0)
        emb_np = emb.squeeze(0).cpu().numpy()      if torch.is_tensor(emb)      else np.squeeze(emb,0)
        embs.append(emb_np)
        score = probs[BIRD_CLASS_IDXS].max()
        present = score > THRESH or int(np.argmax(probs)) in BIRD_CLASS_IDXS
        mask.append(present)
    return np.stack(embs), np.array(mask, dtype=bool)

# Read and sample metadata
meta = pd.read_csv(CSV_PATH)
sampled = []
for label, grp in meta.groupby('primary_label'):
    n = max(1, int(len(grp) * SAMPLE_FRAC))
    sampled.append(grp.sample(n=n, random_state=42))
meta = pd.concat(sampled, ignore_index=True)

# Create subdirectories
subdirs = {os.path.dirname(f) for f in meta['filename']}
for sub in subdirs:
    if sub:
        for base in (MEL_DIR, EMB_DIR, MEL_AUG_DIR):
            os.makedirs(os.path.join(base, sub), exist_ok=True)

# Phase 1: Denoise and Mel
den_manifest = []
mel_manifest = []

def process_phase1(record):
    fname = record['filename']; label = record['primary_label']
    src_fp = os.path.join(AUDIO_DIR, fname)
    y, sr = sf.read(src_fp, dtype='float32')
    if y.ndim>1: y=y.mean(1)
    if sr!=PANNS_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=PANNS_SR); sr=PANNS_SR

    base = os.path.splitext(os.path.basename(fname))[0]
    n_chunks = math.ceil(len(y)/CHUNK_SAMPLES)
    for ci in range(n_chunks):
        seg = y[ci*CHUNK_SAMPLES:(ci+1)*CHUNK_SAMPLES]
        if len(seg)<CHUNK_SAMPLES:
            seg = np.pad(seg,(0,CHUNK_SAMPLES-len(seg)),'constant')
        # denoise
        den = nr.reduce_noise(y=seg, sr=sr, stationary=False, prop_decrease=PROP_DECREASE)
        den /= (np.max(np.abs(den))+1e-9)

        chunk_id = f"{base}_chk{ci}"
        # save denoised audio
        rel_audio = f"/{label}/{chunk_id}.ogg"
        sf.write(os.path.join(DEN_DIR,label,chunk_id+'.ogg'), den, sr, format='OGG', subtype='VORBIS')
        den_manifest.append({'chunk_id':chunk_id,'audio_path':rel_audio,'primary_label':label})
        # save mel
        mel = calculate_mel_spectrogram(den, sr)
        rel_mel = f"/{label}/{chunk_id}.npz"
        np.savez_compressed(os.path.join(MEL_DIR,label,chunk_id+'.npz'),
                            mel=mel.astype(np.float16), primary_label=label)
        mel_manifest.append({'chunk_id':chunk_id,'mel_path':rel_mel,'primary_label':label})
    return True

records = meta[['filename','primary_label']].to_dict('records')
with ThreadPool(os.cpu_count()) as pool:
    list(tqdm(pool.imap_unordered(process_phase1, records),
              total=len(records), desc="Phase 1: denoise & mel"))

pd.DataFrame(den_manifest).to_csv(os.path.join(DEN_DIR,'manifest.csv'), index=False)
pd.DataFrame(mel_manifest).to_csv(os.path.join(MEL_DIR,'manifest.csv'), index=False)

# Phase 2: Embeddings and Augmentation
panns = AudioTagging(checkpoint_path=None, device=device)
panns.model.eval()

emb_manifest     = []
mel_aug_manifest = []

# Gather denoised audio paths
den_paths = sorted([
    os.path.join(root,f)
    for root,_,files in os.walk(DEN_DIR)
    for f in files if f.endswith('.ogg')
])

for i in tqdm(range(0, len(den_paths), CHUNKS_PER_BATCH), desc="Phase1: multi‑chunk batches"):
    batch_files = den_paths[i:i+CHUNKS_PER_BATCH]
    all_segs, mapping = [], []
    for fp in batch_files:
        y, sr = sf.read(fp, dtype='float32')
        if sr != PANNS_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=PANNS_SR)
        n_win = math.ceil(len(y) / WIN_SAMPLES)
        for w in range(n_win):
            seg = y[w*WIN_SAMPLES:(w+1)*WIN_SAMPLES]
            if len(seg) < WIN_SAMPLES:
                seg = np.pad(seg, (0, WIN_SAMPLES-len(seg)), mode='constant')
            all_segs.append(seg)
            mapping.append((fp, w))

    segs_t = torch.from_numpy(np.stack(all_segs)).to(device)
    with torch.no_grad():
        clipwise_all, emb_all = panns.inference(segs_t)

    probs = clipwise_all.cpu().numpy() if isinstance(clipwise_all, torch.Tensor) else clipwise_all
    embs = emb_all.cpu().numpy() if isinstance(emb_all, torch.Tensor) else emb_all

    by_file = {}
    for (fp, w), p_arr, e_arr in zip(mapping, probs, embs):
        if fp not in by_file:
            by_file[fp] = {'embs': [], 'probs': []}
        by_file[fp]['embs'].append(e_arr)
        by_file[fp]['probs'].append(p_arr)

    for fp, data in by_file.items():
        embs_np  = np.stack(data['embs'])
        probs_np = np.stack(data['probs'])
        mask     = (probs_np[:, BIRD_CLASS_IDXS].max(axis=1) > THRESH) \
           | np.isin(probs_np.argmax(axis=1), BIRD_CLASS_IDXS)
        rel      = os.path.relpath(fp, DEN_DIR)
        sub, fn  = os.path.split(rel)
        cid      = os.path.splitext(fn)[0]
        out_dir  = os.path.join(EMB_DIR, sub)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{cid}_emb.npz")

        np.savez_compressed(
            out_path,
            embedding=embs_np.astype(np.float32),
            mask=mask,
            primary_label=sub
        )

        emb_manifest.append({
            'chunk_id':      cid,
            'emb_path':      f"/{sub}/{cid}_emb.npz",
            'primary_label': sub
        })

pd.DataFrame(emb_manifest).to_csv(
    os.path.join(EMB_DIR, 'manifest.csv'),
    index=False
)

# Phase 3: Augmentation
mask_map = {}
for root, _, files in os.walk(EMB_DIR):
    for f in files:
        if f.endswith('_emb.npz'):
            data = np.load(os.path.join(root, f))
            mask = data['mask']
            rel = os.path.relpath(os.path.join(root, f), EMB_DIR)
            sub, fname = os.path.split(rel)
            chunk_id = fname.replace('_emb.npz','')
            den_path = os.path.join(DEN_DIR, sub, chunk_id + '.ogg')
            mask_map[den_path] = mask
den_paths = list(mask_map.keys())

def process_part2(full_path):
    y, sr = sf.read(full_path, dtype='float32')
    if sr != PANNS_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=PANNS_SR)
        sr = PANNS_SR

    mask = mask_map[full_path]
    y_aug = augment_waveform_with_filter(y, mask, sr)
    y_aug_dn = nr.reduce_noise(
        y=y_aug, sr=sr,
        stationary=STATIONARY_NOISE,
        prop_decrease=PROP_DECREASE
    )
    mel_aug = calculate_mel_spectrogram(y_aug_dn, sr)
    rel = os.path.relpath(full_path, DEN_DIR)
    sub, fname = os.path.split(rel)
    chunk_id = os.path.splitext(fname)[0]
    out_dir  = os.path.join(MEL_AUG_DIR, sub)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, chunk_id + '.npz'),
        mel=mel_aug.astype(np.float16),
        primary_label=sub
    )
    mel_aug_manifest.append({
        'chunk_id': chunk_id,
        'mel_aug_path': f"/{sub}/{chunk_id}.npz",
        'primary_label': sub
    })
    return True

with ThreadPool(os.cpu_count()) as pool:
    list(tqdm(pool.imap_unordered(process_part2, den_paths),
              total=len(den_paths),
              desc="Phase 2: augment+denoise+mel"))

pd.DataFrame(mel_aug_manifest).to_csv(
    os.path.join(MEL_AUG_DIR, 'manifest.csv'),
    index=False
)

                             
 
# ── SPLIT INTO TRAIN / VAL / TEST ─────────────────────────────────────────────
den_df = pd.read_csv(os.path.join(DEN_DIR, 'manifest.csv'))[['chunk_id','audio_path','primary_label']]
mel_df = pd.read_csv(os.path.join(MEL_DIR, 'manifest.csv'))[['chunk_id','mel_path']]
emb_df = pd.read_csv(os.path.join(EMB_DIR, 'manifest.csv'))[['chunk_id','emb_path']]
aug_df = pd.read_csv(os.path.join(MEL_AUG_DIR, 'manifest.csv'))[['chunk_id','mel_aug_path']]

df = (den_df
      .merge(mel_df, on='chunk_id')
      .merge(emb_df, on='chunk_id')
      .merge(aug_df, on='chunk_id'))

df['recording_id'] = df['chunk_id'].str.split('_chk').str[0]
unique_recs = df[['recording_id','primary_label']].drop_duplicates()

# First pick one recording per species
rng = np.random.default_rng(42)
seed_per_species = unique_recs.groupby('primary_label')['recording_id'] \
                              .apply(lambda ids: rng.choice(ids.values, 1)[0]) \
                              .tolist()

all_recs = unique_recs['recording_id'].tolist()
remaining = [r for r in all_recs if r not in seed_per_species]

n_total = len(all_recs)
n_train_target = int(round(0.70 * n_total))
n_additional = max(0, n_train_target - len(seed_per_species))

addl_train = rng.choice(remaining, size=n_additional, replace=False).tolist()
train_recs = set(seed_per_species + addl_train)

remaining_after_train = [r for r in all_recs if r not in train_recs]
n_test = int(round(0.10 * n_total))
test_recs = set(rng.choice(remaining_after_train, size=n_test, replace=False).tolist())

val_recs = set(r for r in all_recs if r not in train_recs and r not in test_recs)

df_train = df[df['recording_id'].isin(train_recs)].reset_index(drop=True)
df_test  = df[df['recording_id'].isin(test_recs)].reset_index(drop=True)
df_val   = df[df['recording_id'].isin(val_recs)].reset_index(drop=True)

for name, split_df in [('train', df_train), ('test', df_test), ('val', df_val)]:
    out = split_df.drop(columns=['recording_id'])
    out.to_csv(os.path.join(OUT_DIR, f'manifest_{name}.csv'), index=False)
    print(f"{name.capitalize()}: {len(out)} chunks, {out['primary_label'].nunique()} species")

# ── OVERSAMPLING ───────────────────────────────────────────────────────────────
groups = (
    df_train[['recording_id','primary_label']]
    .drop_duplicates()
    .groupby('primary_label')['recording_id']
    .apply(list)
    .to_dict()
)

target = int(np.median([len(v) for v in groups.values()]))
oversampled_recs = []
for sp, recs in groups.items():
    oversampled_recs.extend(recs)
    n_extra = max(0, target - len(recs))
    if n_extra:
        extras = rng.choice(recs, size=n_extra, replace=True)
        oversampled_recs.extend(extras.tolist())

frames = []
for rec in oversampled_recs:
    frames.append(df_train[df_train['recording_id'] == rec])
oversampled_df = pd.concat(frames, ignore_index=True)
oversampled_df = oversampled_df.drop(columns=['recording_id'])

out_path = os.path.join(OUT_DIR, 'manifest_train_oversampled.csv')
oversampled_df.to_csv(out_path, index=False)
print(f"Oversampled train manifest: {len(oversampled_df)} chunks")
