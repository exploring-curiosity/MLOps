# === Step 6: Train Embedding Classifier (MLP) with PyTorch Lightning ===
# Input: Raw .ogg audio files.
# Process: Load audio -> Chunk (5sec) -> Mel Spec (GPU) -> PANNs Embedding (GPU) -> MLP Classifier.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L # Use 'lightning' if using v2.0+
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import time
import os
from sklearn.metrics import label_ranking_average_precision_score
import subprocess
import sys
import warnings
import ast
import traceback
import librosa
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
# Need torchaudio for Mel spectrogram generation on GPU
try:
    import torchaudio
    import torchaudio.transforms as T
    torchaudio_available = True
except ImportError:
    print("Warning: torchaudio not found. Mel spectrogram generation might be slow or fail.")
    torchaudio_available = False

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")
warnings.filterwarnings("ignore", message="numba.core.errors.NumbaDeprecationWarning")


print("\n--- Running Step 6: Train Embedding Classifier (MLP) with PyTorch Lightning ---")

# --- Environment Setup ---

# 1. Install/Check PyTorch Lightning
try:
    import pytorch_lightning
    print(f"PyTorch Lightning Version: {pytorch_lightning.__version__}")
except ImportError:
    print("Installing pytorch-lightning...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytorch-lightning"])
    import pytorch_lightning as L
    print(f"PyTorch Lightning Version: {pytorch_lightning.__version__}")

# 2. Install/Check torchlibrosa (PANNs dependency)
try:
    import torchlibrosa
    print("torchlibrosa already installed.")
except ImportError:
    print("Installing torchlibrosa...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torchlibrosa"])
        import torchlibrosa
        print("torchlibrosa installed successfully.")
    except Exception as e:
        print(f"Error installing torchlibrosa: {e}. PANNs setup might fail.")

# 3. Clone PANNs Repo & Add Path
panns_repo_path_relative = 'audioset_tagging_cnn'
panns_pytorch_path = os.path.join(panns_repo_path_relative, 'pytorch')
panns_models_file = os.path.join(panns_pytorch_path, 'models.py')
panns_code_available = False
if not os.path.isdir(panns_repo_path_relative):
    print(f"Cloning PANNs repository into ./{panns_repo_path_relative}...")
    try:
        ipython = get_ipython() # Check if in notebook
        if ipython: ipython.system(f"git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git {panns_repo_path_relative}")
        else: os.system(f"git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git {panns_repo_path_relative}")
        if not os.path.isdir(panns_repo_path_relative): print("Error: Cloning failed.")
        else: print("Cloning successful.")
    except NameError: # Not in IPython
        os.system(f"git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git {panns_repo_path_relative}")
        if not os.path.isdir(panns_repo_path_relative): print("Error: Cloning failed.")
        else: print("Cloning successful.")
    except Exception as e: print(f"Error during git clone: {e}")

if os.path.isdir(panns_pytorch_path) and os.path.isfile(panns_models_file):
    if panns_pytorch_path not in sys.path:
        print(f"Adding PANNs path to sys.path: {panns_pytorch_path}")
        sys.path.append(panns_pytorch_path) # Use relative path as before
    else: print("PANNs path already in sys.path.")
    panns_code_available = True
else: print(f"Error: PANNs pytorch directory or models.py not found at {panns_pytorch_path}")


# --- Configuration ---
class CFG:
    # General
    RANDOM_STATE = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Lightning handles device placement

    # Paths
    DATA_DIR = '/kaggle/input/birdclef-2025'
    TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
    TAXONOMY_CSV = os.path.join(DATA_DIR, 'taxonomy.csv')
    TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, 'train_audio')
    OUTPUT_MODEL_DIR = '/kaggle/working/models' # Base directory for models
    PANNS_CHECKPOINT = '/kaggle/input/pannscnn14/panns_cnn14.pth'

    # Data Parameters
    SAMPLE_RATE = 32000
    # *** Chunk duration for embedding extraction (use 5s for consistency with Perch) ***
    CHUNK_DURATION_SEC = 5
    CHUNK_SAMPLES = CHUNK_DURATION_SEC * SAMPLE_RATE
    # Mel parameters needed for PANNs input
    N_MELS = 128 # Use 128 bins consistent with Step 1 regeneration
    N_FFT = 1024
    HOP_LENGTH = 320
    FMIN = 50
    FMAX = 14000
    # PANNs Embedding Dim
    PANNS_EMBED_DIM = 2048

    # DataLoader
    BATCH_SIZE = 64 # MLP can often handle larger batches
    NUM_WORKERS = 2
    VALIDATION_SPLIT = 0.2
    SAMPLE_FRACTION = 0.10 # Use the same 10% sample as Step 1

    # Model & Training Parameters
    NUM_CLASSES = 206 # Will be updated
    EPOCHS_MLP = 15 # MLP might need more epochs
    LOSS_FN = 'BCEWithLogitsLoss'
    # Embedding Model Specific (This Step)
    EMBEDDING_MODEL_HIDDEN_DIMS = [1024, 512]
    EMBEDDING_MODEL_DROPOUT = 0.3
    EMBEDDING_MODEL_LR = 1e-3
    EMBEDDING_MODEL_WD = 1e-5
    EMBEDDING_MODEL_SCHEDULER = 'CosineAnnealingLR'
    EMBEDDING_MODEL_SCHEDULER_PARAMS = {'T_max': EPOCHS_MLP, 'eta_min': 1e-6} # Use MLP epochs
    EMBEDDING_MODEL_SAVED_NAME = "embedding_mlp_lightning_best.ckpt" # Lightning checkpoint name

    # Trainer Config
    PRECISION = "16-mixed" if torch.cuda.is_available() else "32-true" # Use mixed precision on GPU
    ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
    DEVICES = 1

# --- Helper function to update CFG ---
def update_cfg_num_classes(cfg_class):
    try:
        taxonomy_df_temp = pd.read_csv(cfg_class.TAXONOMY_CSV)
        official_species_column = 'primary_label'
        if official_species_column not in taxonomy_df_temp.columns: raise KeyError(f"Col missing")
        num_classes = taxonomy_df_temp[official_species_column].nunique()
        if cfg_class.NUM_CLASSES != num_classes: cfg_class.NUM_CLASSES = num_classes
        print(f"Set CFG.NUM_CLASSES={cfg_class.NUM_CLASSES}")
        cfg_class.SPECIES_LIST = sorted(taxonomy_df_temp[official_species_column].astype(str).unique())
    except Exception as e: print(f"Warning: Error updating CFG num_classes: {e}")

# --- Initial CFG setup ---
update_cfg_num_classes(CFG)
os.makedirs(CFG.OUTPUT_MODEL_DIR, exist_ok=True)
print(f"Target Device (Trainer): {CFG.ACCELERATOR} {CFG.DEVICES}")
print(f"Number of classes set to: {CFG.NUM_CLASSES}")

# --- PANNs Extractor Definition & Loading ---
# (Define classes and load extractor instance here, needed by Dataset)
panns_extractor_model = None
if panns_code_available:
    try:
        from models import Cnn14
        # Define load_panns_cnn14 and PANNsMelFeatureExtractor classes here...
        # (Code omitted for brevity - assume they are defined as in previous steps)
        class PANNsMelFeatureExtractor(nn.Module):
             def __init__(self, full_model):
                 super().__init__(); self.bn0=full_model.bn0; self.conv_block1=full_model.conv_block1; self.conv_block2=full_model.conv_block2; self.conv_block3=full_model.conv_block3; self.conv_block4=full_model.conv_block4; self.conv_block5=full_model.conv_block5; self.conv_block6=full_model.conv_block6; self.fc1=full_model.fc1
             def forward(self, x): # Input: [B, 1, M, T]
                 x=x.permute(0,1,3,2); x=x.transpose(1,3); x=self.bn0(x); x=x.transpose(1,3)
                 x=self.conv_block1(x, pool_size=(2,2), pool_type='avg'); x=self.conv_block2(x, pool_size=(2,2), pool_type='avg'); x=self.conv_block3(x, pool_size=(2,2), pool_type='avg'); x=self.conv_block4(x, pool_size=(2,2), pool_type='avg'); x=self.conv_block5(x, pool_size=(2,2), pool_type='avg'); x=self.conv_block6(x, pool_size=(1,1), pool_type='avg')
                 x=torch.mean(x,dim=3); (x,_)=torch.max(x,dim=2); return self.fc1(x)
        def load_panns_cnn14(checkpoint_path):
             model = Cnn14(sample_rate=CFG.SAMPLE_RATE, window_size=CFG.N_FFT, hop_size=CFG.HOP_LENGTH, mel_bins=CFG.N_MELS, fmin=CFG.FMIN, fmax=CFG.FMAX, classes_num=527)
             if not os.path.exists(checkpoint_path): raise FileNotFoundError()
             checkpoint=torch.load(checkpoint_path, map_location='cpu'); state_dict=checkpoint.get('model', checkpoint)
             model_dict=model.state_dict(); state_dict={k:v for k,v in state_dict.items() if k in model_dict and v.shape==model_dict[k].shape}; model_dict.update(state_dict)
             model.load_state_dict(model_dict, strict=False); return model

        panns_base_model = load_panns_cnn14(CFG.PANNS_CHECKPOINT)
        # Load extractor onto the DEVICE Lightning will use
        panns_extractor_model = PANNsMelFeatureExtractor(panns_base_model).to(CFG.DEVICE)
        panns_extractor_model.eval() # Set to eval mode
        for param in panns_extractor_model.parameters(): # Freeze extractor
            param.requires_grad = False
        del panns_base_model; torch.cuda.empty_cache()
        print("PANNs extractor ready and frozen.")
    except Exception as e: print(f"PANNs setup failed: {e}")
else:
    print("PANNs code not available. Cannot create extractor.")

# --- Torchaudio Transforms ---
mel_spectrogram_transform = None
amplitude_to_db_transform = None
if torchaudio_available:
    try:
        mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=CFG.SAMPLE_RATE, n_fft=CFG.N_FFT, hop_length=CFG.HOP_LENGTH,
            n_mels=CFG.N_MELS, f_min=CFG.FMIN, f_max=CFG.FMAX, power=2.0,
            norm='slaney', mel_scale="htk").to(CFG.DEVICE) # Put transforms on device
        amplitude_to_db_transform = T.AmplitudeToDB(stype='power', top_db=80.0).to(CFG.DEVICE)
        print("Torchaudio transforms initialized on device.")
    except Exception as e: print(f"Warning: Could not initialize torchaudio transforms: {e}")


# --- PyTorch Lightning DataModule ---
class BirdEmbeddingDataModule(L.LightningDataModule):
    def __init__(self, cfg, panns_extractor, mel_transform, db_transform):
        super().__init__()
        self.cfg = cfg
        self.panns_extractor = panns_extractor # Pre-loaded and frozen extractor
        self.mel_transform = mel_transform
        self.db_transform = db_transform
        self.train_df = None
        self.val_df = None
        self.mlb = None
        self.filename_to_labels = {}

    def prepare_data(self):
        # Download or verify data exists (usually done outside Lightning)
        pass

    def setup(self, stage=None):
        # Load metadata, create labels, split data
        try:
            full_train_df = pd.read_csv(self.cfg.TRAIN_CSV)
            taxonomy_df = pd.read_csv(self.cfg.TAXONOMY_CSV)
            official_species_column = 'primary_label'
            species_list = sorted(taxonomy_df[official_species_column].astype(str).unique())
            if len(species_list) != self.cfg.NUM_CLASSES: print("Warning: Class count mismatch in DataModule setup")

            species_to_int = {s: i for i, s in enumerate(species_list)}
            def parse_labels(l_str):
                try: return [str(l) for l in ast.literal_eval(l_str) if str(l) in species_list]
                except: return []

            full_train_df['primary_label_str'] = full_train_df['primary_label'].astype(str)
            full_train_df['secondary_labels_list'] = full_train_df['secondary_labels'].apply(parse_labels)

            self.mlb = MultiLabelBinarizer(classes=species_list)
            labels_all = full_train_df.apply(lambda r: [r['primary_label_str']] + r['secondary_labels_list'], axis=1)
            labels_all_str = [[str(i) for i in s] for s in labels_all]
            labels_encoded = self.mlb.fit_transform(labels_all_str)
            self.filename_to_labels = {r['filename']: labels_encoded[i] for i, r in full_train_df.iterrows()}

            # Apply Sampling Logic
            valid_species_labels = set(species_list)
            stratify_df = full_train_df[full_train_df['primary_label_str'].isin(valid_species_labels)].copy()
            if stratify_df.empty: raise ValueError("No valid entries for stratification.")

            try:
                sampled_indices, _ = train_test_split(
                    stratify_df.index, test_size=1.0 - self.cfg.SAMPLE_FRACTION,
                    stratify=stratify_df['primary_label_str'], random_state=self.cfg.RANDOM_STATE
                )
                train_df_sampled = full_train_df.loc[sampled_indices].reset_index(drop=True)
            except ValueError: # Fallback for stratification failure
                train_df_sampled = stratify_df.sample(frac=self.cfg.SAMPLE_FRACTION, random_state=self.cfg.RANDOM_STATE).reset_index(drop=True)

            # Create DataFrame of files with valid labels from the sample
            audio_files_data = []
            for index, row in train_df_sampled.iterrows():
                filename = row['filename']
                if filename in self.filename_to_labels:
                     labels = self.filename_to_labels[filename]
                     if np.sum(labels) > 0: audio_files_data.append({'filename': filename, 'labels': labels})
            audio_df_sampled = pd.DataFrame(audio_files_data)

            # Split sampled data into train/val
            self.train_df, self.val_df = train_test_split(
                audio_df_sampled, test_size=self.cfg.VALIDATION_SPLIT, random_state=self.cfg.RANDOM_STATE
            )
            print(f"DataModule setup: Train files={len(self.train_df)}, Val files={len(self.val_df)}")

        except Exception as e:
            print(f"Error during DataModule setup: {e}")
            traceback.print_exc()
            self.train_df = pd.DataFrame() # Ensure empty df on error
            self.val_df = pd.DataFrame()

    def _create_dataset(self, df):
        if df is None or df.empty: return None
        return BirdAudioEmbeddingDataset(
            dataframe=df,
            audio_dir=self.cfg.TRAIN_AUDIO_DIR,
            sr=self.cfg.SAMPLE_RATE,
            chunk_samples=self.cfg.CHUNK_SAMPLES,
            num_classes=self.cfg.NUM_CLASSES,
            panns_extractor=self.panns_extractor,
            mel_transform=self.mel_transform,
            db_transform=self.db_transform,
            device=self.cfg.DEVICE # Pass device for transforms/extractor
        )

    def train_dataloader(self):
        train_dataset = self._create_dataset(self.train_df)
        if train_dataset is None: return None
        return DataLoader(
            train_dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=True,
            num_workers=self.cfg.NUM_WORKERS, pin_memory=False, drop_last=True # Pin memory handled by Lightning
        )

    def val_dataloader(self):
        val_dataset = self._create_dataset(self.val_df)
        if val_dataset is None: return None
        return DataLoader(
            val_dataset, batch_size=self.cfg.BATCH_SIZE * 2, shuffle=False,
            num_workers=self.cfg.NUM_WORKERS, pin_memory=False, drop_last=False
        )

# --- PyTorch Dataset for Audio -> Embedding ---
class BirdAudioEmbeddingDataset(Dataset):
    def __init__(self, dataframe, audio_dir, sr, chunk_samples, num_classes,
                 panns_extractor, mel_transform, db_transform, device):
        self.dataframe = dataframe
        self.audio_dir = audio_dir
        self.sr = sr
        self.chunk_samples = chunk_samples
        self.num_classes = num_classes
        self.panns_extractor = panns_extractor # Expects frozen extractor on correct device
        self.mel_transform = mel_transform     # Expects transform on correct device
        self.db_transform = db_transform       # Expects transform on correct device
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def _get_mel_gpu(self, audio_np):
        if self.mel_transform is None: return None
        try:
            audio_tensor = torch.tensor(audio_np, dtype=torch.float32).to(self.device)
            mel_spec_power = self.mel_transform(audio_tensor)
            mel_spec_db = self.db_transform(mel_spec_power)
            # Return tensor directly (shape [M, T]) - will add batch/channel later if needed by extractor
            return mel_spec_db
        except Exception: return None

    def _get_embedding(self, mel_tensor):
        if self.panns_extractor is None or mel_tensor is None:
            return torch.zeros(CFG.PANNS_EMBED_DIM, dtype=torch.float32, device=self.device) # Return on correct device
        try:
            # PANNs extractor expects [B, 1, M, T]
            mel_tensor_unsqueezed = mel_tensor.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                embedding = self.panns_extractor(mel_tensor_unsqueezed)
            return embedding.squeeze(0) # Return [Embed_dim] tensor on device
        except Exception as e:
            # print(f"Warn embed extract: {e}") # Debug
            return torch.zeros(CFG.PANNS_EMBED_DIM, dtype=torch.float32, device=self.device)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        filename = row['filename']
        labels = row['labels']
        file_path = os.path.join(self.audio_dir, filename)

        try:
            wav, loaded_sr = librosa.load(file_path, sr=self.sr, mono=True)
            if loaded_sr != self.sr: print(f"Warning: SR mismatch {filename}")

            if len(wav) >= self.chunk_samples:
                start = np.random.randint(0, len(wav) - self.chunk_samples + 1)
                wav_chunk = wav[start : start + self.chunk_samples]
            else:
                padding = self.chunk_samples - len(wav)
                wav_chunk = np.pad(wav, (0, padding), 'constant')

            # Generate Mel on GPU
            mel_tensor = self._get_mel_gpu(wav_chunk) # Shape [M, T] on device

            # Generate Embedding on GPU from Mel tensor
            embedding_tensor = self._get_embedding(mel_tensor) # Shape [Embed_dim] on device

            label_tensor = torch.tensor(labels, dtype=torch.float32) # Labels on CPU initially

            # Return embedding and labels (move labels to device in training step)
            return embedding_tensor, label_tensor

        except Exception as e:
            print(f"Error loading/processing audio {file_path}: {e}")
            return self._create_dummy_data()

    def _create_dummy_data(self):
        dummy_embed = torch.zeros(CFG.PANNS_EMBED_DIM, dtype=torch.float32)
        dummy_label = torch.zeros(self.num_classes, dtype=torch.float32)
        return dummy_embed, dummy_label


# --- PyTorch Lightning Module (MLP Classifier) ---
class EmbeddingMLPClassifier(L.LightningModule):
    def __init__(self, input_dim, num_classes, hidden_dims, dropout, lr, wd, epochs):
        super().__init__()
        self.save_hyperparameters() # Saves init args to hparams

        layers = []
        last_dim = self.hparams.input_dim
        for hidden_dim in self.hparams.hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.hparams.dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, self.hparams.num_classes))
        self.mlp = nn.Sequential(*layers)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        x, y = batch # Expecting (embedding, label) from DataLoader
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Calculate LRAP for logging
        preds = torch.sigmoid(logits)
        # Ensure numpy conversion happens on CPU
        y_np = y.cpu().numpy()
        preds_np = preds.cpu().numpy()
        try:
            mask = ~np.isnan(preds_np).any(axis=1) & ~np.isinf(preds_np).any(axis=1) & (np.sum(y_np, axis=1) > 0)
            val_lrap = label_ranking_average_precision_score(y_np[mask], preds_np[mask]) if np.sum(mask) > 0 else 0.0
        except ValueError:
            val_lrap = 0.0 # Handle potential errors if no positive labels etc.
        self.log('val_lrap', val_lrap, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd
        )
        # Example scheduler (Cosine Annealing)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.epochs, # Use total epochs
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # Call scheduler step every epoch
                "frequency": 1,
            },
        }

# --- Setup and Run Training ---
if panns_extractor_model is not None and torchaudio_available:
    # Instantiate DataModule
    data_module = BirdEmbeddingDataModule(
        cfg=CFG,
        panns_extractor=panns_extractor_model,
        mel_transform=mel_spectrogram_transform,
        db_transform=amplitude_to_db_transform
    )

    # Instantiate LightningModule
    lit_mlp_model = EmbeddingMLPClassifier(
        input_dim=CFG.PANNS_EMBED_DIM,
        num_classes=CFG.NUM_CLASSES,
        hidden_dims=CFG.EMBEDDING_MODEL_HIDDEN_DIMS,
        dropout=CFG.EMBEDDING_MODEL_DROPOUT,
        lr=CFG.EMBEDDING_MODEL_LR,
        wd=CFG.EMBEDDING_MODEL_WD,
        epochs=CFG.EPOCHS_MLP # Pass total epochs for scheduler
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CFG.OUTPUT_MODEL_DIR,
        filename=CFG.EMBEDDING_MODEL_SAVED_NAME.replace(".ckpt", "-{epoch:02d}-{val_lrap:.4f}"),
        save_top_k=1,
        verbose=True,
        monitor='val_lrap', # Monitor validation LRAP
        mode='max'
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_lrap', # Monitor validation LRAP
        patience=3,         # Stop after 3 epochs with no improvement
        verbose=True,
        mode='max'
    )
    logger = CSVLogger(save_dir=os.path.join(CFG.OUTPUT_MODEL_DIR, "logs"), name="embedding_mlp")


    # Trainer
    trainer = L.Trainer(
        accelerator=CFG.ACCELERATOR,
        devices=CFG.DEVICES,
        max_epochs=CFG.EPOCHS_MLP,
        precision=CFG.PRECISION,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        # limit_train_batches=0.1, # Uncomment for quick testing
        # limit_val_batches=0.1,   # Uncomment for quick testing
    )

    # Train the model
    print("\n--- Starting Embedding MLP Training with PyTorch Lightning ---")
    try:
        trainer.fit(lit_mlp_model, datamodule=data_module)
        print("Training finished.")
        print(f"Best model checkpoint saved to: {checkpoint_callback.best_model_path}")
    except Exception as train_e:
        print(f"An error occurred during training: {train_e}")
        traceback.print_exc()

else:
    print("Skipping MLP training because PANNs extractor or torchaudio is not available.")


print("\n--- Step 6: Train Embedding Classifier Complete ---")