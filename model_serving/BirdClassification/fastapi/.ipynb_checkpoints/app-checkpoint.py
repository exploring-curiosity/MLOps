from fastapi import FastAPI, UploadFile, File
from typing import List
import torchaudio
import torch
import torch.nn.functional as F
import numpy as np
import os
import io
import time
from tritonclient.http import InferenceServerClient, InferInput
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, REGISTRY

# === Constants ===
SAMPLE_RATE = 16000
RAW_MODEL_INPUT_LEN = 320000
TRITON_URL = os.getenv("TRITON_SERVER_URL", "triton_server:8000").replace("http://", "")
CLASSES = [f"class_{i}" for i in range(206)]

# === Initialize FastAPI ===
app = FastAPI(
    title="BirdCLEF Triton API",
    description="Upload .wav/.ogg files for bird classification using ONNX + Triton",
    version="1.0.0"
)

# === Prometheus Instrumentation ===
Instrumentator().instrument(app).expose(app)

# === Metric Cache ===
_metric_cache = {}

def get_or_create_metric(metric_class, name, documentation, *args, **kwargs):
    if name in _metric_cache:
        return _metric_cache[name]
    try:
        metric = metric_class(name, documentation, *args, **kwargs)
        _metric_cache[name] = metric
        return metric
    except ValueError as e:
        existing = REGISTRY._names_to_collectors.get(name)
        if existing:
            _metric_cache[name] = existing
            return existing
        raise e

# === Prometheus Metrics ===
confidence_histogram = get_or_create_metric(
    Histogram,
    "prediction_confidence",
    "Histogram of model prediction confidence",
    buckets=[0.1 * i for i in range(11)]
)

confidence_sum = get_or_create_metric(
    Counter,
    "prediction_confidence_sum",
    "Sum of confidence scores"
)

confidence_count = get_or_create_metric(
    Counter,
    "prediction_confidence_count",
    "Count of predictions"
)

predicted_class_total = get_or_create_metric(
    Counter,
    "predicted_class_total",
    "Number of times each class was predicted",
    ["class_name"]
)

# === Triton Client ===
triton_client = InferenceServerClient(url=TRITON_URL)

# === Inference Helper ===
def infer_model(model_name: str, input_name: str, input_array: np.ndarray, output_name: str):
    infer_input = InferInput(input_name, input_array.shape, "FP32")
    infer_input.set_data_from_numpy(input_array)
    result = triton_client.infer(model_name=model_name, inputs=[infer_input])
    return result.as_numpy(output_name)

# === Preprocessing ===
def preprocess_audio(wav_bytes: bytes):
    waveform, sr = torchaudio.load(io.BytesIO(wav_bytes))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform.mean(dim=0) if waveform.dim() > 1 else waveform
    waveform = F.pad(waveform, (0, max(0, RAW_MODEL_INPUT_LEN - waveform.shape[0])))
    waveform = (waveform - waveform.mean()) / waveform.std().clamp_min(1e-6)
    return waveform[:RAW_MODEL_INPUT_LEN].unsqueeze(0).numpy().astype(np.float32)

def get_mel_and_embedding(waveform_tensor):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=64
    )
    mel = torch.log1p(mel_transform(torch.tensor(waveform_tensor)))
    mel_padded = torch.zeros((1, 64, 313))
    mel_len = min(mel.shape[-1], 313)
    mel_padded[:, :, :mel_len] = mel[:, :, :mel_len]
    mel_np = mel_padded.unsqueeze(0).numpy().astype(np.float32)

    emb = mel_padded.view(1, -1)[:, :2048]
    if emb.shape[1] < 2048:
        emb = F.pad(emb, (0, 2048 - emb.shape[1]))
    emb_np = emb.numpy().astype(np.float32)
    return mel_np, emb_np

# === Multi-file prediction route ===
@app.post("/predict")
async def predict(files: List[UploadFile] = File(..., description="Upload multiple .wav or .ogg audio files")):
    results = []

    for file in files:
        try:
            total_start = time.perf_counter()
            audio_bytes = await file.read()

            # Preprocessing
            t0 = time.perf_counter()
            wav_input = preprocess_audio(audio_bytes)
            mel_input, emb_input = get_mel_and_embedding(torch.tensor(wav_input))
            preprocessing_ms = (time.perf_counter() - t0) * 1000

            # Triton Inference
            t1 = time.perf_counter()
            p1 = infer_model("embedding_classifier_opt", "embedding_input", emb_input, "embedding_output")
            embedding_ms = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            p2 = infer_model("resnet50_multilabel_opt", "mel_aug_input", mel_input, "resnet_output")
            resnet_ms = (time.perf_counter() - t2) * 1000

            t3 = time.perf_counter()
            p3 = infer_model("efficientnet_b3_lora_opt", "mel_input", mel_input, "effnet_output")
            effnet_ms = (time.perf_counter() - t3) * 1000

            t4 = time.perf_counter()
            p4 = infer_model("raw_audio_cnn_opt", "wav_input", wav_input, "raw_output")
            rawaudio_ms = (time.perf_counter() - t4) * 1000

            # Fusion
            t5 = time.perf_counter()
            fused_input = np.concatenate([p1, p2, p3, p4], axis=1).astype(np.float32)
            final_pred = infer_model("meta_mlp_opt", "fusion_input", fused_input, "meta_output")
            fusion_ms = (time.perf_counter() - t5) * 1000

            # Final Prediction
            probs = 1 / (1 + np.exp(-final_pred[0]))
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            total_latency_ms = (time.perf_counter() - total_start) * 1000

            # Prometheus Metrics
            confidence_histogram.observe(confidence)
            confidence_sum.inc(confidence)
            confidence_count.inc()
            predicted_class_total.labels(class_name=CLASSES[idx]).inc()

            results.append({
                "filename": file.filename,
                "predicted_label": CLASSES[idx],
                "probability": confidence,
                "latency_ms": round(total_latency_ms, 2),
                "preprocessing_ms": round(preprocessing_ms, 2),
                "embedding_ms": round(embedding_ms, 2),
                "resnet_ms": round(resnet_ms, 2),
                "effnet_ms": round(effnet_ms, 2),
                "rawaudio_ms": round(rawaudio_ms, 2),
                "fusion_ms": round(fusion_ms, 2)
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return results
