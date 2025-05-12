from fastapi import FastAPI, UploadFile, File
from model_loader import load_model, predict
from preprocessing.chunk_audio import chunk_audio
from preprocessing.preprocess_features import process_all_chunks
import numpy as np
import os
import torch

app = FastAPI()
model, classes = load_model()

@app.post("/infer-audio")
async def infer_audio(file: UploadFile = File(...)):
    raw_path = f"/workspace/tmp_stream/uploads/{file.filename}"
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    chunk_dir = chunk_audio(raw_path, "/workspace/tmp_stream/chunks")
    process_all_chunks(chunk_dir, "/workspace/tmp_stream/features")

    results = []
    for fname in os.listdir("/workspace/tmp_stream/features"):
        data = np.load(os.path.join("/workspace/tmp_stream/features", fname))
        probs = predict(model, data["mel"])
        pred = torch.argmax(probs).item()
        results.append({"chunk": fname, "label": classes[pred], "confidence": float(probs[pred])})
    return {"results": results}
