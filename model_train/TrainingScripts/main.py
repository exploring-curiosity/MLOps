import argparse
import asyncio
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, Query, BackgroundTasks, HTTPException
from typing import Literal, Optional
import subprocess
import json
import re

app = FastAPI()

# map train vs retrain scripts
SCRIPT_MAP = {
    "resnet50":   "trainResNet50.py",
    "rawcnn":     "trainRawCNN.py",
    "emb_mlp":    "trainPannsEmb.py",
    "effb3_lora": "trainEffNetB3.py",
    "meta_mlp":   "trainMetaMLP.py",
}

RETRAIN_SCRIPT_MAP = {
    "resnet50":   "retrainResNet50.py",
    "rawcnn":     "retrainRawCNN.py",
    "emb_mlp":    "retrainPannsEmb.py",
    "effb3_lora": "retrainEffNetB3.py",
    "meta_mlp":   "retrainMetaMLP.py",
}

# Per-model default CPU/GPU allocations
MODEL_RESOURCES = {
    "emb_mlp":    {"num_gpus": 0.25, "num_cpus": 8},
    "resnet50":   {"num_gpus": 0.75, "num_cpus": 16},
    "effb3_lora": {"num_gpus": 1.0,  "num_cpus": 16},
    "rawcnn":     {"num_gpus": 1.0,  "num_cpus": 16},
    "meta_mlp":   {"num_gpus": 1.0,  "num_cpus": 16},
}

EFFNET_TUNE_SCRIPT = "./hyperParameterTuneEffNetB3.py"
EMB_TUNE_SCRIPT   = "./hyperParameterTunePannsEmb.py"

@app.post("/train_model")
def submit_train_job(
    model_name: Literal["resnet50", "rawcnn", "emb_mlp", "effb3_lora", "meta_mlp"] = Query(...),
    batch_size:   int   = Query(64),
    lr:           float = Query(1e-4),
    weight_decay: float = Query(1e-4),
    epochs:       int   = Query(20),
    num_gpus:     Optional[float] = Query(None),
    num_cpus:     Optional[int]   = Query(None),
):
    defaults = MODEL_RESOURCES[model_name]
    gpus = num_gpus if num_gpus is not None else defaults["num_gpus"]
    cpus = num_cpus if num_cpus is not None else defaults["num_cpus"]

    script = SCRIPT_MAP[model_name]
    script_path = f"./{script}"

    cmd = [
        "ray", "job", "submit",
        "--format",      "json",
        "--runtime-env", "runtime.json",
        "--entrypoint-num-gpus", str(gpus),
        "--entrypoint-num-cpus", str(cpus),
        "--working-dir", ".",
        "--",
        "python", script_path,
        "--batch_size",   str(batch_size),
        "--lr",           str(lr),
        "--weight_decay", str(weight_decay),
        "--epochs",       str(epochs),
    ]

    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr)

    try:
        resp = json.loads(proc.stdout)
        job_id = resp["id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Ray response: {e}")

    return {"ray_job_id": job_id}


@app.post("/retrain_model")
def submit_retrain_job(
    model_name: Literal["resnet50", "rawcnn", "emb_mlp", "effb3_lora", "meta_mlp"] = Query(..., description="Which model head to retrain"),
    epochs:       int   = Query(20,   description="Number of retraining epochs"),
    num_gpus:     Optional[float] = Query(None, description="Override number of GPUs"),
    num_cpus:     Optional[int]   = Query(None, description="Override number of CPUs"),
    emb_model_name: str   = Query("PannsMLP_PrimaryLabel", description="Registry name for Embedding MLP"),
    raw_model_name: str   = Query("RawAudioCNN",          description="Registry name for RawAudioCNN"),
    res_model_name: str   = Query("ResNet50_MelAug",      description="Registry name for ResNet50"),
    eff_model_name: str   = Query("EfficientNetB3_LoRA",  description="Registry name for EfficientNetB3"),
    meta_model_name: str  = Query("MetaMLP_Supervisor",   description="Registry name for MetaMLP head"),
):
    # resolve cpu/gpu defaults
    defaults = MODEL_RESOURCES[model_name]
    gpus = num_gpus if num_gpus is not None else defaults["num_gpus"]
    cpus = num_cpus if num_cpus is not None else defaults["num_cpus"]

    script = RETRAIN_SCRIPT_MAP[model_name]
    script_path = f"./{script}"

    cmd = [
        "ray", "job", "submit",
        "--format",      "json",
        "--runtime-env", "runtime.json",
        "--entrypoint-num-gpus", str(gpus),
        "--entrypoint-num-cpus", str(cpus),
        "--working-dir", ".",
        "--",
        "python", script_path,
        "--epochs",           str(epochs),
        "--emb_model_name",   emb_model_name,
        "--raw_model_name",   raw_model_name,
        "--res_model_name",   res_model_name,
        "--eff_model_name",   eff_model_name,
        "--meta_model_name",  meta_model_name,
    ]

    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr)

    try:
        resp = json.loads(proc.stdout)
        job_id = resp["id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Ray response: {e}")

    return {"ray_job_id": job_id}


@app.post("/tune_effb3")
def schedule_hyperparam_tune(
    epochs:      int     = Query(20,  ge=1, description="Number of training epochs"),
    num_samples:int     = Query(20,  ge=1, description="Number of hyperparameter trials"),
    num_cpus:    int     = Query(4,   ge=1, description="CPUs per trial"),
    num_gpus:    float   = Query(1.0, ge=0.0, description="GPUs per trial")
):
    """
    Schedule an ASHA hyperparameter tuning run for EfficientNetB3‑LoRA via Ray Tune.
    Returns the Ray Job ID.
    """
    cmd = [
        "ray", "job", "submit",
        "--format",      "json",
        "--runtime-env", "runtime.json",
        "--entrypoint-num-cpus", str(num_cpus),
        "--entrypoint-num-gpus", str(num_gpus),
        "--",
        "python", EFFNET_TUNE_SCRIPT,
        "--epochs",      str(epochs),
        "--num_samples", str(num_samples)
    ]

    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr)

    # parse out the Ray job id
    try:
        resp = json.loads(proc.stdout)
        job_id = resp["id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not parse Ray CLI output: {e}")

    return {"ray_tune_job_id": job_id}


@app.post("/tune_emb_mlp")
def schedule_embedding_tune(
    epochs:       int     = Query(30,  ge=1, description="Number of training epochs"),
    num_samples:  int     = Query(20,  ge=1, description="Number of hyperparameter trials"),
    num_cpus:     int     = Query(4,   ge=1, description="CPUs per trial"),
    num_gpus:     float   = Query(1.0, ge=0.0, description="GPUs per trial")
):
    """
    Schedule a Ray Tune hyperparameter search for the Embedding MLP with ASHA.
    """
    cmd = [
        "ray", "job", "submit",
        "--format",      "json",
        "--runtime-env", "runtime.json",
        "--entrypoint-num-cpus", str(num_cpus),
        "--entrypoint-num-gpus", str(num_gpus),
        "--",
        "python", EMB_TUNE_SCRIPT,
        "--epochs",      str(epochs),
        "--num_samples", str(num_samples)
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=e.stderr)

    try:
        resp = json.loads(proc.stdout)
        job_id = resp["id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse Ray response: {e}")

    return {"ray_tune_job_id": job_id}

@app.get("/status")
def get_job_status(job_id: str = Query(...)):
    try:
        status = subprocess.check_output(
            ["ray", "job", "status", job_id],
            stderr=subprocess.STDOUT, text=True
        ).strip()
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch status: {e.output}")

    result = {"ray_job_id": job_id, "status": status}

    if status == "SUCCEEDED":
        try:
            logs = subprocess.check_output(
                ["ray", "job", "logs", job_id],
                stderr=subprocess.DEVNULL, text=True
            )
            m = re.search(r"MLFLOW_RUN_ID=(\S+)", logs)
            if m:
                result["mlflow_run_id"] = m.group(1)
        except subprocess.CalledProcessError:
            pass

    return result









# DUmmy for simulation of training.
class RawAudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # initial downsample
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7)
        self.bn1   = nn.BatchNorm1d(16)
        self.pool  = nn.MaxPool1d(4)
        # deeper layers
        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7)
        self.bn2   = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7)
        self.bn3   = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64,128,kernel_size=15, stride=2,padding=7)
        self.bn4   = nn.BatchNorm1d(128)
        # global pooling & head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc          = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, T] → [B,1,T]
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x).squeeze(-1)  # [B,128]
        return self.fc(x)


# 2) MLflow client & experiment setup
client = MlflowClient()

EXP_NAME = "RawCNN"
exp = client.get_experiment_by_name(EXP_NAME)
exp_id = exp.experiment_id if exp else client.create_experiment(EXP_NAME)

# 3) Constants
MODEL_PATH = "./best_rawcnn.pt"
NUM_CLS    = 206

# 4) lock
training_lock = asyncio.Lock()


# 5) Background worker that holds the lock until done
async def _background_train(run_id: str):
    # re‑attach to the run  
    with mlflow.start_run(run_id=run_id, experiment_id=exp_id):
        # your “long” work
        await asyncio.sleep(60)

        # load the state_dict + instantiate the module
        model = RawAudioCNN(num_classes=NUM_CLS)
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        # log it
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model"
        )
    # lock.release() will unblock future /train calls
    training_lock.release()


# 6) Endpoint
@app.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    model_name: str = Query(...),
    data_source: str = Query(...)
):
    # If someone is already training, immediately tell them so
    if training_lock.locked():
        return {"status": "in_progress"}


    print("hello world")
    # Acquire the lock *and hold it* until background task calls release()
    await training_lock.acquire()

    # 1) Create the MLflow run synchronously
    run = client.create_run(
        experiment_id=exp_id,
        tags={"model_name": model_name, "data_source": data_source}
    )
    run_id = run.info.run_id

    # 2) Schedule the heavy work in the background
    background_tasks.add_task(_background_train, run_id)

    # 3) Build the UI link & return immediately
    uri = mlflow.get_tracking_uri().rstrip("/")
    return {
        "status": "started",
        "run_id": run_id,
        "url": f"{uri}/#/experiments/{exp_id}/runs/{run_id}"
    }









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
