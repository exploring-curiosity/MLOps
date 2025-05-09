import argparse
import asyncio
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, Query, BackgroundTasks

# 1) Define your model
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

# 4) FastAPI + lock
app = FastAPI()
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
