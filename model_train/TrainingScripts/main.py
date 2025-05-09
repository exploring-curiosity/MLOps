import argparse
import asyncio
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Query

app = FastAPI()

BEST_CKPT   = "./best_rawcnn.pt"
NUM_CLS     = 206
MODEL_NAME  = "RawCNN"
training_lock = asyncio.Lock()


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

@app.post("/train")
async def train_model(
    model_name: str = Query(...),
    data_source: str = Query(...),
):
    if training_lock.locked():
        return {"status": "in_progress"}

    async with training_lock, mlflow.start_run() as run:
        # simulate your work
        await asyncio.sleep(60)

        # 1) instantiate 
        model = RawAudioCNN(num_classes=NUM_CLS)

        # 2) load the weights you saved
        state = torch.load(BEST_CKPT, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        # 3) log the nn.Module itself
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",           # goes under mlruns/.../artifacts/model
            registered_model_name=MODEL_NAME # optional, to register in the Model Registry
        )

        # build your URL as before
        run_id       = run.info.run_id
        exp_id       = run.info.experiment_id
        uri          = mlflow.get_tracking_uri().rstrip("/")
        mlflow_link  = f"{uri}/#/experiments/{exp_id}/runs/{run_id}"

    return {
        "status":      "done",
        "mlflow_run":  run_id,
        "mlflow_url":  mlflow_link
    }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI training server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to bind the server to"
    )
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "main:app",        # if this file is named main.py
        host=args.host,
        port=args.port,
        reload=True
    )