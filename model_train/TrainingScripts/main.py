import argparse
import asyncio
import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Query

app = FastAPI()

MODEL_PATH = "./best_rawcnn.pt"
MODEL_NAME = "RawCNN"

# Global lock to prevent concurrent training
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
    model_name: str = Query(..., description="The name of the model to train"),
    data_source: str = Query(..., description="Where to load training data from")
):
    """
    Simulate a training job by sleeping for a bit.
    Prevents concurrent runs via an asyncio.Lock.
    """
    if training_lock.locked():
        # Another training is already in progress
        return {
            "status": "in_progress",
            "model_name": model_name,
            "data_source": data_source
        }

    async with training_lock:
        # simulate “work”
        print("hello world")
        with mlflow.start_run() as run:
            await asyncio.sleep(60)
            model = RawAudioCNN(num_classes=206)
            state = torch.load(MODEL_PATH, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            model = torch.load(MODEL_PATH, weights_only=False, map_location=torch.device('cpu'))
            mlflow.pytorch.log_model(model, artifact_path="model")

        run_id        = run.info.run_id
        exp_id        = run.info.experiment_id
        tracking_uri  = mlflow.get_tracking_uri()
        ui_url = f"{tracking_uri.rstrip('/')}/#/experiments/{exp_id}/runs/{run_id}"
        
        return {
            "status":       "done",
            "model_name":   model_name,
            "data_source":  data_source,
            "mlflow_run_id": run_id,
            "mlflow_url":   ui_url    
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