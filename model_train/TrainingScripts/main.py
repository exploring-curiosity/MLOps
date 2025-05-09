import argparse
import asyncio
import mlflow
import torch
from fastapi import FastAPI, Query

app = FastAPI()

MODEL_PATH = "./best_rawcnn.pt"
MODEL_NAME = "RawCNN"

# Global lock to prevent concurrent training
training_lock = asyncio.Lock()


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
            model = torch.load(MODEL_PATH, weights_only=False, map_location=torch.device('cpu'))
            mlflow.pytorch.log_model(model, artifact_path="model")

        run_id        = run.info.run_id
        exp_id        = run.info.experiment_id
        tracking_uri  = mlflow.get_tracking_uri()  # e.g. "http://localhost:5000"
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