import os
import time
import requests

INPUT_DIR = "/mnt/persist/online_input"
POLL_INTERVAL = 5  # seconds
SEEN = set()

def poll_and_trigger():
    print("Polling for new .ogg files...")
    while True:
        for fname in os.listdir(INPUT_DIR):
            if not fname.endswith(".ogg") or fname in SEEN:
                continue
            SEEN.add(fname)
            print(f"[Trigger] Sending {fname} to inference service...")
            path = os.path.join(INPUT_DIR, fname)
            with open(path, "rb") as f:
                files = {"file": (fname, f, "audio/ogg")}
                res = requests.post("http://localhost:8000/infer-audio", files=files)
                print(f"[Response] {res.status_code}: {res.json()}")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    poll_and_trigger()
