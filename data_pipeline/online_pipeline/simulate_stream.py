import os, time, random
from pydub import AudioSegment

BACKGROUND_DIR = "/mnt/persist/train_soundscapes"
CALLS_DIR = "/mnt/persist/train_audio"
OUTPUT_DIR = "/mnt/persist/online_input"
INTERVAL = 10  # seconds
OVERLAY_PROB = 0.3  # 30% chance to overlay a bird call

def pick_random_audio(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".ogg")]
    return os.path.join(folder, random.choice(files)) if files else None

def overlay_random_call(background):
    call_path = pick_random_audio(CALLS_DIR)
    if call_path is None:
        return background
    call = AudioSegment.from_file(call_path)
    start = random.randint(0, max(0, len(background) - len(call)))
    return background.overlay(call, position=start)

def simulate_stream():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    while True:
        base_audio_path = pick_random_audio(BACKGROUND_DIR)
        if base_audio_path is None:
            print("No background files found.")
            break
        base_audio = AudioSegment.from_file(base_audio_path)
        if random.random() < OVERLAY_PROB:
            base_audio = overlay_random_call(base_audio)

        out_name = f"streamed_{int(time.time())}.ogg"
        base_audio.export(os.path.join(OUTPUT_DIR, out_name), format="ogg")
        print(f"[Streamed] {out_name}")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    simulate_stream()
