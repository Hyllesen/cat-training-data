import os
import time
import subprocess
import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from scipy.io import wavfile
from datetime import datetime

# --- CONFIGURATION ---
RTSP_URL = "rtsp://localhost:8554/garden"
# 81=Cat, 82=Meow, 83=Caterwaul, 20=Crying/sobbing, 21=Baby cry
TARGET_CLASSES = [81, 82, 83, 20, 21] 
CONF_THRESHOLD = 0.25  
GAIN_FACTOR = 3.5      # Slightly increased boost for the 8kHz stream
OUTPUT_DIR = "audio_recordings"
LOG_FILE = "audio_log.txt"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Loading YAMNet... Logging to {LOG_FILE}")
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy().decode('utf-8')

class_names = []
with open(class_map_path, 'r') as f:
    reader = csv.reader(f)
    next(reader) 
    for row in reader:
        if len(row) >= 3:
            class_names.append(row[2])

def get_audio_stream():
    command = [
        'ffmpeg', '-i', RTSP_URL,
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-f', 's16le', '-'
    ]
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def log_message(message):
    """Prints to console and appends to a log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    with open(LOG_FILE, "a") as f:
        f.write(formatted_msg + "\n")

def main():
    log_message(f"System Online. Listening on {RTSP_URL}...")
    stream = get_audio_stream()
    chunk_size = 16000 * 2 
    
    try:
        while True:
            raw_bytes = stream.stdout.read(chunk_size)
            if not raw_bytes:
                break
            
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_data = np.clip(audio_data * GAIN_FACTOR, -1.0, 1.0)
            
            scores, embeddings, spectrogram = model(audio_data)
            prediction = np.mean(scores, axis=0)
            top_class = np.argmax(prediction)
            
            # Identify what it's hearing
            top_3_indices = np.argsort(prediction)[-3:][::-1]
            debug_str = " | ".join([f"{class_names[i]}: {prediction[i]:.2f}" for i in top_3_indices])
            
            # Log the top sound every 2 seconds so the terminal doesn't get flooded,
            # but we can still see the history.
            if int(time.time()) % 2 == 0:
                log_message(f"Hearing: {debug_str}")
            
            if top_class in TARGET_CLASSES and prediction[top_class] > CONF_THRESHOLD:
                label = class_names[top_class]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(OUTPUT_DIR, f"{label}_{timestamp}.wav")
                
                log_message(f"🚨 DETECTED {label.upper()} ({prediction[top_class]:.2f}) -> {filename}")
                wavfile.write(filename, 16000, (audio_data * 32767).astype(np.int16))

    except KeyboardInterrupt:
        log_message("Stopping audio monitor...")
    finally:
        stream.terminate()

if __name__ == "__main__":
    main()