import cv2
import os
import shutil
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "models/chicken-proof/best.pt"
POS_VIDEOS = "dataset/positives"   # Videos with cats
NEG_VIDEOS = "dataset/negatives"   # Videos with chickens/nothing
OUTPUT_DIR = "training_data"
CONF_THRESHOLD = 0.5

# Initialize
model = YOLO(MODEL_PATH)
classes = ["stray", "resident_1", "resident_2", "background"]

for cls in classes:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

def process_videos(video_dir, is_negative=False):
    files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mov'))]
    
    for v_name in files:
        cap = cv2.VideoCapture(os.path.join(video_dir, v_name))
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Process every 10th frame to save time/memory
            if frame_idx % 10 == 0:
                results = model(frame, verbose=False, device='mps')
                
                if is_negative:
                    # If it's a negative video and the model "lies" to us, save it as background
                    if len(results[0].boxes) > 0:
                        save_path = f"{OUTPUT_DIR}/background/bg_{v_name}_{frame_idx}.jpg"
                        cv2.imwrite(save_path, frame)
                else:
                    # If it's a positive video, crop the cat for identification
                    for i, box in enumerate(results[0].boxes):
                        if box.conf > CONF_THRESHOLD:
                            b = box.xyxy[0].cpu().numpy().astype(int)
                            crop = frame[b[1]:b[3], b[0]:b[2]]
                            # Save to stray initially, you'll sort them manually later
                            save_path = f"{OUTPUT_DIR}/stray/crop_{v_name}_{frame_idx}_{i}.jpg"
                            cv2.imwrite(save_path, crop)
            frame_idx += 1
        cap.release()
        print(f"Done processing {v_name}")

if __name__ == "__main__":
    print("💎 Processing Positives (Crops)...")
    process_videos(POS_VIDEOS, is_negative=False)
    
    print("🐔 Processing Negatives (Backgrounds)...")
    process_videos(NEG_VIDEOS, is_negative=True)