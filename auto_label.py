import cv2
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
INPUT_DIR = "recordings"      # Folder containing your saved cat clips
IMAGE_OUT_DIR = "images"      # Where the extracted frames will go
LABEL_OUT_DIR = "labels"      # Where the YOLO .txt annotations will go
MODEL_NAME = "yolo26n.pt"     # Using the fast NMS-free model
CONFIDENCE_THRESHOLD = 0.60   # Only auto-label if the AI is 60%+ sure it's a cat
FRAMES_TO_SKIP = 15           # For a 15fps video, this extracts 1 frame per second

# Create output directories if they don't exist
os.makedirs(IMAGE_OUT_DIR, exist_ok=True)
os.makedirs(LABEL_OUT_DIR, exist_ok=True)

# Load the model and explicitly use Apple Silicon hardware acceleration
print(f"Loading {MODEL_NAME} on MPS (Apple Silicon)...")
model = YOLO(MODEL_NAME)

# Process every video in the input directory
for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(('.mp4', '.avi', '.mov')):
        continue
        
    video_path = os.path.join(INPUT_DIR, filename)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved_count = 0
    base_name = os.path.splitext(filename)[0]
    
    print(f"Processing: {filename}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video
            
        # Only process 1 frame per second to ensure dataset diversity
        if frame_count % FRAMES_TO_SKIP == 0:
            # Run inference targeting only class 16 (cat)
            results = model.predict(
                source=frame, 
                device='mps', 
                classes=[16], 
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )
            
            for r in results:
                # If a cat was found in this frame
                if len(r.boxes) > 0:
                    img_name = f"{base_name}_frame_{frame_count}.jpg"
                    txt_name = f"{base_name}_frame_{frame_count}.txt"
                    
                    img_path = os.path.join(IMAGE_OUT_DIR, img_name)
                    txt_path = os.path.join(LABEL_OUT_DIR, txt_name)
                    
                    # Save the image
                    cv2.imwrite(img_path, frame)
                    
                    # Save the YOLO format text file
                    with open(txt_path, 'w') as f:
                        for box in r.boxes:
                            # Extract normalized coordinates: x_center, y_center, width, height
                            # These are the exact metrics YOLO needs for training
                            x_c, y_c, w, h = box.xywhn[0].tolist()
                            
                            # We write '0' as the temporary class ID for all cats.
                            # You will manually change the stray to '1' in your labeling tool.
                            f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    
                    saved_count += 1
                    
        frame_count += 1
        
    cap.release()
    print(f"  -> Extracted and auto-labeled {saved_count} frames from {filename}.")

print("\nAuto-labeling complete! Check your 'images' and 'labels' folders.")