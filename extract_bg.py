import cv2
import os
import time 

# --- SETTINGS ---
video_path = "empty_garden_background.mp4" # Put your video filename here
output_folder = "training_data/background"
save_every_n_frames = 2 

# --- EXECUTION ---
os.makedirs(output_folder, exist_ok=True)
cap = cv2.VideoCapture(video_path)
count = 0
saved_count = 0

print(f"🎬 Processing {video_path}...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % save_every_n_frames == 0:
        # Use a timestamp or unique ID to prevent overwriting existing 83 images
        timestamp = int(time.time() * 1000) # Milliseconds for uniqueness
        img_name = f"bg_frame_{count}_{saved_count}_{timestamp}.jpg"
        cv2.imwrite(os.path.join(output_folder, img_name), frame)
        saved_count += 1

    count += 1

cap.release()
print(f"✅ Done! Saved {saved_count} new background images to {output_folder}")