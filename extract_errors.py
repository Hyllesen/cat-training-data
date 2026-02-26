import cv2
import os

# --- SETTINGS ---
video_path = "detections/horny_meow_p92_20260226_185224.mp4"
actual_class = "horny_meow"  # The true class of the cat in the video (e.g., "orange", "horny_meow", "resident")
output_folder = f"training_data/{actual_class}" # Path to actual class shown in video (e.g., "orange", "horny_meow", "resident")
start_sec = 0  # Start just before the error
end_sec = 1    # End just after the error

# --- EXECUTION ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

count = 0
while cap.isOpened():
    frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ret, frame = cap.read()
    
    if not ret or frame_no > end_frame:
        break

    # Save every 5th frame to avoid nearly identical images
    if count % 5 == 0:
        # Use part of the filename and frame number for traceability
        img_name = f"hard_example_{actual_class}_{int(frame_no)}_video{video_path.split('/')[-1].split('.')[0]}.jpg"
        cv2.imwrite(os.path.join(output_folder, img_name), frame)
        print(f"Saved: {img_name}")
    
    count += 1

cap.release()
print(f"✅ Extraction complete. Added to {output_folder}")