To help you maintain the accuracy of your model as your dataset grows, here is a step-by-step guide for Hard Example Mining. This process ensures that you aren't just adding "more" data, but "better" data that directly fixes the mistakes you're seeing in your v3 videos.

🐱 HARD_EXAMPLE_MINING.md
Follow these steps whenever the model makes a confident mistake (e.g., calling the grey stray "Orange" or vice-versa).

Step 1: Identify the Error
Review the videos in your detections/ folder. Note the filename and the specific timestamps where the label is incorrect.

Example: horny*meow_p94*...mp4 shows ORANGE at 00:01.

Example: orange*p99*...mp4 shows HORNY_MEOW at 00:30.

Step 2: Extract the "Lies"
Use your extraction script to grab only the frames where the model was wrong. Do not extract the whole video; 5–10 frames of the mistake is usually enough.

Open extract_errors.py.

Update video_path, start_sec, and end_sec to match the error.

Set the output_folder to the TRUE identity of the cat.

If the grey stray was called Orange, save frames to training_data/horny_meow.

If the resident was called Stray, save frames to training_data/orange.

Step 3: Verify Dataset Balance
Before training, ensure your folders haven't become heavily imbalanced. The model learns best when classes are roughly equal.

Run the count command in your zsh terminal:

Bash
for d in \*/; do echo "$d $(ls -1 "$d" | wc -l)"; done
If one folder is >20% larger than the others, run python balance_dataset.py.

Step 4: Fine-Tune (The v4+ Command)
Instead of starting from scratch, start from your last "Best" model. We will increase the resolution (imgsz) to 320 to help the model see fur textures more clearly.

Run this command on your Mac Mini M4:

Bash
yolo classify train \
 data=training_data \
 model=runs/classify/cat_identity_v3/weights/best.pt \
 epochs=50 \
 imgsz=320 \
 device=mps \
 batch=16 \
 name=cat_identity_v4
Step 5: Validate and Swap
Once training finishes, check the new confusion matrix in runs/classify/cat_identity_v4/confusion_matrix.png.

Update your cat_monitor.py to point to the new weights:

Python

# Inside cat_monitor.py

CLASSIFIER_MODEL = "runs/classify/cat_identity_v4/weights/best.pt"
Run the monitor and observe if the specific error at that garden spot is resolved.

💡 Pro-Tips for Bohol Sunlight
Lighting Variations: If a mistake happens at noon, extract those frames. If it happens at dusk, extract those too. The model needs to see the "Hard Example" in different lighting to truly understand it.

Background Check: Ensure your background/ folder stays at ~1,000 images so the model doesn't start "hallucinating" cats into empty shadows.
