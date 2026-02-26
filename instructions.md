# Cat Detector Project

## Project Overview
This project monitors the back garden via a Tapo C200C camera to manage cat visits. The primary objective is to distinguish between resident cats and a specific stray cat to automate deterrents.

## The Goal
- **Identify Residents:** 4 Orange cats and 1 Black/White cat (named **Squaky**).
- **Identify Stray:** 1 Grey cat (named **Horny Meow**).
- **Action:** Detect the stray cat ("Horny Meow") and eventually trigger a deterrent (horn, water spray, or noise) while ignoring the resident cats.

## Technical Architecture
The system uses a two-stage computer vision pipeline:

1.  **Detection:** A YOLOv8 object detection model (`models/detector/best.pt`) identifies "cats" in the RTSP stream.
2.  **Classification:** Once a cat is detected, the bounding box is cropped and passed to a YOLOv8-cls classification model (`runs/classify/.../best.pt`) to identify the specific cat.

### Stack
- **Hardware:** Tapo C200C (RTSP Stream).
- **Core:** Python, OpenCV, Ultralytics YOLOv8.
- **Acceleration:** Apple Silicon MPS (Metal Performance Shaders) for GPU acceleration.

## Project Structure
- `cat_monitor.py`: The main real-time monitoring script. It draws green boxes for residents and red boxes for the stray.
- `cat_recorder.py`: Tool for capturing raw footage to build the dataset.
- `train_classifier.py`: Script to train the identity classification model.
- `identity_sorter.py`: Utility to help organize detected crops into folders for training.
- `balance_dataset.py`: Script to ensure even distribution of images across cat identities for training.
- `extract_bg.py`: Extracts background/negative samples from video clips.
- `detections/`: Stores video clips of identified cats for review.
- `training_data/`: Dataset organized by identity:
    - `orange/` (The 4 resident orange cats)
    - `squaky/` (The resident black/white cat)
    - `horny_meow/` (The stray grey cat)
    - `background/` (False positives/empty frames)

## Current Status & Next Steps
- [x] RTSP connection and frame processing.
- [x] YOLO Detection (Cat vs. No Cat).
- [x] Identity Classification (Orange vs. Squaky vs. Horny Meow).
- [x] Visual feedback and video recording of events.
- [ ] **Implementation of Deterrent:** Integrate with hardware (e.g., smart plug, local speaker, or GPIO) to sound a horn or spray water when `horny_meow` is detected with high confidence.

## Guidance for AI Agents
- **Performance:** Always use `device='mps'` when running models to utilize the Mac GPU.
- **Dataset:** New detection clips in `detections/` should be periodically reviewed and added to `training_data/` to improve classifier accuracy.
- **Thresholds:** The `CONF_THRESHOLD` in `cat_monitor.py` is currently set to 0.7 to minimize false alarms.
