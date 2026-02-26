🐾 Cat Identification: Process Guide
Phase 1: Bulk Sorting
Before training, you must separate valid cat footage from "false triggers" (like chickens). Use dataset_prep.py for this stage.

Run the script: python dataset_prep.py.

Review Clips: A window will show a frame from each video in your recordings/ folder.

Sort with Keys:

'C': Moves video to dataset/positives (contains a cat).

'N': Moves video to dataset/negatives (chickens only).

'Q': Quits the sorter.

Phase 2: Automatic Dataset Generation
Instead of manual drawing in CVAT, we use your existing YOLO model to "crop" cats out of the sorted videos.

Configure Paths: Ensure dataset_preparation.py points to your sorted positives:
VIDEOS_DIR = "dataset/positives".

Execute Extraction: python dataset_preparation.py.

How it works:

The script scans every 5th frame to prevent duplicate training data.

It uses MPS (Metal Performance Shaders) to run inference on your M4 GPU.

Any detection above 0.6 confidence is automatically cropped and saved to dataset_raw/.

Phase 3: Manual Classification (Sub-folder Sorting)
Now that you have a folder of cropped images, you must tell the computer who is who.

Create Identity Folders: Inside your training directory, create sub-folders for each target:

/stray

/resident_orange

/resident_tabby

Move Images: Drag and drop the JPEGs from dataset_raw/ into the corresponding folder based on the cat's visual identity.

Phase 4: Training the Custom Model
With your images sorted by identity, we will perform "Transfer Learning" to create the final classifier.

Initialize Training: We will use the ultralytics library to fine-tune a classification model.

M4 Optimization: Always ensure the training command specifies device='mps' to utilize the M4's unified memory and GPU cores.
