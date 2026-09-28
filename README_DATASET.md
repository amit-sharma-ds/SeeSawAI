# Dataset Setup for Training

This folder structure is ready for custom CCTV training.

## Folder layout

- `datasets/cctv_train/images/train/` -> training images
- `datasets/cctv_train/images/val/` -> validation images
- `datasets/cctv_train/labels/train/` -> YOLO labels for training
- `datasets/cctv_train/labels/val/` -> YOLO labels for validation
- `datasets/cctv_train/data.yaml` -> dataset config

## Label format
Each image should have a matching `.txt` annotation file with YOLO format:

`class_id center_x center_y width height`

Example:

```text
0 0.512 0.476 0.214 0.341
```

## How to use
1. Put your training images into `datasets/cctv_train/images/train/`
2. Put validation images into `datasets/cctv_train/images/val/`
3. Put matching labels in the corresponding `labels` folders
4. Run:

```bash
python train_yolo.py
```

## Notes
- This is the correct structure for YOLO training.
- Replace the single class with your real classes if you later add more categories.
- The actual accuracy and performance must be checked with validation metrics after training.
