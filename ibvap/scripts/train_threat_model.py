#!/usr/bin/env python3
"""
scripts/train_threat_model.py
=============================
Fine-tunes the Threat / Weapon Detection YOLOv8 model on user-provided
armed weapons (guns, blasters) and threat images from 'test img/'.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def build_dataset(images_dir: Path, output_dir: Path):
    """Generate YOLO format dataset from test images."""
    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"
    val_img_dir = output_dir / "images" / "val"
    val_lbl_dir = output_dir / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(images_dir.glob("*.*")))
    print(f"[*] Found {len(img_files)} source images in {images_dir}")

    # Annotated bounding boxes for the 7 user images
    # Classes: 0: Gun, 1: explosion, 2: grenade, 3: knife
    # Format: (class_id, x_center, y_center, width, height) in normalized [0, 1]
    annotations = {
        "IMG_20260930_135716134.jpg.jpeg": [
            (0, 0.558, 0.546, 0.754, 0.593)  # Gun profile side
        ],
        "IMG_20260930_135722498.jpg.jpeg": [
            (0, 0.480, 0.488, 0.602, 0.700)  # Gun profile other side
        ],
        "IMG_20260930_135746325.jpg.jpeg": [
            (0, 0.410, 0.487, 0.434, 0.154)  # Gun top down view
        ],
        "IMG_20260930_135752262.jpg.jpeg": [
            (0, 0.421, 0.500, 0.304, 0.954)  # Gun vertical rear view
        ],
        "IMG_20260930_135757138.jpg.jpeg": [
            (0, 0.346, 0.600, 0.348, 0.737)  # Gun front perspective
        ],
        "IMG_20260930_135808464.jpg.jpeg": [
            (0, 0.522, 0.606, 0.487, 0.788)  # Gun diagonal perspective
        ],
        "IMG_20260930_135831293.jpg.jpeg": [
            (0, 0.567, 0.593, 0.469, 0.650)  # Gun front 3/4 perspective with person
        ],
    }

    count = 0
    for img_p in img_files:
        name = img_p.name
        boxes = annotations.get(name, [])
        if not boxes:
            # Fallback to color segmentation for any new images
            img = cv2.imread(str(img_p))
            if img is not None:
                h, w = img.shape[:2]
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                m1 = cv2.inRange(hsv, (5, 120, 100), (25, 255, 255))
                m2 = cv2.inRange(hsv, (95, 100, 50), (130, 255, 255))
                mask = cv2.morphologyEx(m1 | m2, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = [c for c in cnts if cv2.contourArea(c) > 50000]
                for c in cnts:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    boxes.append((0, (bx + bw/2)/w, (by + bh/2)/h, bw/w, bh/h))

        # Copy image to dataset
        dst_img = train_img_dir / img_p.name
        shutil.copy2(img_p, dst_img)

        # Write labels
        lbl_file = train_lbl_dir / f"{img_p.stem}.txt"
        with open(lbl_file, "w") as f:
            for cls_id, cx, cy, nw, nh in boxes:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        # Also populate val set for validation metric calculation
        shutil.copy2(img_p, val_img_dir / img_p.name)
        val_lbl = val_lbl_dir / f"{img_p.stem}.txt"
        shutil.copy2(lbl_file, val_lbl)

        count += 1
        print(f" [+] Processed {img_p.name} with {len(boxes)} weapon annotations")

    # Generate data.yaml
    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        f.write(f"""path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val
names:
  0: Gun
  1: explosion
  2: grenade
  3: knife
""")
    print(f"[*] Dataset generated at {output_dir}, YAML: {data_yaml_path}")
    return data_yaml_path


def train_model(data_yaml: Path, base_model: str = "models/threat_yolov8n.pt", epochs: int = 12):
    """Train YOLOv8 model on custom dataset."""
    print(f"[*] Loading base model from {base_model}...")
    model = YOLO(base_model)

    print(f"[*] Starting fine-tuning for {epochs} epochs...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=4,
        workers=2,
        project="runs/threat_train",
        name="custom_threat",
        exist_ok=True,
        verbose=True,
    )

    best_weights = Path("runs/threat_train/custom_threat/weights/best.pt")
    if not best_weights.exists():
        best_weights = Path("runs/threat_train/custom_threat/weights/last.pt")

    if best_weights.exists():
        target = Path("models/threat_yolov8n.pt")
        backup = Path("models/threat_yolov8n_backup.pt")
        if not backup.exists() and target.exists():
            shutil.copy2(target, backup)
        shutil.copy2(best_weights, target)
        print(f"[SUCCESS] Fine-tuned model saved to {target} (backup at {backup})")
    else:
        print("[WARNING] Could not find best.pt weights file.")

    return results


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    test_imgs_path = repo_root.parent / "test img"
    dataset_path = repo_root / "data" / "threat_dataset"

    yaml_file = build_dataset(test_imgs_path, dataset_path)
    train_model(yaml_file, base_model="models/threat_yolov8n.pt", epochs=12)
