import os
import shutil

import cv2
import numpy as np
import pandas as pd

# Paths - adjust if your folder names are different
DATA_ROOT = "data/kaggle_obj/data"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "training_images")
CSV_PATH = os.path.join(DATA_ROOT, "train_solution_bounding_boxes.csv")

YOLO_ROOT = "data/yolo_dataset"
IMAGES_TRAIN = os.path.join(YOLO_ROOT, "images/train")
IMAGES_VAL = os.path.join(YOLO_ROOT, "images/val")
LABELS_TRAIN = os.path.join(YOLO_ROOT, "labels/train")
LABELS_VAL = os.path.join(YOLO_ROOT, "labels/val")

TRAIN_SPLIT = 0.8  # 80% train, 20% val
CLASS_ID = 0       # single class: 'car'


def ensure_dirs():
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        os.makedirs(d, exist_ok=True)


def load_annotations():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # We expect columns: image, xmin, ymin, xmax, ymax
    expected_cols = {"image", "xmin", "ymin", "xmax", "ymax"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    return df


def yolo_box(row, img_w, img_h):
    """
    Convert CSV bbox to YOLO format:
    class x_center y_center width height  (all normalized 0-1)
    """
    xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    return f"{CLASS_ID} {x_center} {y_center} {width} {height}"


def main():
    ensure_dirs()
    print(f"pwd = {os.getcwd()}")
    df = load_annotations()

    # Get the unique image names
    image_names = df["image"].unique()
    print(f"Found {len(image_names)} annotated images")

    # Shuffle and split
    rng = np.random.default_rng(seed=42)
    rng.shuffle(image_names)

    split_idx = int(len(image_names) * TRAIN_SPLIT)
    train_imgs = set(image_names[:split_idx])
    val_imgs = set(image_names[split_idx:])

    print(f"Train images: {len(train_imgs)}, Val images: {len(val_imgs)}")

    for img_name in image_names:
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: image not found on disk: {img_path}")
            continue

        # Read image to get dimensions
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read image: {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        # Decide subset
        if img_name in train_imgs:
            img_out_dir = IMAGES_TRAIN
            lbl_out_dir = LABELS_TRAIN
        else:
            img_out_dir = IMAGES_VAL
            lbl_out_dir = LABELS_VAL

        # Copy image
        dst_img_path = os.path.join(img_out_dir, img_name)
        shutil.copy2(img_path, dst_img_path)

        # Create label file
        stem = os.path.splitext(img_name)[0]
        label_path = os.path.join(lbl_out_dir, f"{stem}.txt")

        img_rows = df[df["image"] == img_name]

        with open(label_path, "w") as f:
            for _, row in img_rows.iterrows():
                line = yolo_box(row, img_w, img_h)
                f.write(line + "\n")

    print("\n Conversion complete.")
    print(f"Images + labels ready under: {YOLO_ROOT}")


if __name__ == "__main__":
    main()
