import os
from glob import glob

import cv2

YOLO_ROOT = "data/yolo_dataset"
IMAGES_TRAIN = os.path.join(YOLO_ROOT, "images/train")
LABELS_TRAIN = os.path.join(YOLO_ROOT, "labels/train")
DEBUG_TRAIN = os.path.join(YOLO_ROOT, "debug_train")

IMAGES_VAL = os.path.join(YOLO_ROOT, "images/val")
LABELS_VAL = os.path.join(YOLO_ROOT, "labels/val")
DEBUG_VAL = os.path.join(YOLO_ROOT, "debug_val")

# In your car dataset, you have only class 0 = car
CLASS_NAMES = {0: "car"}


def draw_yolo_boxes_on_image(img_path, label_path, out_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"No label file for image: {img_path}")
        cv2.imwrite(out_path, img)  # save original anyway
        return

    with open(label_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            print(f"Bad label line in {label_path}: {line}")
            continue

        class_id = int(parts[0])
        x_center, y_center, bw, bh = map(float, parts[1:])

        # Convert from normalized YOLO format to pixel coords
        x_center_px = x_center * w
        y_center_px = y_center * h
        bw_px = bw * w
        bh_px = bh * h

        x_min = int(x_center_px - bw_px / 2)
        y_min = int(y_center_px - bh_px / 2)
        x_max = int(x_center_px + bw_px / 2)
        y_max = int(y_center_px + bh_px / 2)

        # Draw rectangle
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw label text
        class_name = CLASS_NAMES.get(class_id, str(class_id))
        cv2.putText(
            img,
            class_name,
            (x_min, max(y_min - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    cv2.imwrite(out_path, img)


def visualize_split(images_dir, labels_dir, debug_dir, max_images=20):
    os.makedirs(debug_dir, exist_ok=True)

    image_paths = sorted(glob(os.path.join(images_dir, "*.jpg")))
    print(f"Found {len(image_paths)} images in {images_dir}")

    # Only visualize a subset to keep it quick
    image_paths = image_paths[:max_images]

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        out_path = os.path.join(debug_dir, fname)

        draw_yolo_boxes_on_image(img_path, label_path, out_path)
        print(f"Saved debug image: {out_path}")


def main():
    print("Visualizing TRAIN split...")
    visualize_split(IMAGES_TRAIN, LABELS_TRAIN, DEBUG_TRAIN, max_images=20)

    print("\nVisualizing VAL split...")
    visualize_split(IMAGES_VAL, LABELS_VAL, DEBUG_VAL, max_images=10)

    print("\n✅ Done. Open these folders to inspect:")
    print("  - data/yolo_dataset/debug_train")
    print("  - data/yolo_dataset/debug_val")


if __name__ == "__main__":
    main()
