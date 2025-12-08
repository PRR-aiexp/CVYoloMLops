import os
import sqlite3
from glob import glob
from datetime import datetime

from ultralytics import YOLO

DB_PATH = "data/db/inference_logs.db"
IMAGE_DIR = "data/yolo_dataset/images/val"  # run on validation set first
MODEL_PATH = "yolov8n.pt"  # or your trained model later


def log_to_db(image_path, num_det, avg_conf, min_conf, max_conf):
    """Insert one inference record into the SQLite DB safely."""
    # Make sure the directory for DB exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Ensure table exists (defensive)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            image_path TEXT,
            num_detections INTEGER,
            avg_conf REAL,
            max_conf REAL,
            min_conf REAL
        );
        """
    )

    # Use parameter placeholders (?), NEVER f-strings
    cur.execute(
        """
        INSERT INTO inference_logs
            (timestamp, image_path, num_detections, avg_conf, max_conf, min_conf)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (
            datetime.now().isoformat(),
            image_path,
            int(num_det),
            float(avg_conf) if avg_conf is not None else None,
            float(max_conf) if max_conf is not None else None,
            float(min_conf) if min_conf is not None else None,
        ),
    )

    conn.commit()
    conn.close()


def main():
    # Load model once
    print(f"Loading model from {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    # Collect images
    image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")))
    print(f"Found {len(image_paths)} images in {IMAGE_DIR}")

    if not image_paths:
        print("No images found. Check IMAGE_DIR path.")
        return

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"\n[{idx}/{len(image_paths)}] Running inference on: {img_path}")

        # Run YOLO inference
        results = model(img_path)[0]

        confs = []
        if results.boxes is not None:
            for box in results.boxes:
                # box.conf is a tensor of shape (1,)
                confs.append(float(box.conf[0]))

        if confs:
            avg_conf = sum(confs) / len(confs)
            min_conf = min(confs)
            max_conf = max(confs)
        else:
            avg_conf = None
            min_conf = None
            max_conf = None

        # Log into DB
        log_to_db(
            img_path,
            len(confs),
            avg_conf,
            min_conf,
            max_conf,
        )

        print(
            f"Logged: {len(confs)} detections | "
            f"avg={avg_conf} min={min_conf} max={max_conf}"
        )

    print("\n✅ Logging complete.")


if __name__ == "__main__":
    main()
