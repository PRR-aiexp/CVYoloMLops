from ultralytics import YOLO
import os
from glob import glob

INPUT_DIR = "data/api_raw"
OUTPUT_DIR = "data/api_yolo"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO("yolov8n.pt")

    image_paths = sorted(glob(os.path.join(INPUT_DIR, "*.*")))
    print(f"Found {len(image_paths)} images in {INPUT_DIR}")

    for i, img_path in enumerate(image_paths, start=1):
        filename = os.path.basename(img_path)
        out_path = os.path.join(OUTPUT_DIR, filename)

        results = model(img_path)
        results[0].save(out_path)

        if i % 5 == 0 or i == len(image_paths):
            print(f"Processed {i}/{len(image_paths)} images")

    print("\nYOLO complete.")
    print(f"Annotated images saved in: {OUTPUT_DIR}")
    print(" Open via Files app: Linux files → cv_pipeline → data → api_yolo")

if __name__ == "__main__":
    main()
