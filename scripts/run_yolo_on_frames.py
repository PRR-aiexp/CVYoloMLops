from ultralytics import YOLO
import os
from glob import glob 

INPUT_DIR = "data/raw_frames"
OUTPUT_DIR = "data/frames_yolo"

def main():
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	model = YOLO("yolov8n.pt")
	frame_paths = sorted(glob(os.path.join(INPUT_DIR,"*.jpg")))
	print(f"found {len(frame_paths)} frames")
	
	for cnt, img_path in enumerate(frame_paths, start=1):
		filename = os.path.basename(img_path)
		out_path = os.path.join(OUTPUT_DIR,filename)

		results = model(img_path)
		results[0].save(out_path)

		if cnt%10 == 0 or cnt==len(frame_paths):
			print(f"processed {cnt}/{len(frame_paths)} frames")
	print(f"\n Annotated frames saved in {OUTPUT_DIR}")
	
if __name__ == "__main__":
	main()
