import cv2
import os

VIDEO_PATH = "data/videos/sample.mp4"
OUTPUT_DIR = "data/raw_frames"
FRAME_EVERY_N = 10 #save every 10th frame to reduce number of frames

def main():
	os.makedirs(OUTPUT_DIR,exist_ok=True)
	cap = cv2.VideoCapture(VIDEO_PATH)
	if not cap.isOpened():
		raise RuntimeError(f"could not open video:{VIDEO_PATH} ")
	frame_idx = 0 
	saved = 0

	while True:
		ret, frame = cap.read()
		if not ret:
			break #end of video
		if frame_idx % FRAME_EVERY_N == 0:
			out_name = f"frame_{frame_idx:05d}.jpg"
			out_path = os.path.join(OUTPUT_DIR,out_name)
			cv2.imwrite(out_path,frame)
			saved +=1

		frame_idx +=1
	cap.release()
	print(f"Done saved {saved} frames to {OUTPUT_DIR}")

if __name__ == "__main__":
	main()
