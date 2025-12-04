from ultralytics import YOLO
import matplotlib.pyplot as plt 
from PIL import Image 
import os

def main():
	img_path = "data/samples/street.jpg"
	print("Image exists", os.path.exists(img_path), img_path)
	if not os.path.exists(img_path):
		raise FileNotFoundError(f"image not found: {img_path}")
	model = YOLO("yolov8n.pt")
	results = model(img_path)
	out_path = "data/samples/output_street.jpg"
	results[0].save(out_path)
	print("Saved output to:", out_path)
	#img = Image.open(out_path)
	#plt.imshow(img)
	#plt.axis("off")
	#plt.show()
if __name__ == "__main__":
	main()
