import mlflow
import mlflow.pyfunc
import os
from ultralytics import YOLO

def main():
	
	DATASET_YAML = "data/yolo_dataset/car_detection.yaml"
	MODEL_NAME = "yolov8n.pt" 

	mlflow.set_tracking_uri("file:/home/pritir/CVYoloMlops/mlruns") #use local folder as tracking uri for mlflow
	mlflow.set_experiment("car_detection_yolo")

	with mlflow.start_run(run_name = "yolov8n_run_1"):
		mlflow.log_param("model", MODEL_NAME)
		mlflow.log_param("img_size",640)
		mlflow.log_param("epochs",30)
		mlflow.log_param("batch",8)

		model = YOLO(MODEL_NAME) #load model
		#train model
		results = model.train( data = DATASET_YAML, imgsz = 640, epochs = 30, batch = 8, project = "runs/train", name = "yolo_car_experiment", exist_ok=True)
		
		#get yolo metrics 
		metrics = results.results_dict

		#log metrics
		for k, v in metrics.items():
			try:
				mlflow.log_metric(k, float(v))
			except:
				pass
		#log model weights
		best_model_path = results.best
		mlflow.log_artifact(best_model_path)

		print("training complete")
		

if __name__ == "__main__":
	main()
