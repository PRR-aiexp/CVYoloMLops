import sqlite3
import os

DB_PATH = "data/db/inference_logs.db"

def main():
	os.makedirs("data/db",exist_ok=True)

	conn = sqlite3.connect(DB_PATH)
	cur = conn.cursor()

	cur.execute("""
		CREATE TABLE IF NOT EXISTS infernece_logs(
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp TEXT,
			image_path TEXT,
			num_detections INTEGER,
			avg_conf REAL,
			max_conf REAL,
			min_conf REAL
		);
	""")

	conn.commit()
	conn.close()
	print("inference_logs table created at ", DB_PATH)
	
if __name__ == "__main__":
	main()

