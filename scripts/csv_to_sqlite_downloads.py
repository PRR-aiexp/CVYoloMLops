import os
import csv
import sqlite3

CSV_PATH =  "data/api_raw/download_log.csv"
DB_PATH = "data/db/api_downloads.db"

def ensure_table(conn):
	cur = conn.cursor()
	cur.execute( """ 
		CREATE TABLE IF NOT EXISTS downloads (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			ts TEXT,
			url TEXT,
			out_path TEXT,
			status TEXT
		);
		"""
	)
	conn.commit()

def load_csv_to_db(csv_path, db_path):
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV file not found {csv-path}")
	os.makedirs(os.path.dirname(db_path),exist_ok=True)
	
	conn = sqlite3.connect(db_path)
	ensure_table(conn)

	cur = conn.cursor()
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		rows = list(reader)
	print(f"Found {len(rows)} rows in {csv_path}")
		
	for row in rows:
		cur.execute(
			""" 
			INSERT INTO downloads (ts, url, out_path, status)
			VALUES (?,?,?,?)
			""",
			(row.get("timestamp"),row.get("url"),row.get("out_path"),row.get("status"))
		)

	conn.commit()
	conn.close()
	print(f"loaded csv to sqlite db: {db_path}")

def main():
	load_csv_to_db(CSV_PATH, DB_PATH)

if __name__ == "__main__":
	main()
