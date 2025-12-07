import sqlite3

DB_PATH = "data/db/api_downloads.db"
def main():
	conn = sqlite3.connect(DB_PATH)
	cur = conn.cursor()

	cur.execute("SELECT COUNT(*) FROM downloads;")
	(count,) = cur.fetchone()
	print(f"total rows in downloads {count}")

	cur.execute("SELECT ts, url, status FROM downloads LIMIT 5;")
	rows = cur.fetchall()

	print("\n first 5 rows")
	for r in rows:
		print(r)
	conn.close()


if __name__ == "__main__":
	main()

