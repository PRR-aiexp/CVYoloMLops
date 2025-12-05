import os
import requests
from datetime import datetime

URL_LIST = "data/url_lists/api_lists.txt"
OUTPUT_DIR = "data/api_raw"
LOG_CSV = "data/api_raw/download_log.csv"

def download_image(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    url = url.strip()
    if not url:
        return None, "empty_url"

    # Try to get a filename from the URL
    name = url.split("/")[-1]
    if not name:
        name = "image.jpg"

    # If no extension, add .jpg
    if "." not in name:
        name += ".jpg"

    # Avoid overwriting: add timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{name}"
    out_path = os.path.join(out_dir, filename)

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None, f"http_{resp.status_code}"

        with open(out_path, "wb") as f:
            f.write(resp.content)

        return out_path, "ok"
    except Exception as e:
        return None, f"error: {e}"

def main():
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)

    if not os.path.exists(URL_LIST):
        raise FileNotFoundError(f"URL list file not found: {URL_LIST}")

    with open(URL_LIST, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URLs in {URL_LIST}")

    # Open CSV log file (append if exists)
    header_needed = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", encoding="utf-8") as log_f:
        if header_needed:
            log_f.write("timestamp,url,out_path,status\n")

        for url in urls:
            ts = datetime.utcnow().isoformat()
            out_path, status = download_image(url, OUTPUT_DIR)
            log_f.write(f"{ts},{url},{out_path or ''},{status}\n")
            print(f"{status:20} | {url}")

    print("\nDownload complete.")
    print(f"Images (if ok) are in: {OUTPUT_DIR}")
    print(f"Log CSV: {LOG_CSV}")

if __name__ == "__main__":
    main()
