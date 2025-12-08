import os
import sqlite3
import pandas as pd

DB_PATH = "data/db/inference_logs.db"

def main():
    if not os.path.exists(DB_PATH):
        print(f"DB file not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)

    # Load entire table into a DataFrame
    df = pd.read_sql_query("SELECT * FROM inference_logs", conn)
    conn.close()

    if df.empty:
        print("No rows in inference_logs table yet.")
        return

    print("\n=== Basic stats ===")
    print("Total images logged:", len(df))

    zero_det = df[df["num_detections"] == 0]
    print("Images with ZERO detections:", len(zero_det))

    print("\nSample of images with zero detections:")
    print(zero_det["image_path"].head(10).to_string(index=False))

    # Some rows may have min_conf / avg_conf as NULL (None) when there were 0 detections,
    # so we drop them for confidence-based analysis.
    conf_df = df.dropna(subset=["min_conf", "avg_conf", "max_conf"])

    if not conf_df.empty:
        print("\n=== Lowest min_conf (hardest images) ===")
        print(
            conf_df.nsmallest(10, "min_conf")[["image_path", "min_conf"]]
            .to_string(index=False)
        )

        print("\n=== Lowest avg_conf images ===")
        print(
            conf_df.nsmallest(10, "avg_conf")[["image_path", "avg_conf"]]
            .to_string(index=False)
        )

        print("\n=== Highest detection counts ===")
        print(
            df.nlargest(10, "num_detections")[["image_path", "num_detections"]]
            .to_string(index=False)
        )
    else:
        print("\nNo rows with confidence values (all had zero detections).")

    # Save a CSV report for manual inspection
    out_csv = "data/db/inference_report.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nFull report saved to: {out_csv}")

if __name__ == "__main__":
    main()
