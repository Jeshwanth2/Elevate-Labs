"""
download_data.py
────────────────
Utility to download the Kaggle Fake-and-Real News dataset
programmatically (requires kaggle API token).

Manual alternative:
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
  Download and place Fake.csv and True.csv inside the data/ folder.
"""
import os
import sys
import zipfile
import subprocess

DATA_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_SLUG = "clmentbisaillon/fake-and-real-news-dataset"
REQUIRED     = ["Fake.csv", "True.csv"]


def check_files_exist() -> bool:
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED)


def download_via_kaggle_api():
    """Download using the Kaggle CLI (needs ~/.kaggle/kaggle.json)."""
    print(f"[*] Downloading dataset '{DATASET_SLUG}' via Kaggle API …")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET_SLUG,
             "-p", DATA_DIR, "--unzip"],
            check=True
        )
        print("[✓] Download complete.")
    except FileNotFoundError:
        print("[✗] 'kaggle' command not found. Install it with: pip install kaggle")
        print("    Then set up ~/.kaggle/kaggle.json from https://www.kaggle.com/settings")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[✗] Kaggle CLI error: {e}")
        sys.exit(1)


def main():
    if check_files_exist():
        print("[✓] Dataset files already present in data/")
        for f in REQUIRED:
            path = os.path.join(DATA_DIR, f)
            size_mb = os.path.getsize(path) / 1_048_576
            print(f"    {f}: {size_mb:.1f} MB")
        return

    print("Dataset files not found. Attempting download …\n")
    print("Option A — Automatic (Kaggle API):")
    print("  1. Install kaggle: pip install kaggle")
    print("  2. Download API token from https://www.kaggle.com/settings → API → Create Token")
    print("  3. Place kaggle.json in C:\\Users\\<you>\\.kaggle\\")
    print("  4. Run this script again.\n")
    print("Option B — Manual:")
    print("  1. Go to: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    print("  2. Download the ZIP, extract Fake.csv and True.csv")
    print(f"  3. Place both files in: {DATA_DIR}\n")

    choice = input("Attempt automatic download now? [y/N]: ").strip().lower()
    if choice == "y":
        download_via_kaggle_api()
    else:
        print("Please download manually as described above.")


if __name__ == "__main__":
    main()
