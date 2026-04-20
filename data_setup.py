import os
import zipfile
import urllib.request
import subprocess

# Define the local directories where datasets will be stored
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# ========================================================
# FILL IN THE BLANKS BELOW ONCE YOU HAVE THE LINKS/COMMANDS
# ========================================================
# Example: "kaggle datasets download -d user/yolo-dataset"
KAGGLE_YOLO_CMD = "" 
KAGGLE_DEPTH_CMD = ""
KAGGLE_TTS_CMD = ""

# Example: "https://example.com/dataset.zip"
DIRECT_URL_YOLO = ""
DIRECT_URL_DEPTH = ""
DIRECT_URL_TTS = ""
# ========================================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download_via_kaggle(cmd, dest_folder):
    """Executes a Kaggle CLI command to download the dataset to a specific folder."""
    if not cmd:
        print(f"Skipping Kaggle download for {dest_folder} (No command provided).")
        return

    print(f"Downloading via Kaggle into {dest_folder}...")
    try:
        # cd into the dest_folder and run the command
        subprocess.run(cmd, shell=True, check=True, cwd=dest_folder)
        print("Download successful. Attempting to unzip if necessary...")
        _unzip_all_in_folder(dest_folder)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading via Kaggle CLI: {e}")
        print("Ensure you have Kaggle CLI installed and authenticated (kaggle.json).")


def download_via_url(url, dest_folder, filename="dataset.zip"):
    """Downloads a file directly from a URL."""
    if not url:
        print(f"Skipping Direct URL download for {dest_folder} (No URL provided).")
        return

    file_path = os.path.join(dest_folder, filename)
    print(f"Downloading from {url} to {file_path}...")
    try:
        urllib.request.urlretrieve(url, file_path)
        print("Download successful. Unzipping...")
        _unzip_file(file_path, dest_folder)
    except Exception as e:
        print(f"Error downloading from URL: {e}")


def _unzip_file(zip_path, extract_to):
    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted {zip_path}")
        os.remove(zip_path)  # Clean up zip


def _unzip_all_in_folder(folder):
    for f in os.listdir(folder):
        if f.endswith('.zip'):
            _unzip_file(os.path.join(folder, f), folder)


def setup_datasets():
    print("Starting Dataset Setup...")
    ensure_dir(DATASETS_DIR)

    # 1. Object Detection (YOLO)
    yolo_dir = os.path.join(DATASETS_DIR, "object_detection")
    ensure_dir(yolo_dir)
    print("\n--- Setting up YOLO Dataset ---")
    if KAGGLE_YOLO_CMD:
        download_via_kaggle(KAGGLE_YOLO_CMD, yolo_dir)
    elif DIRECT_URL_YOLO:
        download_via_url(DIRECT_URL_YOLO, yolo_dir)

    # 2. Depth Estimation Dataset
    depth_dir = os.path.join(DATASETS_DIR, "depth_estimation")
    ensure_dir(depth_dir)
    print("\n--- Setting up Depth Estimation Dataset ---")
    if KAGGLE_DEPTH_CMD:
        download_via_kaggle(KAGGLE_DEPTH_CMD, depth_dir)
    elif DIRECT_URL_DEPTH:
        download_via_url(DIRECT_URL_DEPTH, depth_dir)

    # 3. Custom TTS Dataset
    tts_dir = os.path.join(DATASETS_DIR, "text_to_speech")
    ensure_dir(tts_dir)
    print("\n--- Setting up Custom TTS Dataset ---")
    if KAGGLE_TTS_CMD:
        download_via_kaggle(KAGGLE_TTS_CMD, tts_dir)
    elif DIRECT_URL_TTS:
        download_via_url(DIRECT_URL_TTS, tts_dir)

    print("\nDataset Setup Complete. Check the 'datasets/' directory.")

if __name__ == "__main__":
    setup_datasets()
