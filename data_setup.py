import os
import zipfile
import urllib.request
import subprocess
import kagglehub

# Define the local directories where datasets will be stored
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# NYU Depth V2 MAT File URL
DIRECT_URL_DEPTH = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
DIRECT_URL_SPLITS = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/splits.mat"


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
    yolo_dir = os.path.join(DATASETS_DIR, "dsai-unified")
    ensure_dir(yolo_dir)
    print("\n--- Setting up YOLO Dataset ---")
    try:
        print("Downloading dsai-unified-dataset via kagglehub...")
        path = kagglehub.dataset_download("ds22f1001123/dsai-unified-dataset")
        print(f"Dataset successfully downloaded to: {path}")
    except Exception as e:
        print(f"Failed to download YOLO dataset via kagglehub: {e}")

    # 2. Depth Estimation Dataset
    depth_dir = os.path.join(DATASETS_DIR, "depth_nyu_dataset")
    ensure_dir(depth_dir)
    print("\n--- Setting up Depth Estimation Dataset ---")
    mat_file_path = os.path.join(depth_dir, "nyu_depth_v2_labeled.mat")
    splits_file_path = os.path.join(depth_dir, "splits.mat")
    
    if not os.path.exists(mat_file_path):
        download_via_url(DIRECT_URL_DEPTH, depth_dir, filename="nyu_depth_v2_labeled.mat")
    else:
        print("NYU Depth V2 MAT file already exists. Skipping download.")
        
    if not os.path.exists(splits_file_path):
        download_via_url(DIRECT_URL_SPLITS, depth_dir, filename="splits.mat")
    else:
        print("NYU splits.mat file already exists. Skipping download.")
        
    try:
        from scipy.io import loadmat
        import h5py
        
        print("Verifying Depth data configurations using Eigen split...")
        
        # Load test split (Eigen split: what we had used)
        splits = loadmat(splits_file_path)
        test_idx = splits["testNdxs"].flatten() - 1
        
        # Load the dataset
        with h5py.File(mat_file_path, 'r') as f:
            rgb_data = f["images"]    # Shape should be (~1449, 3, 640, 480)
            depth_data = f["depths"]  # Shape should be (~1449, 640, 480)
            
            print(f"Original RGB Shape: {rgb_data.shape}, Original Depth Shape: {depth_data.shape}")
            print(f"Verified Test Indices Count: {len(test_idx)}")
            print("Note: For each sample, RGB is stored as (3, H, W) -> we transpose to (H, W, 3). Depth is (H, W) and already aligned.")
    except Exception as e:
        print("Failed to verify depth matrices. Ensure scipy and h5py are installed.", e)

    # 3. Custom TTS Dataset
    tts_dir = os.path.join(DATASETS_DIR, "text_to_speech")
    ensure_dir(tts_dir)
    print("\n--- Setting up Custom TTS Dataset ---")
    print("TTS dataset consists of 165 samples compiled from Navigation Commands, CMU Arctic, LJ Speech, and LibriSpeech as detailed in the Milestone 6 report.")
    print("Generating a reference JSON...")
    import json
    tts_metadata = {
        "description": "Text-to-Speech Baseline Evaluation Data",
        "total_samples": 165,
        "datasets": ["Navigation Commands (15)", "CMU Arctic (50)", "LJ Speech (50)", "LibriSpeech Test Clean (50)"]
    }
    with open(os.path.join(tts_dir, "tts_metadata.json"), "w") as f:
        json.dump(tts_metadata, f, indent=4)

    print("\nDataset Setup Complete. Check the 'datasets/' directory.")

if __name__ == "__main__":
    setup_datasets()
