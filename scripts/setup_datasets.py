import os
import cv2
import numpy as np
import argparse
import sys
from app.config.settings import load_config
from app.utils.kaggle_data import ensure_egoblind_dataset

def create_mock_scannet(root_dir: str, num_scenes: int = 1, frames_per_scene: int = 20):
    """
    Creates a mock ScanNet directory structure with dummy images for testing.
    """
    print(f"Creating mock ScanNet dataset at {root_dir}...")
    os.makedirs(root_dir, exist_ok=True)

    for i in range(num_scenes):
        scene_id = f"scene{i:04d}_00"
        scene_path = os.path.join(root_dir, scene_id)
        color_path = os.path.join(scene_path, "color")
        depth_path = os.path.join(scene_path, "depth")

        os.makedirs(color_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)

        print(f"  Generating {scene_id}...")
        for f in range(frames_per_scene):
            # Create a dummy image with some movement
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw a moving "obstacle"
            cv2.rectangle(img, (100 + f*10, 200), (200 + f*10, 400), (0, 0, 255), -1)
            cv2.putText(img, f"Mock Frame {f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imwrite(os.path.join(color_path, f"{f}.jpg"), img)
            
            # Dummy depth (constant for now)
            depth = np.full((480, 640), 1000, dtype=np.uint16) # 1 meter
            cv2.imwrite(os.path.join(depth_path, f"{f}.png"), depth)

    print("Mock ScanNet created successfully.")

def download_egoblind_from_kaggle():
    """Download EgoBlind dataset using Kaggle API."""
    print("Attempting to download EgoBlind from Kaggle...")
    config = load_config()
    resolved_path = ensure_egoblind_dataset(config.kaggle)
    if resolved_path:
        print(f"EgoBlind dataset is ready at: {resolved_path}")
    else:
        print("Failed to download EgoBlind. Make sure Kaggle API is configured in your .env or ~/.kaggle/kaggle.json")

def print_scannet_instructions():
    """Print official ScanNet download instructions."""
    print("\n--- ScanNet Official Download Instructions ---")
    print("1. Visit: http://www.scan-net.org/")
    print("2. Sign the Terms of Use agreement and email it to scannet@googlegroups.com")
    print("3. Once you receive the download script (download-scannet.py), run it to get the 'color' and 'depth' frames.")
    print("4. Place the scene folders in 'data_cache/test_datasets/scannet/'")
    print("   Example: data_cache/test_datasets/scannet/scene0000_00/color/*.jpg")
    print("---------------------------------------------\n")

def print_ego4d_instructions():
    """Print official Ego4D download instructions."""
    print("\n--- Ego4D Official Download Instructions ---")
    print("1. Create an account at https://ego4d-data.org/")
    print("2. Install the cli: `pip install ego4d`")
    print("3. Run the downloader for a subset (e.g. 'forecasting'):")
    print("   `ego4d --datasets forecasting --out_dir data_cache/ego4d` --video_dir data_cache/ego4d/videos`")
    print("4. Use the downloaded videos as a 'live' source in app.main.")
    print("--------------------------------------------\n")

def download_model_weights():
    """Download required model weights."""
    import urllib.request
    
    weights = [
        {
            "name": "Depth-Anything-V2 Weights (Small/Metric)",
            "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth?download=true",
            "dest": "model_training/depth_estimation/model_weights/depth_anything_v2_metric_hypersim_vits.pth"
        }
    ]

    for w in weights:
        dest_dir = os.path.dirname(w["dest"])
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.exists(w["dest"]):
            print(f"{w['name']} already exists at {w['dest']}")
            continue
        
        print(f"Downloading {w['name']}...")
        try:
            urllib.request.urlretrieve(w["url"], w["dest"])
            print(f"Successfully downloaded to {w['dest']}")
        except Exception as e:
            print(f"Failed to download {w['name']}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Dataset setup and preprocessing script.")
    parser.add_argument("--type", type=str, choices=["scannet_mock", "ego4d_mock", "download_egoblind", "download_weights", "info_scannet", "info_ego4d"], default="scannet_mock")
    parser.add_argument("--output-dir", type=str, default="data_cache/test_datasets")
    
    args = parser.parse_args()
    
    # Ensure app is in python path
    sys.path.append(os.getcwd())

    if args.type == "scannet_mock":
        create_mock_scannet(os.path.join(args.output_dir, "scannet"))
    elif args.type == "download_egoblind":
        download_egoblind_from_kaggle()
    elif args.type == "download_weights":
        download_model_weights()
    elif args.type == "info_scannet":
        print_scannet_instructions()
    elif args.type == "info_ego4d":
        print_ego4d_instructions()
    else:
        print("Option not fully implemented yet.")

if __name__ == "__main__":
    main()
