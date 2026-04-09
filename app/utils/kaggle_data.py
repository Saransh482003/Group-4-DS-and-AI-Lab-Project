import os
import subprocess
import json
from app.config.pipeline_schema import KaggleSettings

def load_kaggle_secrets():
    """Load Kaggle credentials from kaggle_secrets.txt at the root."""
    from app.utils.paths import get_base_dir
    secrets_path = get_base_dir() / "kaggle_secrets.txt"
    if secrets_path.exists():
        with open(secrets_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

def check_kaggle_credentials() -> bool:
    load_kaggle_secrets()
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if os.environ.get("KAGGLE_API_TOKEN"): # Optional: support the user's specific token key
        return True
    return False

def ensure_egoblind_dataset(settings: KaggleSettings) -> str:
    from app.utils.paths import get_base_dir

    base_dir = get_base_dir()
    cache_root = base_dir / settings.local_cache_root / settings.extracted_folder_name
    extracted_path = cache_root / "extracted"
    raw_path = cache_root / "raw"
    stamp_path = cache_root / "download_stamp.json"

    if stamp_path.exists() and extracted_path.exists():
        print(f"Dataset already cached at {extracted_path}")
        return str(extracted_path)

    if not settings.auto_download_if_missing:
        print("Dataset missing and auto-download is disabled.")
        return ""

    if not check_kaggle_credentials():
        print("Error: Kaggle API credentials not found. Please set ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY.")
        return ""

    print(f"Downloading dataset {settings.dataset_slug}...")

    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(extracted_path, exist_ok=True)

    try:
        # Use Kaggle CLI to download
        cmd = ["kaggle", "datasets", "download", "-d", settings.dataset_slug, "-p", str(raw_path), "--unzip"]
        # On Windows, shell=True is often needed to find the .exe in the environment's Scripts folder
        subprocess.run(
            cmd,
            check=True,
            shell=(os.name == 'nt')
        )

        # If the zip is downloaded and unzipped by kaggle directly into raw, we should move it or just use it.
        # Sometimes kaggle --unzip puts it in raw. Let's move it to extracted.
        for item in os.listdir(raw_path):
            if item.endswith('.zip'):
                continue # kaggle shouldn't leave the zip with --unzip, but just in case
            import shutil
            src = os.path.join(raw_path, item)
            dst = os.path.join(extracted_path, item)
            if os.path.isdir(src):
                shutil.move(src, dst)
            else:
                shutil.move(src, dst)

        # Write stamp
        with open(stamp_path, "w") as f:
            json.dump({"slug": settings.dataset_slug, "downloaded": True}, f)

        print(f"Dataset downloaded and extracted to {extracted_path}")
        return str(extracted_path)

    except subprocess.CalledProcessError as e:
        print(f"Failed to download dataset via Kaggle CLI: {e}")
        return ""
    except Exception as e:
        print(f"Error handling dataset: {e}")
        return ""
