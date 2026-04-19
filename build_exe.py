import os
import sys
import subprocess

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.join(root_dir, "app")
    app_script = os.path.join(app_dir, "app.py")
    
    # Define source paths
    yolo_src = os.path.join(root_dir, "model_training", "object_detection", "best-weights", "YOLO11s-Final-Training.pt")
    depth_src = os.path.join(root_dir, "model_training", "depth_estimation", "model_weights", "depth_anything_v2_metric_hypersim_vits.pth")
    piper_src = os.path.join(root_dir, "piper")
    piper_voices_src = os.path.join(root_dir, "piper", "piper_voices")
    depth_anything_src = os.path.join(root_dir, "Depth-Anything-V2")

    # The format for --add-data is "source;destination" on Windows, but let's use the CLI's standard format since we pass via Python
    add_data_args = [
        f"--add-data={yolo_src};model_weights",
        f"--add-data={depth_src};model_weights",
        f"--add-data={piper_src};piper",
        f"--add-data={piper_voices_src};piper_voices",
        f"--add-data={depth_anything_src};Depth-Anything-V2",
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name=DSAI_Nav_App",
        "--console",
        "--hidden-import=ultralytics",
        "--hidden-import=torch",
        "--hidden-import=cv2",
        "--hidden-import=depth_anything_v2",
        "--hidden-import=mechanics",
        f"--paths={depth_anything_src}",
        f"--paths={app_dir}",
    ] + add_data_args + [app_script]

    print("Running PyInstaller with command:")
    print(" ".join(cmd))
    
    # Run pyinstaller
    subprocess.run(cmd, cwd=root_dir)

if __name__ == "__main__":
    main()