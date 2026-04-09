# Developer Notes: Dataset & Model Setup

This document provides a consolidated guide for developers to set up the real-world datasets and model weights required for the indoor navigation pipeline.

---

## 🚀 1. Automated Setup (Recommended)

Use the provided setup script to handle automated downloads and mock data generation.

### A. Download Model Weights
Downloads the required monocular depth estimation and object detection weights.
```powershell
poetry run python scripts/setup_datasets.py --type download_weights
```

### B. Download EgoBlind (Kaggle)
Downloads the short-context EgoBlind frames using the Kaggle CLI.
*Note: Ensure your Kaggle API key is configured in `.env` or `~/.kaggle/kaggle.json`.*
```powershell
poetry run python scripts/setup_datasets.py --type download_egoblind
```

### C. Generate Mock ScanNet Data
Generates a small synthetic ScanNet-like folder structure for rapid pipeline testing without large downloads.
```powershell
poetry run python scripts/setup_datasets.py --type scannet_mock
```

---

## ℹ️ 2. Official Dataset Instructions (Manual)

Because of licensing and size, ScanNet and Ego4D must be registered for individually.

### ScanNet
1.  **Register:** Visit [http://www.scan-net.org/](http://www.scan-net.org/).
2.  **Agreement:** Sign the Terms of Use and email it to `scannet@googlegroups.com`.
3.  **Download:** Use the provided official Python script to download specific scenes.
4.  **Structure:** Organize frames as: `data_cache/test_datasets/scannet/sceneXXXX_YY/color/*.jpg`.

### Ego4D
1.  **Register:** Create an account at [https://ego4d-data.org/](https://ego4d-data.org/) or  [https://ego4d.dev/request/ego4d](https://ego4d.dev/request/ego4d).
2.  **Install CLI:** `pip install ego4d`
3.  **Download:** Run `ego4d --datasets <subset> --out_dir data_cache/ego4d`.
4.  **Usage:** Use the resulting `.mp4` files as a `live` source in the main application.

---

## 📊 3. End-to-End Evaluation Commands

Once data is set up, use these commands to evaluate the pipeline:

### Test with ScanNet (E2E)
```powershell
poetry run python -m app.main --mode dataset_eval --dataset scannet --source-path data_cache/test_datasets/scannet
```

### Test with Ego4D / Video (Live Simulation)
```powershell
poetry run python -m app.main --mode live --source-path path/to/video.mp4
```

### Test with EgoBlind (Frame Folders)
```powershell
poetry run python -m app.main --mode dataset_eval --dataset egoblind
```

---

## ⚙️ 5. Automation & Configurations (YAML)

To simplify the execution of complex experiments and eliminate long CLI strings, the system supports unified configuration files in the `e2e_runner_configs/` folder.

### A. Run Command Format
Use the `--runner-config` flag to load a YAML profile. Every YAML file includes its specific run command as a comment at the bottom for easy copy-pasting.
```powershell
poetry run python -m app.main --runner-config e2e_runner_configs/your_config.yaml
```

### B. Standard Templates
| Config File | Use Case |
| :--- | :--- |
| `live_webcam.yaml` | Real-time navigation with webcam and TTS. |
| `egoblind_benchmark.yaml` | Performance sweep (8 modes) on the EgoBlind dataset. |
| `custom_video_eval.yaml` | Single evaluation pass on custom recorded videos. |
| `custom_video_benchmark.yaml`| Performance sweep (8 modes) on custom videos. |
| `scannet_parallel_bench.yaml`| High-fidelity testing of the **Threaded Parallel Executor**. |

### C. Configuration Parameters
| Parameter | Description |
| :--- | :--- |
| `mode` | `live`, `dataset_eval`, or `benchmark`. |
| `dataset` | `egoblind`, `scannet`, or `ego4d` (handles auto-pathing). |
| `source_path` | Webcam ID (`0`), or path to a video file / frame folder. |
| `execution_mode` | `sequential` or `threaded_parallel`. |
| `max_frames` | Stops processing and exports results after $M$ frames. |
| `stride` | Only processes every $N$-th frame (helps performance). |
| `enable_tts` | Toggles text-to-speech audio feedback. |
| `save_annotated_video`| Saves annotated output to `results/` folder. |

### D. Benchmark Suite Customization
The performance sweep logic is defined in `app/config/benchmark_suite.yaml`. You can customize the combinations or use `--benchmark-target` to filter them:
```powershell
# Only run the most critical modes from the suite
poetry run python -m app.main --mode benchmark --benchmark-target detection_only parallel_full_with_tts
```

---

## 🛠️ 6. Debugging & Results

*   **Outputs:** Every run creates a timestamped folder in `outputs/`.
*   **Results Folder:** Annotated videos are automatically copied to the central `results/` folder for convenience.
*   **Metrics:** Check `outputs/run_name_timestamp/frame_metrics.csv` for per-frame FPS and navigation decisions.
*   **Dashboard:** Use `--save-annotated-video` to generate a playback of the run with bboxes and depth maps overlaid.
*   **Source Issues:** If the camera or video fails to open, verify the `--source-path` (e.g. `0` for default webcam, or absolute path for files).

