# Action-Oriented Indoor Navigation Assistance for the Visually Impaired

## 1. Project Overview

Visually impaired individuals navigating indoor environments often rely on assistive technologies that announce detected objects. However, object identification alone does not indicate whether an obstacle lies directly in the user's walking path or requires an immediate change in movement. Hearing *"chair detected"* does not inform the user whether they are about to collide with it or can safely continue forward.

This project focuses on **action-oriented indoor navigation assistance**, where visual perception is converted into **movement guidance rather than descriptive awareness**. Instead of reporting objects, the system provides directional instructions such as *"Move Left"* or *"Move Right"* when obstacles pose a risk.

### Key Features
* **Real-time Object Detection:** Identifies indoor objects using custom-trained YOLO models.
* **Monocular Depth Estimation:** Calculates depth maps dynamically to gauge object proximity.
* **Spatial Risk Assessment:** Explicit rule-based reasoning engine decides if an object falls in the user's "Walking Zone."
* **Actionable Audio Guidance:** Converts decisions to near-instant Text-to-Speech (TTS) commands instead of listing obstacles.

### High-Level Architecture
```text
Webcam Input
     │
     ▼
Object Detection Model  ──►  Monocular Depth Estimation
                                  │
     ┌────────────────────────────┘
     ▼
Spatial Risk Assessment (Walking Zone Overlap Check)
     │
     ▼
Rule-Based Reasoning Engine (Lateral Clearance Calculation)
     │
     ▼
Navigation Guidance (Audio Output via Custom TTS)
```

---

## 2. Environment Setup

* **Python:** 3.9+ (Recommended 3.10)
* **OS:** Windows / Linux / macOS
* **Hardware Requirements:** CPU is supported, but a CUDA-enabled GPU (NVIDIA) is strongly recommended for real-time inference (YOLO + Depth). 

### Quick OS-Specific Setup

**For Windows Users:**
Run the batch script from the repository root. It will create a virtual environment (`env`), install dependencies, and setup the `.env` file.
```cmd
setup.bat
```

**For Linux/macOS Users:**
```bash
chmod +x setup.sh
./setup.sh
```

*Note: The setup scripts install dependencies from both `requirements.txt` and `requirements-hf.txt` as necessary.*

---

## 3. Configuration & Secrets

The application depends on environment variables to manage access tokens and file paths. 

### Setting up `.env`
Our setup scripts (`setup.bat` / `setup.sh`) will automatically generate a `.env` file from `.env.example`. Make sure you configure the following variables inside `.env`:

* `HF_TOKEN`: Your Hugging Face access token (required to download certain model checkpoints or push logs).
* `MODEL_DIR`: Custom directory if you wish to store the downloaded checkpoints somewhere other than the default paths.
* `CAMERA_INDEX`: Default camera index (typically `0` for the built-in webcam, `1` for external USB cameras).

---

## 4. Data & Inputs

The training pipeline utilizes multiple distinct datasets. *A dedicated dataset setup script (`data_setup.py`) is planned to automatically download and structure these datasets in the future.*

### Dataset Sources
* **Object Detection (YOLO):** Hosted on Kaggle [DSAI Unified Indoor Dataset](https://www.kaggle.com/datasets/ds22f1001123/dsai-unified-dataset). Downloadable via the `kagglehub` package.
* **Depth Estimation:** Based on the [NYU Depth V2 dataset](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat). The `.mat` (v7.3) array requires `h5py` for direct extraction.
* **Text-to-Speech (TTS):** 165 evaluation samples compiled and cross-benchmarked from Navigation Commands, CMU Arctic, LJ Speech, and LibriSpeech Test Clean (detailed in Milestone 6 pipeline report).
* **Evaluation Sequences:** Recorded navigation sequences hosted on Kaggle [LINK_PENDING]

### Data Format
* YOLO bounding box annotations (`.txt` format with `class x_center y_center width height`).
* Depth maps processed as normalized arrays corresponding to RGB inputs.

---

## 5. Running the Application (Local Deployment)

You have multiple ways to interact with the project: Real-time camera streaming, isolated image testing, or Docker deployment.

### A. Real-Time Streaming (Main Application)
To run the live navigation assistant with your webcam:
```bash
python app/app.py --camera 0
```
*(You can override the `.env` default camera by providing the `--camera` argument).* 

### B. Single Image Testing (Streamlit)
If you want to test the pipeline on static images, run the Streamlit interface:
```bash
streamlit run app/streamlit_app.py
```

### C. Gradio Web Interface
For an alternative interactive web UI, you can run the Hugging Face Gradio app:
```bash
python app/huggingface_app.py
```

*Note: A future executable script that automatically tunnels the Gradio/Streamlit UI via `ngrok` for public URL sharing is heavily requested and in the roadmap.*

---

## 6. Model / Pipeline Execution

### Working with Model Checkpoints
We have completed over 60 hyperparameter tuning runs for our custom models. 

> **Note on Checkpoint Hosting:** Storing all the heavily iterated model checkpoints in a Git repository bloats the repository size. Therefore, the models and their 60 tuning variations are hosted on Kaggle. 
> *Recommendation:* The absolute *best/final* weights (e.g., `best.pt`) should be hosted via GitHub Releases or Hugging Face Model Hub so users can download just the required weights seamlessly during inference startup.

### Inference Steps
1. Checkpoints should be placed in the `model_training/` subdirectories.
2. The pipeline handles initialization automatically when `app.py`, `huggingface_app.py`, or `streamlit_app.py` is invoked. 
3. During execution, it will log the current frames processed, detected bounding boxes overlaying the depth mask, and output audio cues.

---

## 7. End-to-End Reproducibility

To reproduce our results on a clean machine:
1. **Clone the repo:** `git clone https://github.com/Saransh482003/Group-4-DS-and-AI-Lab-Project.git && cd Group-4-DS-and-AI-Lab-Project`
2. **Run setup:** `./setup.bat` (or `setup.sh` on Linux)
3. **Set Secrets:** Edit the generated `.env` file with your `HF_TOKEN` and preferred `CAMERA_INDEX`.
4. **Download Models:** Copy the finalized `.pt` and `.onnx` models into the designated model folders (or wait for the automated script).
5. **Start system:** `python app/app.py`

---

## 8. Deployment Details

### Local Docker Deployment (Recommended)
If you have Docker installed, you can skip Python virtual environments entirely:
```bash
docker-compose up --build
```
* Access the web app locally at `http://localhost:7860`.

### Hosting Limitations
Due to real-time latency checks, deploying the webcam streaming script on a remote cloud server is not recommended. Video streaming latency through standard web protocols makes spatial guidance unsafe. Local or Edge deployment is mandatory for actual user testing.

---

## 9. Evaluation & Results

The reasoning rule engine is continually evaluated on both recorded video sequences and static benchmarks.

* **Evaluation Dataset:** Captured scenarios representing complex indoor obstacles.
* **Concrete Outputs:** Evaluation metrics (Collision counts, Navigation completion times, Corrective stops) and pipeline inferences are recorded and maintained in the `app/navigation_eval_outputs/` directory.
* *Real-Time Metrics:* Pending field-testing trials.
* **Limitations:** Current latency is highly hardware-dependent. Monocular depth generation can occasionally misrepresent specular reflections (glass/mirrors) as empty pathways.

---

## 10. Repository Structure

```
dsai_group4_project
│
├── app/                         # Core Application logic and web UIs
│   ├── mechanics/               # Submodules: Depth, Object Detect, TTS, Logic
│   ├── app.py                   # Main real-time streaming entrypoint
│   ├── streamlit_app.py         # Static image tester via Streamlit
│   ├── huggingface_app.py       # Static image tester via Gradio
│   └── navigation_eval_outputs/ # Concrete evaluation logs & results
│
├── data_sources/                # Data preparation & inspection notebooks
├── datasets/                    # Processing scripts & YAML definitions
├── Depth-Anything-V2/           # Depth Estimation submodule reference
├── model_training/              # Training logic for the model components
├── pipeline_evaluations/        # Scripts evaluating E2E pipeline accuracy
├── piper/                       # TTS (Text-to-Speech) dependencies
│
├── setup.bat / setup.sh         # Environment and dependency bootstrappers
├── requirements.txt             # Core python dependencies
├── docker-compose.yml           # Docker deployment definitions
├── CHANGELOG.md                 # Project contribution records
└── README.md                    # This documentation file
```

---

## 11. Troubleshooting

* **Webcam Not Opening / Blank Feed:** Check your `CAMERA_INDEX` in the `.env` file or pass it as an argument (`python app/app.py --camera 1`). Ensure privacy settings allow terminal/python apps to use the camera.
* **CUDA Out of Memory (OOM):** `Depth-Anything-V2` can be memory-intensive. Lower the processing resolution dynamically in the `.env` file or switch to CPU inference if you have <4GB VRAM.
* **Missing Dependencies:** Ensure you have activated your virtual environment before running (`env/Scripts/activate` on Windows, or `source env/bin/activate` on Linux) and ran the setup scripts properly.
* **Environment file missing:** Copy the `.env.example` to `.env` if the automatic setup script fails.

---

## 12. Contribution Summary

Detailed contribution records are logged iteratively inside [`CHANGELOG.md`](CHANGELOG.md).

**Team Details:**
**Group Number:** 4 | **Client:** Mr. M Nagarajan | **Number of Members:** 5

| Name               | Email                              | GitHub           |
| ------------------ | ---------------------------------- | ---------------- |
| Saransh Saini      | 22f1001123@ds.study.iitm.ac.in      | Saransh482003    |
| Divyang Panchasara | 22f1000411@ds.study.iitm.ac.in      | 22f1000411       |
| Samyuktha Shriram  | 22f2001444@ds.study.iitm.ac.in      | SamyukthaSh24    |
| Prasoon Shukla     | 23f3003434@ds.study.iitm.ac.in      | 23f3003434       |
| Rohit Prajapat     | 22f1001536@ds.study.iitm.ac.in      | rohitblpprajapat |

---

## 13. Future Improvements / Limitations

Our immediate and long-term milestones for this system include:

1. **Edge Deployment:** Porting the models (via quantization / TensorRT) to run efficiently on an edge device (e.g., Raspberry Pi or Jetson Nano), moving towards a functional wearable prototype.
2. **Latency & Accuracy Optimizations:** Further reducing the processing loop overhead. We heavily prioritize depth hazard accuracy as the source of truth for collision prevention, aiming for near-zero False Negatives.
3. **Public Dataset Setup Script:** Releasing scripts to fully automate downloading and compiling the Kaggle/NYU datasets into the repo structure.
4. **Ngrok Public Exposer:** Adding a one-click executable to expose the Gradio/Streamlit validation interfaces publicly via Ngrok.
5. **Research Publication:** Preparing our evaluation methodology and real-time inference metrics into a comprehensive research paper on assistive navigation technologies.
