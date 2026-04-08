# Action-Oriented Indoor Navigation Assistance for the Visually Impaired

## Overview

Visually impaired individuals navigating indoor environments often rely
on assistive technologies that announce detected objects. However,
object identification alone does not indicate whether an obstacle lies
directly in the user's walking path or requires an immediate change in
movement.

For example, hearing _"chair detected"_ does not inform the user whether
they are about to collide with it or can safely continue forward.

This project focuses on **action-oriented indoor navigation
assistance**, where visual perception is converted into **movement
guidance rather than descriptive awareness**.

Instead of only reporting objects, the system provides **directional
instructions such as "move left" or "move right" when obstacles pose a
collision risk.**

---

# Problem Statement

Existing assistive systems typically perform **object detection and
verbal announcements**, but they do not:

- Identify whether the object lies **within the user's walking path**
- Estimate **collision risk**
- Provide **immediate movement guidance**

As a result, visually impaired users may still face difficulty
determining how to safely navigate around obstacles.

---

# Proposed Solution

The system performs **real-time perception and rule-based reasoning** to
convert visual inputs into **actionable navigation instructions**.

The solution focuses specifically on two indoor navigation risks:

1.  **Obstacle Collision**
2.  **Blocked Walking Path**

within a **near-field range of 0--3 meters**.

The system uses:

- **Webcam input** (proof-of-concept)
- **Real-time object detection**
- **Monocular depth estimation**
- **Rule-based spatial reasoning**

to generate **concise navigation prompts**.

---

# System Architecture

```
Webcam Input
     │
     ▼
Object Detection Model
     │
     ▼
Monocular Depth Estimation
     │
     ▼
Spatial Risk Assessment
     │
     ▼
Rule-Based Reasoning Engine
     │
     ▼
Navigation Guidance
(Audio Output)
```

---

# Key Concepts

### Walking Zone

The **walking zone** represents the **central region of the camera
frame**, corresponding to the user's forward movement path.

### Collision Risk

An obstacle is considered a **collision risk** when:

- It lies **within the walking zone**
- Its **distance falls below a predefined threshold**

A **collision** is defined as **any physical contact between the
participant and an obstacle during navigation.**

### Directional Guidance

If the walking zone is blocked, the system evaluates **lateral zones**
to determine a safer path and generates instructions such as:

- **Move Left**
- **Move Right**

---

# Reasoning Framework

The navigation decisions are generated using an **explicit rule-based
reasoning system**.

Rules rely on:

- Object distance from the user
- Spatial overlap with the walking zone
- Lateral clearance in adjacent zones

This approach ensures:

- **Interpretability**
- **Deterministic behavior**
- **Feasibility within the project timeline**

---

# Baseline Comparison

To evaluate the effectiveness of action-oriented navigation guidance,
the proposed system will be compared with a **baseline system**.

System Output

---

Baseline Object labels only (e.g., "Chair detected")
Proposed Directional instructions (e.g., "Move Left")

Both systems will use the **same detection and depth estimation models**
to ensure fair comparison.

---

# Experimental Setup

Evaluation will be conducted in a **controlled indoor obstacle course**.

### Environment

- Indoor path length: **10 meters**
- **5 standardized obstacles**

### Participants

- **Minimum 5 blindfolded participants**
- Blindfolding ensures **consistent testing conditions**

### Trials

- **3 trials per system mode per participant**
- Total: **15+ navigation trials**

Multiple trials help reduce **learning effects and randomness**.

---

# Evaluation Metrics

System performance will be evaluated using the following metrics:

- **Collision Count** -- Number of physical contacts with obstacles.
- **Navigation Completion Time** -- Total time required to complete
  the obstacle course.
- **Corrective Stops** -- A stop is counted when the participant halts
  ≥3 seconds, steps backward, or requires manual intervention.
- **Reaction Time** -- Time between audio instruction delivery and
  user response movement.

---

# Latency and Usability Requirements

End-to-end latency is measured from:

Obstacle enters walking zone → System processes scene → Navigation
instruction delivered

The system must satisfy two usability criteria:

1.  Users should **react smoothly without abrupt stopping**
2.  Users should **walk continuously without frequent pauses caused by
    delayed feedback**

---

# Implementation Constraints

- **Offline execution**
- **Webcam-based input**
- **Audio feedback via earphones**
- Indoor navigation only
- Near-field obstacle detection (0--3 meters)

---

# Project Objective

The objective of this project is to evaluate whether **translating
visual perception into actionable navigation guidance** can improve
**safe indoor mobility for visually impaired individuals** compared to
traditional object announcement systems.

---

# Repository Structure

```
project-root
│
├── app
│   └── app.py                     # Main application script
│
├── data
│   └── dataset_links.md           # Dataset references and links
│
├── docs
│   └── problem-statement.pdf      # Project problem statement
│
├── feedbacks
│   ├── Milestone1.md              # Feedback for milestone 1
│   └── Milestone2.md              # Feedback for milestone 2
│
├── Review Meeting PPTs
│   ├── Milestone1_Presentation.md              # Presentation for milestone 1
│   └── Milestone2_Presentation.md              # Presentation for milestone 2
│
├── reports
│   ├── Milestone1.pdf             # Milestone 1 report
│   └── Milestone2.pdf             # Milestone 2 report
│
├── CHANGELOG.md                   # Contribution log of team members
│
└── README.md                      # Project documentation
```

---

# How to Run

This project uses **Poetry** for dependency management, and instructions are provided for Windows PowerShell.

### 1. Install Poetry (Windows)

If you do not have Poetry installed, you can install it via PowerShell:

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

After installation, add the Poetry `bin` directory to your PATH (e.g., `%APPDATA%\Python\Scripts` or `%USERPROFILE%\AppData\Roaming\Python\Scripts`). You can verify it by running `poetry --version`.

### 2. Model Setup & Dependencies

This system requires external models that are not included in the main repository due to their size.

#### A. Depth-Anything-V2 Repository (Optional)

You should clone the Depth-Anything-V2 repository into the project root:

```powershell
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
```

#### B. TTS Setup (Piper)

The Piper TTS engine is required for audio feedback. If you see `app/piper_windows_amd64.zip`, extract it to the project root so you have a `piper/` folder containing `piper.exe`.

```powershell
Expand-Archive -Path "app\piper_windows_amd64.zip" -DestinationPath "." -Force
```

#### C. Model Weights

Weights should be placed in `model_training/`:

- **YOLO Weights**: `model_training/object_detection/best-weights/YOLOv8n-uni.pt`
- **Depth Weights**: `model_training/depth_estimation/model_weights/depth_anything_v2_metric_hypersim_vits.pth`

You can override weights via CLI flags (see below).

### 3. Install Python Dependencies

Once Poetry is installed and models are set up, install the Python library dependencies:

```powershell
poetry install
```

---

### Alternative: Standard `pip` & `venv` (No Poetry)

If you prefer not to use Poetry, you can use standard Python tools. Since the project uses PEP 621, `pip` can install dependencies directly from `pyproject.toml`.

**1. Create & Activate Virtual Environment:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**2. Install Dependencies:**
```powershell
pip install .
```

**3. Run the Application:**
Simply omit `poetry run` from any command:
```powershell
python -m app.main --mode live
```

---

#### D. Kaggle API for Datasets

To evaluate the system using datasets like `egoblind`, you need to configure your Kaggle API credentials:

1. Log in to your Kaggle account and go to [Settings](https://www.kaggle.com/settings).
2. Click "Create New API Token" to download `kaggle.json`.
3. You can either place this file in `~/.kaggle/kaggle.json` or open your `.env` file and set the `KAGGLE_USERNAME` and `KAGGLE_KEY` values directly.

### 3. CLI Execution

The main application entry point is `app/main.py`. You can run it via the CLI using Poetry:

**Live Mode (Webcam):**

```powershell
poetry run python -m app.main --mode live
```

_Note: If your camera doesn't open, ensure `VIDEO_SOURCE=0` in `.env` (or use the flag `--source-path 0`)._

**Dataset Evaluation Mode:**

```powershell
poetry run python -m app.main --mode dataset_eval --dataset egoblind
poetry run python -m app.main --mode dataset_eval --source-path data_cache/kaggle/egoblind-short-context-frames/extracted/...
```

**Benchmark Mode:**

```powershell
poetry run python -m app.main --mode benchmark --dataset egoblind --execution-mode sequential
poetry run python -m app.main --mode benchmark --dataset egoblind --execution-mode threaded_parallel
```

### 4. Supported CLI Flags

The CLI supports various flags to customize the execution:

- `--max-frames`: Limit the maximum number of frames to process.
- `--stride`: Set the frame stride for dataset evaluation.
- `--enable-tts`: Enable text-to-speech audio output.
- `--save-annotated-video`: Save the processed and annotated video output.
- `--show-windows`: Display the visualizer windows during execution.
- `--yolo-weights`: Specify a custom YOLO weight file (.pt). Falls back to `yolov8n.pt` if missing.
- `--output-dir`: Set the directory where output artifacts will be saved.

### In-Depth: Frame Sampling (`--max-frames` & `--stride`)

When evaluating the system on large datasets or video files, you can control the sampling density using the `--max-frames` and `--stride` flags.

#### 1. Frame Stride (`--stride`)

The **stride** determines how many frames the system skips before processing the next one.

- **Stride = 1**: Processes every single frame (highest fidelity, slowest).
- **Stride = 5**: Processes every 5th frame (jumps 4 frames, faster).
- **Impact**: Higher stride values significantly reduce processing time while still providing a representative look at the sequence. In **Video Databases**, this allows the system to "fast-forward" through the footage. In **Image Databases** (frame folders), it simply skips alphabetical files.

#### 2. Max Frames (`--max-frames`)

This flag limits the **total number of frames actually processed** by the pipeline.

- **Example**: If you set `--max-frames 100`, the system will stop immediately after a successful processing of 100 frames, regardless of the dataset size.

#### 3. How they combine

The flags work together to define your test coverage:

- **`--stride 5 --max-frames 100`**: The system will jump through the dataset in steps of 5 frames. It will stop once it has processed 100 frames. This means it effectively "looks" at 500 total frames of raw video/sequence but only runs inference on 100 of them.
- **Relationship**: `Total Source Frames Inspected = Stride * Max Frames`.

#### 4. Effects on different Database Types

| Database Type    | Stride Effect                           | Max Frames Effect                            |
| :--------------- | :-------------------------------------- | :------------------------------------------- |
| **Image Folder** | Skips `N-1` files in the directory.     | Stops after `M` files are processed.         |
| **Video File**   | Skips `N-1` frames in the video stream. | Stops after `M` frames are processed.        |
| **Live Webcam**  | Ignored (system runs in real-time).     | Closes the app after `M` frames are handled. |

---

# Automation with YAML Configurations

To simplify the execution of complex experiments and benchmarks, the system supports YAML-based configuration files.

### 1. Unified Runner Profiles (`--runner-config`)

Instead of typing long CLI commands, you can consolidate all your parameters (mode, dataset, limits, feature toggles) into a single profile in the `e2e_runner_configs/` directory.

**Example execution:**

```powershell
# Run a pre-defined live webcam test
poetry run python -m app.main --runner-config e2e_runner_configs/live_webcam.yaml

# Run a custom video benchmark sweep
poetry run python -m app.main --runner-config e2e_runner_configs/custom_video_benchmark.yaml
```

### 2. Customizing Benchmark Suites (`--benchmark-config`)

The **Benchmark Mode** runs a sweep of several pipeline configurations. You can customize exactly which combinations are tested by editing `app/config/benchmark_suite.yaml`.

### 3. Filtering Benchmark Targets (`--benchmark-target`)

If you only care about specific configurations from your suite, you can use the `--benchmark-target` flag to skip the others and run only the named targets:

```powershell
# Only run two specific modes from the suite
poetry run python -m app.main --mode benchmark --dataset egoblind --benchmark-target detection_only parallel_full_with_tts
```

---

---

# End-to-End Testing (ScanNet / Ego4D)

For high-fidelity testing of the navigation pipeline, you can use continuous sequence datasets like **ScanNet** (indoor walking) or **Ego4D** (first-person videos).

### 1. Generating Mock Data for Testing

If you don't have the full ScanNet dataset, you can generate a small mock sequence to test the pipeline:

```powershell
poetry run python scripts/setup_datasets.py --type scannet_mock
```

### 2. Running ScanNet Evaluation

ScanNet expects a folder containing scene subfolders (with `color/` and `depth/` subdirs):

```powershell
poetry run python -m app.main --mode dataset_eval --dataset scannet --source-path data_cache/test_datasets/scannet
```

### 3. Running Ego4D or Custom Video

Ego4D videos can be processed as a "live" stream (real-time simulation):

```powershell
poetry run python -m app.main --mode live --source-path data_cache/Custom_Videos/indore_test_divyang_vid.mp4
```

---

# Custom Depth Engine Integration

The pipeline is modular and supports swapping the default depth estimator for a custom one.

### 1. Where to Implement
All depth estimation logic is encapsulated in:
👉 `app/mechanics/depth_estimation.py`

### 2. Implementation Interface
To plug in a new engine, the custom class must implement the following interface:

```python
### 2. Implementation Interface (Multi-implementation example)
If multiple team members (e.g., Rohit, Samyuktha) are testing different engines, you can define them as separate classes and choose the active one.

**File to edit: `app/mechanics/depth_estimation.py`**

```python
# app/mechanics/depth_estimation.py

class DepthEstimatorRohit:
    def predict(self, frame): 
        # Rohit's logic
        return d, c

class DepthEstimatorSamyuktha:
    def predict(self, frame):
        # Samyuktha's logic
        return d, c

# ACTIVE IMPLEMENTATION
DepthEstimator = DepthEstimatorRohit  # Just swap this line to switch engines
```
```

### 3. Graceful Skipping
If the external `Depth-Anything-V2` dependencies are not found, the system will automatically alert the user and continue running ONLY the detection and navigation logic.

---


# Testing Requirements

Tests have been added for the new architecture. You can run the test suite using `pytest`:

```powershell
poetry run pytest tests/
```

### Covered Test Areas

- **`test_frame_sources.py`**:
  - Video source initialization
  - Frame folder ordering
  - Graceful empty folder handling
- **`test_pipeline_executor.py`**:
  - Sequential executor basic behavior
  - Threaded parallel executor basic behavior
  - Merge of detector and depth outputs
- **`test_metrics_aggregation.py`**:
  - Average/median/p95 calculations
  - Command distribution
  - Slowest frame extraction
  - Top risk frame extraction
- **`test_kaggle_data.py`**:
  - Path resolution
  - Cache hit logic
  - No-download when cache exists
  - Command construction for Kaggle download
  - Unzip/extract logic with mocks

External systems and models are mocked within tests to ensure reliable and fast execution.

---

# Team Details

**Group Number:** 4\
**Client:** Mr. M Nagarajan\
**Number of Members:** 5

| Name               | Email                                                                   | GitHub           |
| ------------------ | ----------------------------------------------------------------------- | ---------------- |
| Saransh Saini      | [22f1001123@ds.study.iitm.ac.in](mailto:22f1001123@ds.study.iitm.ac.in) | Saransh482003    |
| Divyang Panchasara | [22f1000411@ds.study.iitm.ac.in](mailto:22f1000411@ds.study.iitm.ac.in) | 22f1000411       |
| Samyuktha Shriram  | [22f2001444@ds.study.iitm.ac.in](mailto:22f2001444@ds.study.iitm.ac.in) | SamyukthaSh24    |
| Prasoon Shukla     | [23f3003434@ds.study.iitm.ac.in](mailto:23f3003434@ds.study.iitm.ac.in) | 23f3003434       |
| Rohit Prajapat     | [22f1001536@ds.study.iitm.ac.in](mailto:22f1001536@ds.study.iitm.ac.in) | rohitblpprajapat |

---

# Citations

```bibtex
@inproceedings{xiao2025egoblind,
  title={EgoBlind: Towards Egocentric Visual Assistance for the Blind},
  author={Xiao, Junbin and Huang, Nanxin and Qiu, Hao and Tao, Zhulin and Yang, Xun and Hong, Richang and Wang, Meng and Yao, Angela},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```
