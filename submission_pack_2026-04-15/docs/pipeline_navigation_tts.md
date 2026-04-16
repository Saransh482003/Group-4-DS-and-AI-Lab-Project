# Pipeline, TTS Engine, and Navigation Algorithm

## Pipeline (End-to-End)

Entry point:

- `python -m app.main` (recommended via `poetry run`)

Core stages (conceptual):

1. **Frame source**: webcam, video file, or frame folder
2. **Object detection**: YOLO weights `model_training/object_detection/best-weights/YOLOv8n-uni.pt`
3. **Monocular depth estimation**: Depth-Anything-V2 weights `model_training/depth_estimation/model_weights/depth_anything_v2_metric_hypersim_vits.pth`
4. **Fusion**: attach distance estimates to detections
5. **Navigation decision**: compute zone risks + choose a stable navigation command
6. **TTS (optional)**: announce navigation command (rate-limited + de-duplicated)
7. **Visualization (optional)**: overlays + annotated video export to `results/`
8. **Metrics export**: per-frame + summary CSV/JSON to `outputs/<run>/`

Execution modes:

- `sequential`: processes each stage in order
- `threaded_parallel`: runs some stages in parallel (configured via CLI/YAML)

## TTS Engine (Piper)

Engine:

- Piper executable: `piper/piper.exe`
- Voice model: `app/piper_voices/en_US-amy-medium.onnx`
- Voice config: `app/piper_voices/en_US-amy-medium.onnx.json`

Paths are controlled by `.env`:

- `PIPER_EXE_REL`
- `PIPER_VOICE_MODEL_REL`
- `PIPER_VOICE_CONFIG_REL`

TTS runtime behavior:

- Uses an async queue by default (so per-frame `tts_latency_ms` is mostly enqueue overhead).
- Gates speech to avoid spamming: only speaks on command change, and respects a minimum time interval.

## Navigation Algorithm (Deterministic Rule-Based)

Implementation:

- `app/mechanics/navigation_logic.py`

High-level idea:

- Split each frame into 3 horizontal zones: left, center, right.
- For each detected object, compute an object risk score from its estimated distance.
- Distribute that object risk into zones based on bounding-box overlap.
- Smooth zone risks using EMA to reduce jitter.
- Convert zone risks into a stable navigation command using thresholds + hysteresis.

Risk computation:

- Object risk increases when distance decreases (inverse-distance style).
- Zone risk is the sum of weighted object risks in that zone.

Stability logic:

- EMA smoothing: `ema_alpha` (default `0.3`)
- Turn command hold: prevents rapid left/right flipping (`command_hold_seconds`)
- Switch confirmation: requires repeated candidate command (`switch_confirm_frames`)
- Hysteresis margin: avoids oscillation (`turn_hysteresis`, `significant_margin`)

Commands generated (examples):

- `Path clear. Continue straight.`
- `Clear path on your left. Turn left.`
- `Clear path on your right. Turn right.`
- `Obstacle ahead. Move slightly left/right.`
- `Path blocked. Please scan left and right.`
- `Searching for path. Turn back.`

