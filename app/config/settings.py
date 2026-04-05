import os
from pathlib import Path
import torch

from app.config.pipeline_schema import AppConfig, PipelineSettings, ModelSettings, BenchmarkSettings, KaggleSettings
from app.utils.paths import env_rel_path, get_base_dir
from app.utils.env import load_env_file

def load_config() -> AppConfig:
    """Load configuration from environment variables and defaults."""
    load_env_file()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = PipelineSettings(
        source_type=os.getenv("PIPELINE_SOURCE_TYPE", "webcam"),
        source_path=os.getenv("VIDEO_SOURCE", "1"),
        execution_mode=os.getenv("EXECUTION_MODE", "sequential"),
        enable_tts=os.getenv("TTS_ENABLED", "1") == "1",
        show_windows=os.getenv("SHOW_WINDOWS", "1") == "1",
        plot_stream_mode=int(os.getenv("PLOT_STREAM_MODE", "0")),
        nav_logic_mode=int(os.getenv("NAV_LOGIC_MODE", "0")),
        warmup_frames=int(os.getenv("WARMUP_FRAMES", "15")),
        sample_resources_every_n_frames=int(os.getenv("RESOURCE_SAMPLE_EVERY_N_FRAMES", "10"))
    )

    yolo_weights = env_rel_path(
        "YOLO_WEIGHTS_REL",
        os.path.join("model_training", "object_detection", "best-weights", "YOLOv8n-uni.pt"),
    )

    depth_model = env_rel_path(
        "DEPTH_MODEL_DIR_REL",
        os.path.join(
            "model_training",
            "depth_estimation",
            "model_weights",
            "depth_anything_v2_metric_hypersim_vits.pth",
        ),
    )

    models = ModelSettings(
        yolo_weights_path=str(yolo_weights),
        depth_model_path=str(depth_model),
        device=device
    )

    return AppConfig(pipeline=pipeline, models=models)

config = load_config()
