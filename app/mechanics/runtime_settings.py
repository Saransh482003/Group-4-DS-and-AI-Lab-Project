import os
import sys

from mechanics.tts_config import DEFAULT_SHORTEN_TTS_COMMANDS

def get_base_dir():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_env_file(env_file_path):
    if not env_file_path or not os.path.exists(env_file_path):
        return

    with open(env_file_path, "r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _env_rel_path(base_dir, env_key, default_rel_path):
    return os.path.abspath(os.path.join(base_dir, os.getenv(env_key, default_rel_path)))


def _env_bool(env_key, default_bool):
    default_value = "1" if default_bool else "0"
    return os.getenv(env_key, default_value) == "1"


def _env_float(env_key, default_value):
    return float(os.getenv(env_key, str(default_value)))


def is_frozen():
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

def load_shared_runtime_settings(env_file_path=None):
    base_dir = get_base_dir()
    if not is_frozen():
        load_env_file(env_file_path)

    if is_frozen():
        yolo_path = os.path.join("model_weights", "YOLO11s-Final-Training.pt")
        depth_path = os.path.join("model_weights", "depth_anything_v2_metric_hypersim_vits.pth")
    else:
        yolo_path = os.path.join("..", "model_training", "object_detection", "best-weights", "YOLO11s-Final-Training.pt")
        depth_path = os.path.join("..", "model_training", "depth_estimation", "model_weights", "depth_anything_v2_metric_hypersim_vits.pth")

    return {
        "YOLO_WEIGHTS": _env_rel_path(
            base_dir,
            "YOLO_WEIGHTS_REL" if not is_frozen() else "IGNORE_YOLO",
            yolo_path,
        ),
        "DEPTH_MODEL_FILE": _env_rel_path(
            base_dir,
            "DEPTH_MODEL_DIR_REL" if not is_frozen() else "IGNORE_DEPTH",
            depth_path,
        ),
        "PIPER_EXE": _env_rel_path(
            base_dir, 
            "PIPER_EXE_REL" if not is_frozen() else "IGNORE_PIPER", 
            os.path.join("piper", "piper.exe")
        ),
        "PIPER_VOICE_MODEL": _env_rel_path(
            base_dir,
            "PIPER_VOICE_MODEL_REL" if not is_frozen() else "IGNORE_PIPER_VOICE",
            os.path.join("piper_voices", "en_US-amy-medium.onnx"),
        ),
        "PIPER_VOICE_CONFIG": _env_rel_path(
            base_dir,
            "PIPER_VOICE_CONFIG_REL" if not is_frozen() else "IGNORE_PIPER_CONFIG",
            os.path.join("piper_voices", "en_US-amy-medium.onnx.json"),
        ),
        "DEPTH_DANGER_THRESHOLD_M": _env_float("DEPTH_DANGER_THRESHOLD_M", 1.2),
        "DEPTH_WARNING_THRESHOLD_M": _env_float("DEPTH_WARNING_THRESHOLD_M", 2.0),
        "DEPTH_HAZARD_DANGER_WEIGHT": _env_float("DEPTH_HAZARD_DANGER_WEIGHT", 25.0),
        "DEPTH_HAZARD_WARNING_WEIGHT": _env_float("DEPTH_HAZARD_WARNING_WEIGHT", 8.0),
        "SHORTEN_TTS_COMMANDS": _env_bool("SHORTEN_TTS_COMMANDS", DEFAULT_SHORTEN_TTS_COMMANDS),
    }
