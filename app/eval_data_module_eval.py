import csv
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from mechanics.depth_estimation import DepthEstimator
from mechanics.frame_parser import SharedFrameParser
from mechanics.nav_tts_piper import PiperTTS
from mechanics.navigation_logic import NavigationLogic
from mechanics.object_detection import ObjectDetector
from mechanics.object_detection import draw_centered_label
from mechanics.runtime_settings import load_shared_runtime_settings
from mechanics.tts_config import DEFAULT_TTS_PHRASE_CACHE_MAXSIZE
from mechanics.tts_config import DEFAULT_TTS_USE_IN_MEMORY
from mechanics.tts_phrase_cache import TtsPhraseCache


# ----------------------
# Hardcoded configuration
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
SHARED_SETTINGS = load_shared_runtime_settings(BASE_DIR, env_file_path=ENV_FILE)

IMAGES_DIR = os.path.join(PROJECT_ROOT, "pipeline_evaluations", "eval_set", "images")
LABELS_DIR = os.path.join(PROJECT_ROOT, "pipeline_evaluations", "eval_set", "labels")
DEPTH_GT_DIR = os.path.join(PROJECT_ROOT, "pipeline_evaluations", "eval_set", "depth")
CLASSES_JSON = os.path.join(PROJECT_ROOT, "datasets", "classes.json")

YOLO_WEIGHTS = SHARED_SETTINGS["YOLO_WEIGHTS"]
DEPTH_MODEL_FILE = SHARED_SETTINGS["DEPTH_MODEL_FILE"]

PIPER_EXE = SHARED_SETTINGS["PIPER_EXE"]
PIPER_VOICE_MODEL = SHARED_SETTINGS["PIPER_VOICE_MODEL"]
PIPER_VOICE_CONFIG = SHARED_SETTINGS["PIPER_VOICE_CONFIG"]

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
MAX_IMAGES = None  # set an integer like 10 for quick runs
EVAL_STATEFUL_NAV = os.getenv("EVAL_STATEFUL_NAV", "0") == "1"

DANGER_THRESHOLD_M = SHARED_SETTINGS["DEPTH_DANGER_THRESHOLD_M"]
WARNING_THRESHOLD_M = SHARED_SETTINGS["DEPTH_WARNING_THRESHOLD_M"]
DEPTH_HAZARD_DANGER_WEIGHT = SHARED_SETTINGS["DEPTH_HAZARD_DANGER_WEIGHT"]
DEPTH_HAZARD_WARNING_WEIGHT = SHARED_SETTINGS["DEPTH_HAZARD_WARNING_WEIGHT"]

# TTS optimization switches for eval runs.
SHORTEN_TTS_COMMANDS = SHARED_SETTINGS["SHORTEN_TTS_COMMANDS"]
# Quality-first default: file-based Piper synthesis is typically the cleanest.
TTS_USE_IN_MEMORY = DEFAULT_TTS_USE_IN_MEMORY
# Eval latency should reflect synthesis only; skip WAV artifact writes.
TTS_SAVE_AUDIO_ARTIFACTS = False
# Eval latency should reflect actual synthesis work each time.
TTS_ENABLE_PHRASE_CACHE = True
TTS_PHRASE_CACHE_MAXSIZE = DEFAULT_TTS_PHRASE_CACHE_MAXSIZE


def ensure_paths():
    required = [
        IMAGES_DIR,
        LABELS_DIR,
        CLASSES_JSON,
        YOLO_WEIGHTS,
        DEPTH_MODEL_FILE,
        PIPER_EXE,
        PIPER_VOICE_MODEL,
        PIPER_VOICE_CONFIG,
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid classes json: {path}")
    return [str(x) for x in data]


def yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h):
    x1 = int((cx - bw / 2.0) * img_w)
    y1 = int((cy - bh / 2.0) * img_h)
    x2 = int((cx + bw / 2.0) * img_w)
    y2 = int((cy + bh / 2.0) * img_h)

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w, x2))
    y2 = max(0, min(img_h, y2))
    return [x1, y1, x2, y2]


def parse_label_file(label_path, img_w, img_h, class_names):
    if not os.path.isfile(label_path):
        return []

    gts = []
    with open(label_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                class_id = int(float(parts[0]))
                cx, cy, bw, bh = [float(v) for v in parts[1:5]]
            except ValueError:
                continue

            bbox = yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h)
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue

            class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
            gts.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "bbox": bbox,
                }
            )
    return gts


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def match_predictions(preds, gts, iou_threshold=0.5):
    candidates = []
    for p_idx, pred in enumerate(preds):
        for g_idx, gt in enumerate(gts):
            if pred["class_id"] != gt["class_id"]:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, p_idx, g_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_p = set()
    used_g = set()
    matches = []

    for iou, p_idx, g_idx in candidates:
        if p_idx in used_p or g_idx in used_g:
            continue
        used_p.add(p_idx)
        used_g.add(g_idx)
        matches.append((p_idx, g_idx, iou))

    unmatched_p = [i for i in range(len(preds)) if i not in used_p]
    unmatched_g = [i for i in range(len(gts)) if i not in used_g]
    return matches, unmatched_p, unmatched_g


def safe_mean(values):
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    return float(np.mean(clean))


def safe_percentile(values, p):
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    return float(np.percentile(clean, p))


def safe_div(a, b):
    if b == 0:
        return None
    return a / b


def load_depth_gt(depth_path):
    if not os.path.isfile(depth_path):
        return None, "not_found"

    gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        return None, "unreadable"

    if gt.ndim == 3:
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

    gt = gt.astype(np.float32)
    valid = np.isfinite(gt) & (gt > 0)
    if not np.any(valid):
        return None, "invalid"

    # For SUNRGBD-style depth maps, values are often millimeters.
    max_val = float(np.max(gt[valid]))
    if max_val > 255:
        gt_m = gt / 1000.0
        mode = "mm_to_m"
    else:
        gt_m = gt
        mode = "as_is"

    return gt_m, mode


def compute_depth_metrics(pred_depth_m, gt_depth_m):
    if gt_depth_m is None:
        return {
            "depth_abs_rel": None,
            "depth_rmse_m": None,
            "depth_delta1": None,
            "depth_valid_ratio": None,
        }

    if gt_depth_m.shape[:2] != pred_depth_m.shape[:2]:
        gt_depth_m = cv2.resize(gt_depth_m, (pred_depth_m.shape[1], pred_depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)

    valid = np.isfinite(pred_depth_m) & np.isfinite(gt_depth_m) & (pred_depth_m > 0) & (gt_depth_m > 0)
    valid_ratio = float(np.mean(valid))
    if np.sum(valid) < 20:
        return {
            "depth_abs_rel": None,
            "depth_rmse_m": None,
            "depth_delta1": None,
            "depth_valid_ratio": valid_ratio,
        }

    pred = pred_depth_m[valid].astype(np.float64)
    gt = gt_depth_m[valid].astype(np.float64)

    # Scale-align prediction to GT by median ratio for fair monocular depth comparison.
    scale = np.median(gt) / max(np.median(pred), 1e-8)
    pred = pred * scale

    abs_rel = float(np.mean(np.abs(pred - gt) / (gt + 1e-8)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    ratio = np.maximum(pred / (gt + 1e-8), gt / (pred + 1e-8))
    delta1 = float(np.mean(ratio < 1.25))

    return {
        "depth_abs_rel": abs_rel,
        "depth_rmse_m": rmse,
        "depth_delta1": delta1,
        "depth_valid_ratio": valid_ratio,
    }


def wrap_text(text, max_width, font, font_scale, thickness):
    words = text.split()
    if not words:
        return [""]

    lines = []
    line = ""

    for word in words:
        candidate = word if not line else f"{line} {word}"
        width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if width <= max_width:
            line = candidate
            continue

        if line:
            lines.append(line)
            line = ""

        # If one word itself is too long, split by characters.
        part = ""
        for ch in word:
            c2 = part + ch
            c2_w = cv2.getTextSize(c2, font, font_scale, thickness)[0][0]
            if c2_w <= max_width or not part:
                part = c2
            else:
                lines.append(part)
                part = ch
        if part:
            line = part

    if line:
        lines.append(line)

    return lines


def draw_wrapped_command(frame, command_text):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 2
    max_width = max(80, w - 20)

    lines = wrap_text(command_text, max_width, font, font_scale, thickness)

    line_h = cv2.getTextSize("Ay", font, font_scale, thickness)[0][1]
    padding = 8
    spacing = 6

    block_h = len(lines) * line_h + (len(lines) - 1) * spacing + 2 * padding
    y2 = h - 8
    y1 = max(0, y2 - block_h)
    x1 = 8
    x2 = w - 8

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    y = y1 + padding + line_h
    for line in lines:
        cv2.putText(frame, line, (x1 + padding, y), font, font_scale, (255, 255, 255), thickness)
        y += line_h + spacing


def draw_nav_zones(frame, nav_logic, zone_risks):
    h, w = frame.shape[:2]
    left_end = int(getattr(nav_logic, "left_end", int(0.30 * w)))
    center_end = int(getattr(nav_logic, "center_end", int(0.70 * w)))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (left_end, h), (0, 0, 255), -1)
    cv2.rectangle(overlay, (left_end, 0), (center_end, h), (0, 255, 0), -1)
    cv2.rectangle(overlay, (center_end, 0), (w, h), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    cv2.rectangle(frame, (0, 0), (left_end, h), (0, 0, 255), 2)
    cv2.rectangle(frame, (left_end, 0), (center_end, h), (0, 255, 0), 2)
    cv2.rectangle(frame, (center_end, 0), (w, h), (0, 0, 255), 2)

    cv2.putText(frame, f"Left Risk: {zone_risks['left']:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Center Risk: {zone_risks['center']:.2f}", (max(10, left_end + 10), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Right Risk: {zone_risks['right']:.2f}", (max(10, center_end + 10), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_depth_hazard_overlay(frame, hazard_result, alpha=0.30):
    if not hazard_result:
        return

    danger_mask = hazard_result.get("danger_mask")
    warning_mask = hazard_result.get("warning_mask")
    if danger_mask is None and warning_mask is None:
        return

    overlay = frame.copy()
    if warning_mask is not None:
        overlay[warning_mask] = (0, 215, 255)
    if danger_mask is not None:
        overlay[danger_mask] = (0, 0, 255)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def save_plots(rows, summary, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    idx = np.arange(1, len(rows) + 1)

    # 1) Per-image latency trend
    plt.figure(figsize=(14, 7))
    plt.plot(idx, [r["yolo_latency_ms"] for r in rows], label="YOLO", linewidth=1.8)
    plt.plot(idx, [r["depth_latency_ms"] for r in rows], label="Depth", linewidth=1.8)
    plt.plot(idx, [r["hazard_scan_latency_ms"] for r in rows], label="DepthHazard", linewidth=1.8)
    plt.plot(idx, [r["navigation_latency_ms"] for r in rows], label="Navigation", linewidth=1.8)
    plt.plot(idx, [r["tts_latency_ms"] for r in rows], label="TTS", linewidth=1.8)
    plt.plot(idx, [r["pipeline_latency_ms"] for r in rows], label="Pipeline", linewidth=2.0)
    plt.xlabel("Image Index")
    plt.ylabel("Latency (ms)")
    plt.title("Per-image Module Latencies")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "latency_per_image.png"), dpi=180)
    plt.close()

    # 2) Module latency mean vs p95
    modules = ["yolo", "depth", "hazard_scan", "navigation", "tts", "pipeline"]
    means = [summary["latency_ms"][m]["mean"] for m in modules]
    p95 = [summary["latency_ms"][m]["p95"] for m in modules]

    x = np.arange(len(modules))
    width = 0.38
    plt.figure(figsize=(11, 6))
    plt.bar(x - width / 2, means, width=width, label="Mean")
    plt.bar(x + width / 2, p95, width=width, label="P95")
    plt.xticks(x, [m.upper() for m in modules])
    plt.ylabel("Latency (ms)")
    plt.title("Module-wise Latency Summary")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "module_latency_summary.png"), dpi=180)
    plt.close()

    # 3) YOLO quality metrics
    yolo_keys = ["precision", "recall", "f1", "mean_iou"]
    yolo_vals = [summary["yolo_detection"][k] if summary["yolo_detection"][k] is not None else 0.0 for k in yolo_keys]

    plt.figure(figsize=(8, 5))
    bars = plt.bar([k.upper() for k in yolo_keys], yolo_vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    plt.ylim(0, 1)
    plt.title("YOLO Detection Metrics")
    for b, v in zip(bars, yolo_vals):
        plt.text(b.get_x() + b.get_width() / 2.0, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "yolo_metrics.png"), dpi=180)
    plt.close()

    # 4) Navigation command distribution
    commands = summary["navigation"]["command_distribution"]
    if commands:
        labels = list(commands.keys())
        values = list(commands.values())
        order = np.argsort(values)[::-1]
        labels = [labels[i] for i in order]
        values = [values[i] for i in order]

        plt.figure(figsize=(14, 6))
        plt.bar(np.arange(len(labels)), values, color="#2E8B57")
        plt.xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
        plt.ylabel("Count")
        plt.title("Navigation Command Distribution")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "navigation_command_distribution.png"), dpi=180)
        plt.close()

    # 5) Warning/danger pixel trend
    plt.figure(figsize=(14, 6))
    plt.plot(idx, [r["danger_pixel_count"] for r in rows], label="Danger Pixels", linewidth=2.0)
    plt.plot(idx, [r["warning_pixel_count"] for r in rows], label="Warning Pixels", linewidth=2.0)
    plt.plot(idx, [r["near_pixel_count"] for r in rows], label="Near Pixels", linewidth=2.0)
    plt.xlabel("Image Index")
    plt.ylabel("Pixel Count")
    plt.title("Depth Hazard Pixel Counts")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "depth_hazard_pixels_per_image.png"), dpi=180)
    plt.close()


def main():
    ensure_paths()

    if WARNING_THRESHOLD_M < DANGER_THRESHOLD_M:
        raise ValueError("WARNING_THRESHOLD_M must be >= DANGER_THRESHOLD_M")

    class_names = load_class_names(CLASSES_JSON)
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_files.sort()
    if MAX_IMAGES is not None:
        image_files = image_files[:MAX_IMAGES]

    if not image_files:
        raise RuntimeError("No eval images found.")

    run_name = f"module_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(BASE_DIR, "navigation_eval_outputs", run_name)
    annotated_dir = os.path.join(run_dir, "annotated_images")
    hazard_coords_dir = os.path.join(run_dir, "hazard_coords")
    audio_dir = os.path.join(run_dir, "tts_audio")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(annotated_dir, exist_ok=True)
    os.makedirs(hazard_coords_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Total eval images: {len(image_files)}")
    print(f"Depth hazard thresholds (m): danger<={DANGER_THRESHOLD_M}, warning<={WARNING_THRESHOLD_M}")
    print(f"Eval navigation stateful: {EVAL_STATEFUL_NAV}")

    # Load modules once
    model_load_start = time.perf_counter()
    detector = ObjectDetector(YOLO_WEIGHTS)
    detector.load_model()

    depth_estimator = DepthEstimator(DEPTH_MODEL_FILE, device=device)
    depth_estimator.load_model()

    tts = PiperTTS(
        piper_executable=PIPER_EXE,
        voice_model_path=PIPER_VOICE_MODEL,
        voice_config_path=PIPER_VOICE_CONFIG,
        output_dir=audio_dir,
    )
    frame_parser = SharedFrameParser(
        object_detector=detector,
        depth_estimator=depth_estimator,
        nav_logic_factory=lambda frame_width: NavigationLogic(
            frame_width=frame_width,
            depth_hazard_danger_weight=DEPTH_HAZARD_DANGER_WEIGHT,
            depth_hazard_warning_weight=DEPTH_HAZARD_WARNING_WEIGHT,
        ),
        device=device,
        depth_hazard_enabled=True,
        danger_threshold_m=DANGER_THRESHOLD_M,
        warning_threshold_m=WARNING_THRESHOLD_M,
        stateful_navigation=EVAL_STATEFUL_NAV,
    )
    
    # Prewarm the TTS phrase cache to drop latency to <1ms for common commands.
    tts_phrase_cache = TtsPhraseCache(TTS_PHRASE_CACHE_MAXSIZE) if TTS_ENABLE_PHRASE_CACHE else None
    tts_prewarm_ms = 0.0
    if tts_phrase_cache is not None:
        print("Prewarming TTS phrase cache...")
        tts_prewarm_start = time.perf_counter()
        phrases_to_prewarm = [
            "Go straight.", "Turn left.", "Turn right.", 
            "Move slightly left.", "Move slightly right.",
            "Path blocked. Scan around.", "Searching for path. Turn back."
        ]
        for phrase in phrases_to_prewarm:
            tts_phrase_cache.put(phrase, tts.synthesize_wav_bytes(phrase))
        tts_prewarm_ms = (time.perf_counter() - tts_prewarm_start) * 1000.0
        print(f"TTS phrase prewarming completed in {tts_prewarm_ms:.1f}ms for {len(phrases_to_prewarm)} phrases")

    model_load_ms = (time.perf_counter() - model_load_start) * 1000.0

    rows = []
    command_counter = Counter()
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "matches": 0})
    depth_scale_counter = Counter()


    for i, image_file in enumerate(image_files, start=1):
        stem = os.path.splitext(image_file)[0]
        image_path = os.path.join(IMAGES_DIR, image_file)
        label_path = os.path.join(LABELS_DIR, f"{stem}.txt")
        depth_gt_path = os.path.join(DEPTH_GT_DIR, f"{stem}.png")

        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            print(f"[{i}/{len(image_files)}] Skipping unreadable image: {image_file}")
            continue

        h, w = frame_bgr.shape[:2]
        gt_boxes = parse_label_file(label_path, w, h, class_names)

        pipeline_start = time.perf_counter()

        parsed = frame_parser.parse_frame(
            frame_bgr,
            shorten_tts_commands=SHORTEN_TTS_COMMANDS,
            sync_after_yolo=(device == "cuda"),
            sync_after_depth=(device == "cuda"),
            confidence_threshold=CONF_THRESHOLD,
            hazard_return_coords=True,
        )
        latencies = parsed["latencies"]
        yolo_latency_ms = latencies.get("yolo_latency_ms", 0.0)
        depth_latency_ms = latencies.get("depth_latency_ms", 0.0)
        hazard_scan_latency_ms = latencies.get("hazard_scan_latency_ms", 0.0)
        navigation_latency_ms = latencies.get("navigation_latency_ms", 0.0)

        depth_pred_m = parsed["depth_float"]
        hazard_result = parsed["hazard_result"] or {}
        danger_coords_xy = hazard_result.get("danger_coords_xy")
        warning_coords_xy = hazard_result.get("warning_coords_xy")
        if danger_coords_xy is None:
            danger_coords_xy = np.empty((0, 2), dtype=np.int32)
        if warning_coords_xy is None:
            warning_coords_xy = np.empty((0, 2), dtype=np.int32)
        hazard_global = hazard_result.get("global_summary", {})
        hazard_coords_path = os.path.join(hazard_coords_dir, f"{stem}_hazard_coords.npz")

        pred_boxes = [
            {
                "class_id": int(d["class_id"]),
                "class_name": str(d["class_name"]),
                "confidence": float(d["confidence"]),
                "bbox": [int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])],
                "depth": d.get("depth"),
                "distance": d.get("distance"),
            }
            for d in parsed["frame_boxes"]
        ]

        nav_logic = parsed["navigation_logic"]
        zone_risks = parsed["zone_risks"]
        nav_command = parsed["nav_command"]
        tts_text = parsed["tts_text"]
        command_counter[nav_command] += 1

        # TTS latency: synthesis only (no WAV file write time).
        tts_file = ""
        tts_latency_ms = 0.0
        tts_error = ""
        tts_generated = 0
        tts_audio_saved = 0
        tts_cache_hit = 0
        tts_mode = "in_memory" if TTS_USE_IN_MEMORY else "file"
        tts_start_time = time.perf_counter()
        try:
            if TTS_USE_IN_MEMORY:
                wav_bytes = None
                if tts_phrase_cache is not None:
                    wav_bytes = tts_phrase_cache.get(tts_text)
                    if wav_bytes is not None:
                        tts_cache_hit = 1
                        tts_mode = "in_memory_cache"
                if wav_bytes is None:
                    wav_bytes = tts.synthesize_wav_bytes(tts_text)
                    if tts_phrase_cache is not None:
                        tts_phrase_cache.put(tts_text, wav_bytes)
                tts_generated = 1
            else:
                tts_mode = "synthesis_only"
                _ = tts.synthesize_wav_bytes(tts_text)
                tts_generated = 1
            tts_latency_ms = (time.perf_counter() - tts_start_time) * 1000.0
        except Exception as exc:
            tts_error = str(exc)
            tts_file = ""

        pipeline_latency_ms = (time.perf_counter() - pipeline_start) * 1000.0

        # Save large coordinate arrays to disk manually after we record final pipeline time
        np.savez_compressed(
            hazard_coords_path,
            danger_coords_xy=danger_coords_xy,
            warning_coords_xy=warning_coords_xy,
        )

        # YOLO detection quality for this image
        matches, unmatched_pred, unmatched_gt = match_predictions(pred_boxes, gt_boxes, IOU_THRESHOLD)
        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)

        mean_iou = safe_mean([m[2] for m in matches])

        for p_idx, g_idx, iou in matches:
            cls = pred_boxes[p_idx]["class_name"]
            class_stats[cls]["tp"] += 1
            class_stats[cls]["iou_sum"] += float(iou)
            class_stats[cls]["matches"] += 1
        for p_idx in unmatched_pred:
            cls = pred_boxes[p_idx]["class_name"]
            class_stats[cls]["fp"] += 1
        for g_idx in unmatched_gt:
            cls = gt_boxes[g_idx]["class_name"]
            class_stats[cls]["fn"] += 1

        # Depth quality for this image
        depth_gt_m, depth_mode = load_depth_gt(depth_gt_path)
        depth_scale_counter[depth_mode] += 1
        d_metrics = compute_depth_metrics(depth_pred_m, depth_gt_m)

        # Visualization output
        vis = frame_bgr.copy()
        draw_depth_hazard_overlay(vis, hazard_result, alpha=0.30)
        draw_nav_zones(vis, nav_logic, zone_risks)

        cv2.putText(
            vis,
            f"Hazard px (D/W): {hazard_global['danger_pixel_count']}/{hazard_global['warning_pixel_count']}",
            (10, max(60, h - 90)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        for d in pred_boxes:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 220, 255), 2)
            cv2.putText(
                vis,
                f"{d['class_name']} {d['confidence']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (80, 220, 255),
                2,
            )
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            dist_label = "dist: N/A" if d.get("distance") is None else f"dist: {d['distance']:.2f}m"
            draw_centered_label(vis, dist_label, cx, cy)

        # Wrapped command text at bottom.
        draw_wrapped_command(vis, nav_command)

        # Quick latency strip at top-right, also wrapped.
        latency_text = (
            f"YOLO:{yolo_latency_ms:.1f}ms Depth:{depth_latency_ms:.1f}ms Hazard:{hazard_scan_latency_ms:.1f}ms "
            f"Nav:{navigation_latency_ms:.1f}ms TTS:{tts_latency_ms:.1f}ms "
            f"Pipe:{pipeline_latency_ms:.1f}ms"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = wrap_text(latency_text, max(120, int(w * 0.56)), font, 0.45, 1)
        y = 58
        x = max(8, int(w * 0.42))
        cv2.rectangle(vis, (x - 6, 10), (w - 8, 70), (0, 0, 0), -1)
        cv2.addWeighted(vis, 0.92, vis, 0.08, 0, vis)
        for line in lines:
            cv2.putText(vis, line, (x, y), font, 0.45, (255, 255, 255), 1)
            y += 18

        out_img = os.path.join(annotated_dir, f"{i:03d}_{stem}.jpg")
        cv2.imwrite(out_img, vis)

        row = {
            "sample_id": i,
            "image_stem": stem,
            "image_path": image_path,
            "label_path": label_path,
            "depth_gt_path": depth_gt_path,
            "annotated_image_path": out_img,
            "hazard_coords_path": hazard_coords_path,
            "danger_coords_count": int(danger_coords_xy.shape[0]),
            "warning_coords_count": int(warning_coords_xy.shape[0]),
            "tts_audio_path": tts_file,
            "tts_audio_generated": tts_generated,
            "tts_audio_saved": tts_audio_saved,
            "tts_mode": tts_mode,
            "tts_cache_hit": tts_cache_hit,
            "tts_error": tts_error,
            "pred_count": len(pred_boxes),
            "gt_count": len(gt_boxes),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
            "yolo_latency_ms": yolo_latency_ms,
            "depth_latency_ms": depth_latency_ms,
            "hazard_scan_latency_ms": hazard_scan_latency_ms,
            "navigation_latency_ms": navigation_latency_ms,
            "tts_latency_ms": tts_latency_ms,
            "pipeline_latency_ms": pipeline_latency_ms,
            "nav_command": nav_command,
            "tts_text": tts_text,
            "left_risk": zone_risks["left"],
            "center_risk": zone_risks["center"],
            "right_risk": zone_risks["right"],
            "danger_pixel_count": hazard_global["danger_pixel_count"],
            "warning_pixel_count": hazard_global["warning_pixel_count"],
            "near_pixel_count": hazard_global["near_pixel_count"],
            "danger_ratio": hazard_global["danger_ratio"],
            "warning_ratio": hazard_global["warning_ratio"],
            "near_ratio": hazard_global["near_ratio"],
            "hazard_min_depth_m": hazard_global["min_depth_m"],
            "hazard_median_depth_m": hazard_global["median_depth_m"],
            "danger_left_ratio": hazard_result["zone_summary"]["left"]["danger_ratio"],
            "danger_center_ratio": hazard_result["zone_summary"]["center"]["danger_ratio"],
            "danger_right_ratio": hazard_result["zone_summary"]["right"]["danger_ratio"],
            "warning_left_ratio": hazard_result["zone_summary"]["left"]["warning_ratio"],
            "warning_center_ratio": hazard_result["zone_summary"]["center"]["warning_ratio"],
            "warning_right_ratio": hazard_result["zone_summary"]["right"]["warning_ratio"],
            "depth_gt_mode": depth_mode,
            "depth_abs_rel": d_metrics["depth_abs_rel"],
            "depth_rmse_m": d_metrics["depth_rmse_m"],
            "depth_delta1": d_metrics["depth_delta1"],
            "depth_valid_ratio": d_metrics["depth_valid_ratio"],
        }
        rows.append(row)

        print(
            f"[{i}/{len(image_files)}] {stem} | "
            f"TP/FP/FN={tp}/{fp}/{fn} | "
            f"Latency(ms) YOLO:{yolo_latency_ms:.1f} Depth:{depth_latency_ms:.1f} "
            f"Haz:{hazard_scan_latency_ms:.1f} Nav:{navigation_latency_ms:.1f} "
            f"TTS:{tts_latency_ms:.1f} Pipe:{pipeline_latency_ms:.1f} | "
            f"Danger/Warn px:{hazard_global['danger_pixel_count']}/{hazard_global['warning_pixel_count']}"
        )

    if not rows:
        raise RuntimeError("No images processed.")

    # Save per-image metrics CSV
    csv_path = os.path.join(run_dir, "per_image_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate metrics
    total_tp = sum(r["tp"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    total_fn = sum(r["fn"] for r in rows)

    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)

    summary = {
        "run_info": {
            "run_name": run_name,
            "run_dir": run_dir,
            "device": device,
            "model_load_ms": model_load_ms,
            "tts_prewarm_ms": tts_prewarm_ms,
            "images_processed": len(rows),
            "eval_stateful_nav": EVAL_STATEFUL_NAV,
            "yolo_conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "danger_threshold_m": DANGER_THRESHOLD_M,
            "warning_threshold_m": WARNING_THRESHOLD_M,
            "depth_hazard_danger_weight": DEPTH_HAZARD_DANGER_WEIGHT,
            "depth_hazard_warning_weight": DEPTH_HAZARD_WARNING_WEIGHT,
            "shorten_tts_commands": SHORTEN_TTS_COMMANDS,
            "tts_use_in_memory": TTS_USE_IN_MEMORY,
            "tts_save_audio_artifacts": TTS_SAVE_AUDIO_ARTIFACTS,
            "tts_enable_phrase_cache": TTS_ENABLE_PHRASE_CACHE,
            "tts_phrase_cache_maxsize": TTS_PHRASE_CACHE_MAXSIZE,
        },
        "yolo_detection": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": safe_mean([r["mean_iou"] for r in rows]),
        },
        "depth_estimation": {
            "depth_gt_mode_counts": dict(depth_scale_counter),
            "abs_rel_mean": safe_mean([r["depth_abs_rel"] for r in rows]),
            "rmse_m_mean": safe_mean([r["depth_rmse_m"] for r in rows]),
            "delta1_mean": safe_mean([r["depth_delta1"] for r in rows]),
            "valid_ratio_mean": safe_mean([r["depth_valid_ratio"] for r in rows]),
        },
        "depth_hazard_detection": {
            "danger_threshold_m": DANGER_THRESHOLD_M,
            "warning_threshold_m": WARNING_THRESHOLD_M,
            "danger_pixel_mean": safe_mean([r["danger_pixel_count"] for r in rows]),
            "warning_pixel_mean": safe_mean([r["warning_pixel_count"] for r in rows]),
            "near_pixel_mean": safe_mean([r["near_pixel_count"] for r in rows]),
            "images_with_danger": int(sum(1 for r in rows if r["danger_pixel_count"] > 0)),
            "images_with_warning": int(sum(1 for r in rows if r["warning_pixel_count"] > 0)),
            "danger_ratio_mean": safe_mean([r["danger_ratio"] for r in rows]),
            "warning_ratio_mean": safe_mean([r["warning_ratio"] for r in rows]),
            "near_ratio_mean": safe_mean([r["near_ratio"] for r in rows]),
            "hazard_min_depth_m_mean": safe_mean([r["hazard_min_depth_m"] for r in rows]),
        },
        "latency_ms": {
            "yolo": {
                "mean": safe_mean([r["yolo_latency_ms"] for r in rows]),
                "p50": safe_percentile([r["yolo_latency_ms"] for r in rows], 50),
                "p95": safe_percentile([r["yolo_latency_ms"] for r in rows], 95),
            },
            "depth": {
                "mean": safe_mean([r["depth_latency_ms"] for r in rows]),
                "p50": safe_percentile([r["depth_latency_ms"] for r in rows], 50),
                "p95": safe_percentile([r["depth_latency_ms"] for r in rows], 95),
            },
            "hazard_scan": {
                "mean": safe_mean([r["hazard_scan_latency_ms"] for r in rows]),
                "p50": safe_percentile([r["hazard_scan_latency_ms"] for r in rows], 50),
                "p95": safe_percentile([r["hazard_scan_latency_ms"] for r in rows], 95),
            },
            "navigation": {
                "mean": safe_mean([r["navigation_latency_ms"] for r in rows]),
                "p50": safe_percentile([r["navigation_latency_ms"] for r in rows], 50),
                "p95": safe_percentile([r["navigation_latency_ms"] for r in rows], 95),
            },
            "tts": {
                "mean": safe_mean([r["tts_latency_ms"] for r in rows]),
                "p50": safe_percentile([r["tts_latency_ms"] for r in rows], 50),
                "p95": safe_percentile([r["tts_latency_ms"] for r in rows], 95),
            },
            "pipeline": {
                "mean": safe_mean([r["pipeline_latency_ms"] for r in rows]),
                "p50": safe_percentile([r["pipeline_latency_ms"] for r in rows], 50),
                "p95": safe_percentile([r["pipeline_latency_ms"] for r in rows], 95),
            },
        },
        "navigation": {
            "command_distribution": dict(command_counter),
            "unique_commands": len(command_counter),
        },
        "tts": {
            "audio_generated_count": int(sum(r["tts_audio_generated"] for r in rows)),
            "audio_saved_count": int(sum(r["tts_audio_saved"] for r in rows)),
            "audio_failed_count": int(sum(1 for r in rows if not r["tts_audio_generated"])),
        },
        "classwise_yolo": {},
    }

    for cls, stats in class_stats.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        f = None if p is None or r is None or (p + r) == 0 else 2 * p * r / (p + r)
        miou = safe_div(stats["iou_sum"], max(stats["matches"], 1))
        summary["classwise_yolo"][cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f1": f,
            "mean_iou": miou,
        }

    summary_json = os.path.join(run_dir, "summary_metrics.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Human-readable analysis
    report_path = os.path.join(run_dir, "analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Module-wise Evaluation Report\n\n")
        f.write("## YOLO Object Detection\n")
        f.write(f"- Precision: {summary['yolo_detection']['precision']}\n")
        f.write(f"- Recall: {summary['yolo_detection']['recall']}\n")
        f.write(f"- F1: {summary['yolo_detection']['f1']}\n")
        f.write(f"- Mean IoU: {summary['yolo_detection']['mean_iou']}\n\n")

        f.write("## Depth Estimation\n")
        f.write(f"- AbsRel mean: {summary['depth_estimation']['abs_rel_mean']}\n")
        f.write(f"- RMSE (m) mean: {summary['depth_estimation']['rmse_m_mean']}\n")
        f.write(f"- Delta1 mean: {summary['depth_estimation']['delta1_mean']}\n")
        f.write(f"- Valid ratio mean: {summary['depth_estimation']['valid_ratio_mean']}\n\n")

        f.write("## Depth Hazard Detection\n")
        f.write(f"- Danger threshold (m): {summary['depth_hazard_detection']['danger_threshold_m']}\n")
        f.write(f"- Warning threshold (m): {summary['depth_hazard_detection']['warning_threshold_m']}\n")
        f.write(f"- Mean danger pixels: {summary['depth_hazard_detection']['danger_pixel_mean']}\n")
        f.write(f"- Mean warning pixels: {summary['depth_hazard_detection']['warning_pixel_mean']}\n")
        f.write(f"- Images with danger pixels: {summary['depth_hazard_detection']['images_with_danger']}\n")
        f.write(f"- Mean nearest hazard depth (m): {summary['depth_hazard_detection']['hazard_min_depth_m_mean']}\n\n")

        f.write("## Latency (ms)\n")
        for module_name, vals in summary["latency_ms"].items():
            f.write(
                f"- {module_name}: mean={vals['mean']}, p50={vals['p50']}, p95={vals['p95']}\n"
            )
        f.write("\n")

        f.write("## Navigation + TTS\n")
        f.write(f"- Command distribution: {summary['navigation']['command_distribution']}\n")
        f.write(f"- TTS generated count: {summary['tts']['audio_generated_count']}\n")
        f.write(f"- TTS failed count: {summary['tts']['audio_failed_count']}\n")

    save_plots(rows, summary, plots_dir)

    print("\nEvaluation finished.")
    print(f"Run dir: {run_dir}")
    print(f"Per-image CSV: {csv_path}")
    print(f"Summary JSON: {summary_json}")
    print(f"Report: {report_path}")
    print(f"Annotated images: {annotated_dir}")
    print(f"Hazard coords (npz): {hazard_coords_dir}")
    print(f"TTS audios: {audio_dir}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
