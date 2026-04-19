import os
import cv2
import gradio as gr
import numpy as np
import time

import spaces
import torch
import tempfile
import sys
import shutil

from mechanics.depth_estimation import DepthEstimator
from mechanics.frame_parser import SharedFrameParser
from mechanics.navigation_logic import NavigationLogic
from mechanics.object_detection import ObjectDetector, draw_centered_label
from mechanics.runtime_settings import load_shared_runtime_settings
from mechanics.nav_tts_piper import PiperTTS
from mechanics.tts_phrase_cache import TtsPhraseCache
from mechanics.tts_config import DEFAULT_TTS_PHRASE_CACHE_MAXSIZE

# ========================================================
# 1. INITIALIZATION & SETUP
# ========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", ".env"))
SHARED_SETTINGS = load_shared_runtime_settings(env_file_path=ENV_FILE)

# System-level Piper fallback for Hugging Face Spaces (Linux Container) vs Windows.
PIPER_EXE = "piper" if sys.platform != "win32" else SHARED_SETTINGS.get("PIPER_EXE", "piper")

# Initialize Models Once Globally
print("Loading Models into Memory...")
object_detector = ObjectDetector(SHARED_SETTINGS["YOLO_WEIGHTS"])
object_detector.load_model()

# Notice: DepthEstimator initialized on CPU here; ZeroGPU handles moving to CUDA dynamically.
depth_estimator = DepthEstimator(SHARED_SETTINGS["DEPTH_MODEL_FILE"], device="cpu")
depth_estimator.load_model()

def fallback_nav_logic_factory(frame_width):
    return NavigationLogic(
        frame_width=frame_width,
        depth_hazard_danger_weight=SHARED_SETTINGS.get("DEPTH_HAZARD_DANGER_WEIGHT", 25.0),
        depth_hazard_warning_weight=SHARED_SETTINGS.get("DEPTH_HAZARD_WARNING_WEIGHT", 20.0),
    )

frame_parser = SharedFrameParser(
    object_detector=object_detector,
    depth_estimator=depth_estimator,
    nav_logic_factory=fallback_nav_logic_factory,
    device="cpu", # Handled dynamically below
    depth_hazard_enabled=True,
    danger_threshold_m=SHARED_SETTINGS.get("DEPTH_DANGER_THRESHOLD_M", 1.8),
    warning_threshold_m=SHARED_SETTINGS.get("DEPTH_WARNING_THRESHOLD_M", 2.5),
    stateful_navigation=False
)

print("Initializing TTS Engine and Cache...")
tts_engine = PiperTTS(
    piper_executable=PIPER_EXE,
    voice_model_path=SHARED_SETTINGS["PIPER_VOICE_MODEL"],
    voice_config_path=SHARED_SETTINGS["PIPER_VOICE_CONFIG"],
)

tts_cache = TtsPhraseCache(DEFAULT_TTS_PHRASE_CACHE_MAXSIZE)
phrases_to_prewarm = [
    "Go straight.", "Turn left.", "Turn right.", 
    "Move slightly left.", "Move slightly right.",
    "Path blocked. Scan around.", "Searching for path. Turn back."
]
for phrase in phrases_to_prewarm:
    try:
        tts_cache.put(phrase, tts_engine.synthesize_wav_bytes(phrase))
    except Exception:
        pass


# ========================================================
# 2. VISUALIZATION FUNCTIONS 
# ========================================================

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
    if not nav_logic or not zone_risks:
        return
    h, w = frame.shape[:2]
    left_end = int(getattr(nav_logic, "left_end", int(0.30 * w)))
    center_end = int(getattr(nav_logic, "center_end", int(0.70 * w)))

    # overlay = frame.copy()
    # cv2.rectangle(overlay, (0, 0), (left_end, h), (0, 0, 255), -1)
    # cv2.rectangle(overlay, (left_end, 0), (center_end, h), (0, 255, 0), -1)
    # cv2.rectangle(overlay, (center_end, 0), (w, h), (0, 0, 255), -1)
    # cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    cv2.rectangle(frame, (0, 0), (left_end, h), (0, 0, 255), 2)
    cv2.rectangle(frame, (left_end, 0), (center_end, h), (0, 255, 0), 2)
    cv2.rectangle(frame, (center_end, 0), (w, h), (0, 0, 255), 2)

    cv2.putText(frame, f"Left Risk: {zone_risks.get('left', 0):.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Center Risk: {zone_risks.get('center', 0):.2f}", (max(10, left_end + 10), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Right Risk: {zone_risks.get('right', 0):.2f}", (max(10, center_end + 10), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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


# ========================================================
# 3. GRADIO PROCESSING LOGIC with ZEROGPU
# ========================================================
# This decorator automatically shifts the model execution to the free A100 GPU when a user clicks the button.
@spaces.GPU
def process_image(image_rgb):
    if image_rgb is None:
        return None, "Please upload an image.", None, ""
        
    pipeline_start = time.perf_counter()
    
    # Push models dynamically to GPU since we are now entering the ZeroGPU environment
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    frame_parser.device = device_str
    if hasattr(depth_estimator, "model") and hasattr(depth_estimator.model, "to"):
        depth_estimator.model.to(device_str)
        
    # Gradio passes numpy arrays as RGB.
    # Convert incoming RGB to Grayscale, then immediately back to 3-channel BGR 
    # so we can draw colored annotations on top of a gray background map.
    frame_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    results = frame_parser.parse_frame(
        frame_bgr, 
        shorten_tts_commands=SHARED_SETTINGS.get("SHORTEN_TTS_COMMANDS", True),
        sync_after_yolo=True,
        sync_after_depth=True,
        confidence_threshold=0.25,
        hazard_return_coords=True
    )
    
    latencies = results.get("latencies", {})
    yolo_latency_ms = latencies.get("yolo_latency_ms", 0.0)
    depth_latency_ms = latencies.get("depth_latency_ms", 0.0)
    hazard_scan_latency_ms = latencies.get("hazard_scan_latency_ms", 0.0)
    navigation_latency_ms = latencies.get("navigation_latency_ms", 0.0)
    
    tts_text = results["tts_text"]
    nav_command = results["nav_command"]
    frame_boxes = results["frame_boxes"]
    hazard_result = results.get("hazard_result", {})
    hazard_global = hazard_result.get("global_summary", {}) if hazard_result else {}
    nav_logic = results.get("navigation_logic")
    zone_risks = results.get("zone_risks", {})

    # 1. Audio Synthesis (using Cache)
    tts_start_time = time.perf_counter()
    wav_bytes = None
    temp_wav_path = None
    try:
        tts_engine._validate_paths()
        # Hit the memory cache first
        wav_bytes = tts_cache.get(tts_text)
        if wav_bytes is None:
            wav_bytes = tts_engine.synthesize_wav_bytes(tts_text)
            tts_cache.put(tts_text, wav_bytes)
            
        temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
        os.write(temp_wav_fd, wav_bytes)
        os.close(temp_wav_fd)
    except Exception as e:
        print(f"TTS synthesis error: {e}")
        
    tts_latency_ms = (time.perf_counter() - tts_start_time) * 1000.0
    pipeline_latency_ms = (time.perf_counter() - pipeline_start) * 1000.0

    # 2. Overlay Drawing Logic
    h, w = frame_bgr.shape[:2]
    
    draw_depth_hazard_overlay(frame_bgr, hazard_result, alpha=0.30)
    draw_nav_zones(frame_bgr, nav_logic, zone_risks)

    if hazard_global:
        cv2.putText(
            frame_bgr,
            f"Hazard px (D/W): {hazard_global.get('danger_pixel_count', 0)}/{hazard_global.get('warning_pixel_count', 0)}",
            (10, max(60, h - 90)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    for d in frame_boxes:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (80, 220, 255), 2)
        cv2.putText(
            frame_bgr,
            f"{d['class_name']} {d.get('confidence', 0.0):.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 255), 2
        )
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dist_label = "dist: N/A" if d.get("distance") is None else f"dist: {d['distance']:.2f}m"
        draw_centered_label(frame_bgr, dist_label, cx, cy)

    draw_wrapped_command(frame_bgr, nav_command)

    # Convert back to RGB for Gradio UI 
    annotated_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Populate the Dashboard texts
    dashboard_md = f"""
### Latency Breakdown
- **Total Pipeline:** {pipeline_latency_ms:.1f} ms
- **YOLO Detection:** {yolo_latency_ms:.1f} ms
- **Depth Estimation:** {depth_latency_ms:.1f} ms
- **Hazard Scanning:** {hazard_scan_latency_ms:.1f} ms
- **TTS Generation:** {tts_latency_ms:.1f} ms
"""
    return annotated_rgb, tts_text, temp_wav_path, dashboard_md

# ========================================================
# 4. GRADIO UI DEFINITION (Dashboard Layout)
# ========================================================
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

with gr.Blocks(title="DSAI Blind Navigation Tool", theme=theme) as demo:
    gr.Markdown("# 🤖 Wearable Blind Navigation Assistant")
    gr.Markdown("Upload an image of an indoor environment to see the YOLO object detections and Depth Anything hazard map in action. The tool calculates a safe navigation path and reads it aloud using Piper TTS.")
    
    with gr.Row():
        # Left Sidebar for Upload
        with gr.Column(scale=1):
            gr.Markdown("### Input Controls")
            input_image = gr.Image(type="numpy", label="Upload Subject Image")
            submit_btn = gr.Button("Analyze Environment", variant="primary")
            gr.Markdown("---")
            gr.Markdown("This tool uses **ZeroGPU** to process your path instantly.")

        # Right Area for Results Layout
        with gr.Column(scale=3):
            gr.Markdown("### Analyzed View")
            output_image = gr.Image(type="numpy", label="Environment View")
            
            gr.Markdown("### Results Dashboard")
            with gr.Row():
                with gr.Column(scale=2):
                    audio_output = gr.Audio(label="Spoken Navigation Warning", type="filepath")
                    text_output = gr.Textbox(label="TTS Transcript")
                with gr.Column(scale=1):
                    metrics_output = gr.Markdown("### Latency Breakdown\nWaiting for analysis...")

    submit_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_image, text_output, audio_output, metrics_output]
    )

if __name__ == "__main__":
    demo.launch()