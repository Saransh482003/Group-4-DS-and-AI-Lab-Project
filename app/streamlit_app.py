import os
import time
import cv2
import numpy as np
import tempfile
import sys
import torch
import streamlit as st

st.set_page_config(page_title="DSAI Blind Navigation Tool", page_icon="??", layout="wide")

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

@st.cache_resource
def load_settings():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ENV_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", ".env"))
    return load_shared_runtime_settings(env_file_path=ENV_FILE)

SHARED_SETTINGS = load_settings()

PIPER_EXE = "piper" if sys.platform != "win32" else SHARED_SETTINGS.get("PIPER_EXE", "piper")

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    object_detector = ObjectDetector(SHARED_SETTINGS["YOLO_WEIGHTS"])
    object_detector.load_model()

    depth_estimator = DepthEstimator(SHARED_SETTINGS["DEPTH_MODEL_FILE"], device=device)
    depth_estimator.load_model()
    
    def fallback_nav_logic_factory(frame_width):
        return NavigationLogic(
            frame_width=frame_width,
            depth_hazard_danger_weight=SHARED_SETTINGS.get("DEPTH_HAZARD_DANGER_WEIGHT", 3.0),
            depth_hazard_warning_weight=SHARED_SETTINGS.get("DEPTH_HAZARD_WARNING_WEIGHT", 1.0),
        )

    frame_parser = SharedFrameParser(
        object_detector=object_detector,
        depth_estimator=depth_estimator,
        nav_logic_factory=fallback_nav_logic_factory,
        device=device,
        depth_hazard_enabled=True,
        danger_threshold_m=SHARED_SETTINGS.get("DEPTH_DANGER_THRESHOLD_M", 1.2),
        warning_threshold_m=SHARED_SETTINGS.get("DEPTH_WARNING_THRESHOLD_M", 2.0),
        stateful_navigation=False
    )

    tts_engine = PiperTTS(
        piper_executable=PIPER_EXE,
        voice_model_path=SHARED_SETTINGS["PIPER_VOICE_MODEL"],
        voice_config_path=SHARED_SETTINGS["PIPER_VOICE_CONFIG"],
    )
    
    # --- WARMUP PROCESS ---
    # PyTorch models take significantly longer on their first forward pass due to GPU initialization and graph compilation.
    # We do a dummy pass here so the first real image uploaded by the user is fully fast!
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = frame_parser.parse_frame(dummy_frame, shorten_tts_commands=True, sync_after_yolo=True, sync_after_depth=True)
    
    # Prewarm the TTS phrase cache to drop latency to <1ms for common commands.
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
    
    return frame_parser, tts_engine, tts_cache

# Initialize models
try:
    frame_parser, tts_engine, tts_cache = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()


# ========================================================
# 2. VISUALIZATION FUNCTIONS (from eval_data_module_eval.py)
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

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (left_end, h), (0, 0, 255), -1)
    cv2.rectangle(overlay, (left_end, 0), (center_end, h), (0, 255, 0), -1)
    cv2.rectangle(overlay, (center_end, 0), (w, h), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

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
# 3. STREAMLIT UI & LOGIC
# ========================================================

st.title("🤖 Wearable Blind Navigation Assistant")

# Move file uploader and controls to the sidebar to save screen space
with st.sidebar:
    st.header("Input Controls")
    st.markdown("Upload an image of an indoor environment to see object detections and Depth Anything hazard maps.")
    uploaded_file = st.file_uploader("Upload Subject Image", type=["jpg", "jpeg", "png", "webp"])
    
    analyze_btn = False
    if uploaded_file is not None:
        analyze_btn = st.button("Analyze Environment", type="primary", use_container_width=True)
        
    st.divider()
    st.markdown("### About")
    st.caption("This tool calculates a safe navigation path using depth grids and YOLO detections, then reads it aloud using Piper TTS.")

if uploaded_file is not None:
    # Convert uploaded file to opencv image format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Use a two-column layout for immediate side-by-side comparison
    col_img, col_res = st.columns([1.2, 1])

    with col_img:
        if not analyze_btn:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.info("👈 Click **Analyze Environment** in the sidebar to start processing.")

    if analyze_btn:
        with col_img:
            st.subheader("Analyzed View")
            # Create a placeholder in col_img for the spinner so it shows up where the image will be
            img_placeholder = st.empty()
            
        with col_res:
            st.subheader("Results Dashboard")
            res_placeholder = st.empty()
            
        with img_placeholder.container():
            with st.spinner("Running ML Pipeline & Generating Audio..."):
                try:
                    pipeline_start = time.perf_counter()
                    
                    device_str = "cuda" if torch.cuda.is_available() else "cpu"
                    parsed = frame_parser.parse_frame(
                        frame_bgr, 
                        shorten_tts_commands=SHARED_SETTINGS.get("SHORTEN_TTS_COMMANDS", True),
                        sync_after_yolo=(device_str == "cuda"),
                        sync_after_depth=(device_str == "cuda"),
                        confidence_threshold=0.25,
                        hazard_return_coords=True
                    )
                    
                    latencies = parsed.get("latencies", {})
                    yolo_latency_ms = latencies.get("yolo_latency_ms", 0.0)
                    depth_latency_ms = latencies.get("depth_latency_ms", 0.0)
                    hazard_scan_latency_ms = latencies.get("hazard_scan_latency_ms", 0.0)
                    navigation_latency_ms = latencies.get("navigation_latency_ms", 0.0)
                    
                    tts_text = parsed["tts_text"]
                    nav_command = parsed["nav_command"]
                    frame_boxes = parsed["frame_boxes"]
                    hazard_result = parsed.get("hazard_result", {})
                    hazard_global = hazard_result.get("global_summary", {}) if hazard_result else {}
                    
                    nav_logic = parsed.get("navigation_logic")
                    zone_risks = parsed.get("zone_risks", {})

                    # Audio Synthesis 
                    tts_start_time = time.perf_counter()
                    wav_bytes = None
                    
                    try:
                        tts_engine._validate_paths()
                        
                        # First try cache to eliminate generation latency
                        wav_bytes = tts_cache.get(tts_text)
                        
                        # If cache miss, generate audio and put into cache
                        if wav_bytes is None:
                            wav_bytes = tts_engine.synthesize_wav_bytes(tts_text)
                            tts_cache.put(tts_text, wav_bytes)
                    except Exception as e:
                        st.error(f"TTS synthesis error: {e}")
                        
                    tts_latency_ms = (time.perf_counter() - tts_start_time) * 1000.0
                    pipeline_latency_ms = (time.perf_counter() - pipeline_start) * 1000.0

                    # Overlay Drawing Logic
                    vis = frame_bgr.copy()
                    h, w = vis.shape[:2]
                    
                    draw_depth_hazard_overlay(vis, hazard_result, alpha=0.30)
                    draw_nav_zones(vis, nav_logic, zone_risks)

                    # Draw Hazard Pixel summary
                    if hazard_global:
                        cv2.putText(
                            vis,
                            f"Hazard px (D/W): {hazard_global.get('danger_pixel_count', 0)}/{hazard_global.get('warning_pixel_count', 0)}",
                            (10, max(60, h - 90)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

                    # Draw YOLO Boxes
                    for d in frame_boxes:
                        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 220, 255), 2)
                        cv2.putText(
                            vis,
                            f"{d['class_name']} {d.get('confidence', 0.0):.2f}",
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
                    
                    # Display the final annotated image
                    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                except Exception as e:
                    import traceback
                    st.error(f"An error occurred during processing: {e}")
                    st.code(traceback.format_exc())
                    
        # Populate the Results Dashboard in the right column
        if 'parsed' in locals():
            with col_res:
                st.success("Analysis Complete!")
                
                st.markdown("### TTS Transcript")
                st.info(tts_text)
                
                if wav_bytes is not None:
                    st.markdown("### Spoken Navigation Warning")
                    st.audio(wav_bytes, format='audio/wav')
                
                st.divider()
                st.markdown("### Latency Breakdown")
                st.metric("Total Pipeline", f"{pipeline_latency_ms:.1f} ms")
                metrics_col1, metrics_col2 = st.columns(2)
                metrics_col1.metric("YOLO", f"{yolo_latency_ms:.1f} ms")
                metrics_col2.metric("Depth", f"{depth_latency_ms:.1f} ms")
                metrics_col1.metric("Hazard Scan", f"{hazard_scan_latency_ms:.1f} ms")
                metrics_col2.metric("TTS Generation", f"{tts_latency_ms:.1f} ms")
else:
    # Landing page state
    st.info("Please upload an image from the sidebar to begin.")
