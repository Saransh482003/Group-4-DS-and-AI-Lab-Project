import os
import time

import cv2
import torch

from mechanics.depth_estimation import DepthEstimator
from mechanics.frame_parser import SharedFrameParser
from mechanics.navigation_logic import NavigationLogic
from mechanics.object_detection import draw_centered_label
from mechanics.object_detection import ObjectDetector
from performance_pipeline import PipelinePerformanceTracker
from mechanics.runtime_settings import load_shared_runtime_settings
from mechanics.tts_config import DEFAULT_TTS_DIRECT_PLAYBACK
from mechanics.tts_config import DEFAULT_TTS_MIN_INTERVAL_SECONDS
from mechanics.tts_config import DEFAULT_TTS_PHRASE_CACHE_MAXSIZE
from mechanics.tts_config import DEFAULT_TTS_PLAY_AUDIO
from mechanics.tts_config import DEFAULT_TTS_QUEUE_MAXSIZE
from mechanics.tts_config import DEFAULT_TTS_LATEST_ONLY_QUEUE
from mechanics.tts_config import DEFAULT_TTS_ENABLE_PRIORITY_PREEMPT
from mechanics.tts_config import DEFAULT_TTS_PREWARM_COMMON_PHRASES
from mechanics.tts_config import DEFAULT_TTS_SPEAK_ONCE_PER_EXECUTION
from mechanics.tts_config import DEFAULT_TTS_SPEAK_ON_COMMAND_CHANGE
from mechanics.tts_config import DEFAULT_TTS_USE_RUNTIME_SYNTHESIS
from mechanics.tts_config import DEFAULT_TTS_USE_PROCESS_WORKER
from mechanics.text_to_speech import TextToSpeech
from mechanics.text_to_speech import TTSRuntimeController


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", ".env"))
SHARED_SETTINGS = load_shared_runtime_settings(env_file_path=ENV_FILE)

YOLO_WEIGHTS = SHARED_SETTINGS["YOLO_WEIGHTS"]
DEPTH_MODEL_DIR = SHARED_SETTINGS["DEPTH_MODEL_FILE"]

VIDEO_SOURCE = int(os.getenv("VIDEO_SOURCE", "1"))
SHOW_WINDOWS = os.getenv("SHOW_WINDOWS", "1") == "1"

# 0 = plot overlays on RGB frame, 1 = plot overlays on depth map
PLOT_STREAM_MODE = int(os.getenv("PLOT_STREAM_MODE", "0"))

INFERENCE_DURATION_MINUTES = int(os.getenv("INFERENCE_DURATION_MINUTES", "10"))
WARMUP_FRAMES = int(os.getenv("WARMUP_FRAMES", "15"))
RESOURCE_SAMPLE_EVERY_N_FRAMES = int(os.getenv("RESOURCE_SAMPLE_EVERY_N_FRAMES", "10"))
DISABLE_DISPLAY_FOR_BENCHMARK = os.getenv("DISABLE_DISPLAY_FOR_BENCHMARK", "0") == "1"
ENABLE_ASYNC_TTS = os.getenv("ENABLE_ASYNC_TTS", "1") == "1"
GPU_TIMING_STRICT_SYNC = os.getenv("GPU_TIMING_STRICT_SYNC", "0") == "1"

TTS_ENABLED = os.getenv("TTS_ENABLED", "1") == "1"
# Quality-first defaults for cleaner audio output.
SHORTEN_TTS_COMMANDS = SHARED_SETTINGS["SHORTEN_TTS_COMMANDS"]
TTS_PLAY_AUDIO = os.getenv("TTS_PLAY_AUDIO", "1" if DEFAULT_TTS_PLAY_AUDIO else "0") == "1"
TTS_DIRECT_PLAYBACK = (
	os.getenv("TTS_DIRECT_PLAYBACK", "1" if DEFAULT_TTS_DIRECT_PLAYBACK else "0") == "1"
)
TTS_SPEAK_ONCE_PER_EXECUTION = (
	os.getenv("TTS_SPEAK_ONCE_PER_EXECUTION", "1" if DEFAULT_TTS_SPEAK_ONCE_PER_EXECUTION else "0")
	== "1"
)
TTS_SPEAK_ON_COMMAND_CHANGE = (
	os.getenv("TTS_SPEAK_ON_COMMAND_CHANGE", "1" if DEFAULT_TTS_SPEAK_ON_COMMAND_CHANGE else "0")
	== "1"
)
TTS_MIN_INTERVAL_SECONDS = float(
	os.getenv("TTS_MIN_INTERVAL_SECONDS", str(DEFAULT_TTS_MIN_INTERVAL_SECONDS))
)
TTS_QUEUE_MAXSIZE = int(os.getenv("TTS_QUEUE_MAXSIZE", str(DEFAULT_TTS_QUEUE_MAXSIZE)))
TTS_USE_PROCESS_WORKER = (
	os.getenv("TTS_USE_PROCESS_WORKER", "1" if DEFAULT_TTS_USE_PROCESS_WORKER else "0") == "1"
)
TTS_USE_RUNTIME_SYNTHESIS = (
	os.getenv("TTS_USE_RUNTIME_SYNTHESIS", "1" if DEFAULT_TTS_USE_RUNTIME_SYNTHESIS else "0")
	== "1"
)
TTS_ENABLE_PRIORITY_PREEMPT = (
	os.getenv("TTS_ENABLE_PRIORITY_PREEMPT", "1" if DEFAULT_TTS_ENABLE_PRIORITY_PREEMPT else "0")
	== "1"
)
TTS_LATEST_ONLY_QUEUE = (
	os.getenv("TTS_LATEST_ONLY_QUEUE", "1" if DEFAULT_TTS_LATEST_ONLY_QUEUE else "0") == "1"
)
TTS_PREWARM_COMMON_PHRASES = (
	os.getenv(
		"TTS_PREWARM_COMMON_PHRASES",
		"1" if DEFAULT_TTS_PREWARM_COMMON_PHRASES else "0",
	)
	== "1"
)
TTS_PHRASE_CACHE_MAXSIZE = int(
	os.getenv("TTS_PHRASE_CACHE_MAXSIZE", str(DEFAULT_TTS_PHRASE_CACHE_MAXSIZE))
)
PIPER_EXE = SHARED_SETTINGS["PIPER_EXE"]
PIPER_VOICE_MODEL = SHARED_SETTINGS["PIPER_VOICE_MODEL"]
PIPER_VOICE_CONFIG = SHARED_SETTINGS["PIPER_VOICE_CONFIG"]

ZONE_SHADE_OPACITY = float(os.getenv("ZONE_SHADE_OPACITY", "0.1"))
DEPTH_HAZARD_ENABLED = os.getenv("DEPTH_HAZARD_ENABLED", "1") == "1"
DEPTH_DANGER_THRESHOLD_M = SHARED_SETTINGS["DEPTH_DANGER_THRESHOLD_M"]
DEPTH_WARNING_THRESHOLD_M = SHARED_SETTINGS["DEPTH_WARNING_THRESHOLD_M"]
DEPTH_HAZARD_DANGER_WEIGHT = SHARED_SETTINGS["DEPTH_HAZARD_DANGER_WEIGHT"]
DEPTH_HAZARD_WARNING_WEIGHT = SHARED_SETTINGS["DEPTH_HAZARD_WARNING_WEIGHT"]
DEPTH_HAZARD_MASK_ALPHA = float(os.getenv("DEPTH_HAZARD_MASK_ALPHA", "0.28"))
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "2"))

COMMON_TTS_PHRASES = [
	"Turn left.",
	"Turn right.",
	"Go straight.",
	"Path blocked. Scan around.",
	"Move slightly left.",
	"Move slightly right.",
	"Searching for path. Turn back.",
]

def validate_inputs():
	if not os.path.exists(YOLO_WEIGHTS):
		raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
	if not os.path.isfile(DEPTH_MODEL_DIR):
		raise FileNotFoundError(f"Depth model file not found: {DEPTH_MODEL_DIR}")
	if PLOT_STREAM_MODE not in (0, 1):
		raise ValueError("PLOT_STREAM_MODE must be 0 (RGB) or 1 (depth map).")
	if DEPTH_DANGER_THRESHOLD_M <= 0:
		raise ValueError("DEPTH_DANGER_THRESHOLD_M must be > 0.")
	if DEPTH_WARNING_THRESHOLD_M < DEPTH_DANGER_THRESHOLD_M:
		raise ValueError("DEPTH_WARNING_THRESHOLD_M must be >= DEPTH_DANGER_THRESHOLD_M.")
	if PROCESS_EVERY_N_FRAMES < 1:
		raise ValueError("PROCESS_EVERY_N_FRAMES must be >= 1.")
	if TTS_ENABLED:
		if not os.path.exists(PIPER_EXE):
			raise FileNotFoundError(f"Piper executable not found: {PIPER_EXE}")
		if not os.path.exists(PIPER_VOICE_MODEL):
			raise FileNotFoundError(f"Piper voice model not found: {PIPER_VOICE_MODEL}")
		if not os.path.exists(PIPER_VOICE_CONFIG):
			raise FileNotFoundError(f"Piper voice config not found: {PIPER_VOICE_CONFIG}")


def load_models(device):
	object_detector = ObjectDetector(YOLO_WEIGHTS)
	object_detector.load_model()
	depth_estimator = DepthEstimator(DEPTH_MODEL_DIR, device=device)
	depth_estimator.load_model()
	return object_detector, depth_estimator


def apply_zone_shading(frame, navigation_logic):
	"""Apply subtle translucent shading for left/center/right navigation zones."""
	h, w = frame.shape[:2]
	left_end = int(getattr(navigation_logic, "left_end", int(0.30 * w)))
	center_end = int(getattr(navigation_logic, "center_end", int(0.70 * w)))

	overlay = frame.copy()
	cv2.rectangle(overlay, (0, 0), (left_end, h), (0, 0, 255), -1)
	cv2.rectangle(overlay, (left_end, 0), (center_end, h), (0, 255, 0), -1)
	cv2.rectangle(overlay, (center_end, 0), (w, h), (0, 0, 255), -1)

	cv2.addWeighted(overlay, ZONE_SHADE_OPACITY, frame, 1.0 - ZONE_SHADE_OPACITY, 0, frame)


def apply_depth_hazard_overlay(frame, hazard_result):
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

	cv2.addWeighted(overlay, DEPTH_HAZARD_MASK_ALPHA, frame, 1.0 - DEPTH_HAZARD_MASK_ALPHA, 0, frame)


def _create_plot_base_frame(frame_bgr, depth_color=None):
	if PLOT_STREAM_MODE == 0:
		plot_frame = frame_bgr.copy()
		gray_frame = cv2.cvtColor(plot_frame, cv2.COLOR_BGR2GRAY)
		return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
	if depth_color is not None:
		return depth_color.copy()
	return frame_bgr.copy()


def _draw_detection_boxes(plot_frame, detections):
	for det in detections:
		x1 = int(det["x1"])
		y1 = int(det["y1"])
		x2 = int(det["x2"])
		y2 = int(det["y2"])
		center_x = (x1 + x2) // 2
		center_y = (y1 + y2) // 2
		distance_value = det.get("distance_relative")
		if distance_value is None:
			distance_value = det.get("distance")
		distance_label = "dist: N/A" if distance_value is None else f"dist: {float(distance_value):.3f}"

		cv2.rectangle(plot_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(
			plot_frame,
			f"{det['class_name']} {float(det['confidence']):.2f}",
			(x1, max(20, y1 - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.4,
			(0, 255, 0),
			2,
		)
		draw_centered_label(plot_frame, distance_label, center_x, center_y)


def _build_cached_inference(frame_boxes, hazard_result, zone_risks, nav_command, tts_text):
	cached_boxes = [
		{
			"x1": int(det["x1"]),
			"y1": int(det["y1"]),
			"x2": int(det["x2"]),
			"y2": int(det["y2"]),
			"class_name": str(det["class_name"]),
			"confidence": float(det["confidence"]),
			"distance_relative": det.get("distance_relative"),
		}
		for det in frame_boxes
	]
	return {
		"frame_boxes": cached_boxes,
		"hazard_result": hazard_result,
		"zone_risks": dict(zone_risks),
		"nav_command": nav_command,
		"tts_text": tts_text,
	}


def process_frame(
	frame_idx,
	frame_bgr,
	frame_parser,
	tts_controller,
	performance_tracker,
):
	frame_start = time.perf_counter()
	frame_metrics = performance_tracker.init_frame_metrics(frame_idx, WARMUP_FRAMES)
	frame_metrics["frame_skipped"] = False
	frame_metrics["skip_reason"] = None
	frame_metrics["inference_reused"] = False

	parsed = frame_parser.parse_frame(
		frame_bgr,
		shorten_tts_commands=SHORTEN_TTS_COMMANDS,
		sync_after_yolo=True,
		sync_after_depth=GPU_TIMING_STRICT_SYNC,
	)

	latencies = parsed["latencies"]
	frame_boxes = parsed["frame_boxes"]
	depth_color = parsed["depth_color"]
	hazard_result = parsed["hazard_result"]
	zone_risks = parsed["zone_risks"]
	nav_command = parsed["nav_command"]
	tts_text = parsed["tts_text"]
	navigation_logic = parsed["navigation_logic"]

	frame_metrics["yolo_latency_ms"] = latencies.get("yolo_latency_ms")
	frame_metrics["detection_count"] = len(frame_boxes)
	frame_metrics["depth_latency_ms"] = latencies.get("depth_latency_ms")
	frame_metrics["depth_hazard_latency_ms"] = latencies.get("hazard_scan_latency_ms")
	if hazard_result:
		global_hazard = hazard_result.get("global_summary", {})
		frame_metrics["depth_hazard_danger_pixel_count"] = global_hazard.get("danger_pixel_count", 0)
		frame_metrics["depth_hazard_warning_pixel_count"] = global_hazard.get("warning_pixel_count", 0)
		frame_metrics["depth_hazard_near_pixel_count"] = global_hazard.get("near_pixel_count", 0)
		frame_metrics["depth_hazard_min_depth_m"] = global_hazard.get("min_depth_m")
	else:
		frame_metrics["depth_hazard_danger_pixel_count"] = 0
		frame_metrics["depth_hazard_warning_pixel_count"] = 0
		frame_metrics["depth_hazard_near_pixel_count"] = 0
		frame_metrics["depth_hazard_min_depth_m"] = None

	plot_frame = _create_plot_base_frame(frame_bgr, depth_color)
	if DEPTH_HAZARD_ENABLED and hazard_result is not None and PLOT_STREAM_MODE == 0:
		apply_depth_hazard_overlay(plot_frame, hazard_result)
	frame_metrics["spatial_latency_ms"] = latencies.get("spatial_latency_ms")
	_draw_detection_boxes(plot_frame, frame_boxes)
	frame_metrics["navigation_latency_ms"] = latencies.get("navigation_latency_ms")

	visualization_start = time.perf_counter()
	apply_zone_shading(plot_frame, navigation_logic)
	navigation_logic.draw_overlays(plot_frame, zone_risks, nav_command)
	frame_metrics["visualization_latency_ms"] = (time.perf_counter() - visualization_start) * 1000.0

	if tts_controller is not None:
		tts_start = time.perf_counter()
		try:
			tts_result = tts_controller.handle_command(tts_text)
			frame_metrics["tts_should_speak"] = tts_result["tts_should_speak"]
			frame_metrics["tts_skip_reason"] = tts_result["tts_skip_reason"]
			frame_metrics["tts_mode"] = tts_result["tts_mode"]
			frame_metrics["tts_error"] = tts_result["tts_error"]
			frame_metrics["tts_enqueue_ok"] = tts_result["tts_enqueue_ok"]
			frame_metrics["tts_cache_hit"] = tts_result.get("tts_cache_hit")
			frame_metrics["tts_worker_latency_ms"] = tts_result.get("tts_worker_latency_ms")
		except Exception as exc:
			print(f"TTS error: {exc}")
			frame_metrics["tts_error"] = str(exc)

		frame_metrics["tts_latency_ms"] = (time.perf_counter() - tts_start) * 1000.0

	frame_metrics["frame_total_latency_ms"] = (time.perf_counter() - frame_start) * 1000.0
	frame_metrics["zone_risks"] = zone_risks
	frame_metrics["nav_command"] = nav_command
	frame_metrics["tts_text"] = tts_text
	cached_inference = _build_cached_inference(
		frame_boxes=frame_boxes,
		hazard_result=hazard_result,
		zone_risks=zone_risks,
		nav_command=nav_command,
		tts_text=tts_text,
	)
	return plot_frame, frame_metrics, cached_inference, navigation_logic


def render_cached_frame(frame_idx, frame_bgr, navigation_logic, performance_tracker, cached_inference):
	frame_start = time.perf_counter()
	frame_metrics = performance_tracker.init_frame_metrics(frame_idx, WARMUP_FRAMES)
	frame_metrics["frame_skipped"] = True
	frame_metrics["skip_reason"] = "policy_nth_frame_skip"
	frame_metrics["inference_reused"] = True
	frame_metrics["tts_should_speak"] = False
	frame_metrics["tts_skip_reason"] = "frame_skipped"

	hazard_result = cached_inference.get("hazard_result") if cached_inference else None
	if hazard_result:
		global_hazard = hazard_result.get("global_summary", {})
		frame_metrics["depth_hazard_danger_pixel_count"] = global_hazard.get("danger_pixel_count", 0)
		frame_metrics["depth_hazard_warning_pixel_count"] = global_hazard.get("warning_pixel_count", 0)
		frame_metrics["depth_hazard_near_pixel_count"] = global_hazard.get("near_pixel_count", 0)
		frame_metrics["depth_hazard_min_depth_m"] = global_hazard.get("min_depth_m")

	cached_boxes = cached_inference.get("frame_boxes", []) if cached_inference else []
	frame_metrics["detection_count"] = len(cached_boxes)

	zone_risks = cached_inference.get("zone_risks") if cached_inference else None
	if zone_risks is None:
		zone_risks = {"left": 0.0, "center": 0.0, "right": 0.0}

	nav_command = (
		cached_inference.get("nav_command", "Searching for path. Turn back.")
		if cached_inference
		else "Searching for path. Turn back."
	)
	tts_text = (
		cached_inference.get("tts_text", nav_command)
		if cached_inference
		else nav_command
	)

	visualization_start = time.perf_counter()
	plot_frame = _create_plot_base_frame(frame_bgr)
	if DEPTH_HAZARD_ENABLED and hazard_result is not None and PLOT_STREAM_MODE == 0:
		apply_depth_hazard_overlay(plot_frame, hazard_result)
	_draw_detection_boxes(plot_frame, cached_boxes)
	apply_zone_shading(plot_frame, navigation_logic)
	navigation_logic.draw_overlays(plot_frame, zone_risks, nav_command)
	frame_metrics["visualization_latency_ms"] = (time.perf_counter() - visualization_start) * 1000.0

	frame_metrics["frame_total_latency_ms"] = (time.perf_counter() - frame_start) * 1000.0
	frame_metrics["zone_risks"] = zone_risks
	frame_metrics["nav_command"] = nav_command
	frame_metrics["tts_text"] = tts_text
	return plot_frame, frame_metrics


def main():
	validate_inputs()
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")
	print(f"Plot stream mode: {PLOT_STREAM_MODE} ({'rgb' if PLOT_STREAM_MODE == 0 else 'depth'})")
	print(f"Max inference duration: {INFERENCE_DURATION_MINUTES} minutes")
	print(f"Warmup frames: {WARMUP_FRAMES}")
	print(f"Resource sampling interval: every {RESOURCE_SAMPLE_EVERY_N_FRAMES} frames")
	print(f"Async TTS enabled: {ENABLE_ASYNC_TTS}")
	print(f"Process TTS worker enabled: {TTS_USE_PROCESS_WORKER}")
	print(f"TTS runtime synthesis mode: {TTS_USE_RUNTIME_SYNTHESIS}")
	print(f"TTS priority preemption: {TTS_ENABLE_PRIORITY_PREEMPT}")
	print(f"TTS latest-only queue: {TTS_LATEST_ONLY_QUEUE}")
	print(f"TTS prewarm common phrases: {TTS_PREWARM_COMMON_PHRASES}")
	print(f"Shorten TTS commands: {SHORTEN_TTS_COMMANDS}")
	print(f"TTS phrase cache size: {TTS_PHRASE_CACHE_MAXSIZE}")
	print(f"Process every N frames: {PROCESS_EVERY_N_FRAMES}")
	print(f"Benchmark display disabled: {DISABLE_DISPLAY_FOR_BENCHMARK}")

	performance_tracker = PipelinePerformanceTracker(
		base_dir=BASE_DIR,
		plot_stream_mode=PLOT_STREAM_MODE,
		video_source=VIDEO_SOURCE,
		show_windows=SHOW_WINDOWS,
		tts_enabled=TTS_ENABLED,
		yolo_weights=YOLO_WEIGHTS,
		depth_model_dir=DEPTH_MODEL_DIR,
		process_every_n_frames=PROCESS_EVERY_N_FRAMES,
	)

	model_load_start = time.perf_counter()
	object_detector, depth_estimator = load_models(device)
	performance_tracker.set_model_load_ms((time.perf_counter() - model_load_start) * 1000.0)
	frame_parser = SharedFrameParser(
		object_detector=object_detector,
		depth_estimator=depth_estimator,
		nav_logic_factory=lambda frame_width: NavigationLogic(
			frame_width=frame_width,
			depth_hazard_danger_weight=DEPTH_HAZARD_DANGER_WEIGHT,
			depth_hazard_warning_weight=DEPTH_HAZARD_WARNING_WEIGHT,
		),
		device=device,
		depth_hazard_enabled=DEPTH_HAZARD_ENABLED,
		danger_threshold_m=DEPTH_DANGER_THRESHOLD_M,
		warning_threshold_m=DEPTH_WARNING_THRESHOLD_M,
		stateful_navigation=True,
	)

	cap = cv2.VideoCapture(VIDEO_SOURCE)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

	navigation_logic = None
	tts_controller = None
	captured_frame_idx = 0
	processed_frame_idx = 0
	cached_inference = None

	if TTS_ENABLED:
		tts_engine = TextToSpeech(
			piper_executable=PIPER_EXE,
			voice_model_path=PIPER_VOICE_MODEL,
			voice_config_path=PIPER_VOICE_CONFIG,
			phrase_cache_maxsize=TTS_PHRASE_CACHE_MAXSIZE,
		)
		tts_engine.load_engine()
		if TTS_PREWARM_COMMON_PHRASES:
			if TTS_USE_RUNTIME_SYNTHESIS:
				prewarm = tts_engine.prewarm_phrase_cache(COMMON_TTS_PHRASES)
				print(
					"TTS prewarm complete (cache): "
					f"new={prewarm['warmed']} cached={prewarm['skipped']}"
				)
			else:
				prewarm = tts_engine.prewarm_local_phrase_wavs(COMMON_TTS_PHRASES)
				print(
					"TTS prewarm complete (local wav store): "
					f"new={prewarm['warmed']} cached={prewarm['skipped']}"
				)
		tts_controller = TTSRuntimeController(
			tts_engine=tts_engine,
			enable_async=ENABLE_ASYNC_TTS,
			use_process_worker=TTS_USE_PROCESS_WORKER,
			direct_playback=TTS_DIRECT_PLAYBACK,
			play_audio=TTS_PLAY_AUDIO,
			speak_once_per_execution=TTS_SPEAK_ONCE_PER_EXECUTION,
			speak_on_command_change=TTS_SPEAK_ON_COMMAND_CHANGE,
			min_interval_seconds=TTS_MIN_INTERVAL_SECONDS,
			queue_maxsize=TTS_QUEUE_MAXSIZE,
			phrase_cache_maxsize=TTS_PHRASE_CACHE_MAXSIZE,
			latest_only_queue=TTS_LATEST_ONLY_QUEUE,
			use_runtime_synthesis=TTS_USE_RUNTIME_SYNTHESIS,
			enable_priority_preempt=TTS_ENABLE_PRIORITY_PREEMPT,
		)

	print("Press 'q' to stop.")
	print("If preview windows are unavailable, press Ctrl+C to stop.")
	run_start_time = time.time()
	max_duration_seconds = max(1, int(INFERENCE_DURATION_MINUTES * 60))
	preview_enabled = SHOW_WINDOWS and (not DISABLE_DISPLAY_FOR_BENCHMARK)
	preview_unavailable_logged = False

	try:
		while True:
			if (time.time() - run_start_time) >= max_duration_seconds:
				print("Reached configured inference duration. Stopping run.")
				break

			loop_start = time.perf_counter()
			capture_start = time.perf_counter()
			ok, frame_bgr = cap.read()
			capture_latency_ms = (time.perf_counter() - capture_start) * 1000.0
			if not ok:
				break

			captured_frame_idx += 1

			should_process = cached_inference is None or (
				((captured_frame_idx - 1) % PROCESS_EVERY_N_FRAMES) == 0
			)
			if should_process:
				processed_frame_idx += 1
				combined, frame_metrics, cached_inference, navigation_logic = process_frame(
					processed_frame_idx,
					frame_bgr,
					frame_parser,
					tts_controller,
					performance_tracker,
				)
			else:
				if navigation_logic is None:
					navigation_logic = frame_parser.get_current_navigation_logic()
				if navigation_logic is None:
					navigation_logic = NavigationLogic(
						frame_width=frame_bgr.shape[1],
						depth_hazard_danger_weight=DEPTH_HAZARD_DANGER_WEIGHT,
						depth_hazard_warning_weight=DEPTH_HAZARD_WARNING_WEIGHT,
					)
				combined, frame_metrics = render_cached_frame(
					processed_frame_idx,
					frame_bgr,
					navigation_logic,
					performance_tracker,
					cached_inference,
				)

			frame_metrics["captured_frame_idx"] = captured_frame_idx
			frame_metrics["processed_frame_idx"] = processed_frame_idx
			frame_metrics["process_every_n_frames"] = PROCESS_EVERY_N_FRAMES

			video_write_latency_ms = 0.0

			display_latency_ms = 0.0
			if preview_enabled:
				display_start = time.perf_counter()
				try:
					cv2.imshow("Depth + BBoxes + Navigation", combined)
					if cv2.waitKey(1) & 0xFF == ord("q"):
						display_latency_ms = (time.perf_counter() - display_start) * 1000.0
						performance_tracker.attach_loop_metrics(
							frame_metrics,
							loop_start,
							capture_latency_ms,
							video_write_latency_ms,
							display_latency_ms,
						)
						resource_snapshot = performance_tracker.sample_resources(device)
						performance_tracker.log_frame(captured_frame_idx, frame_metrics, resource_snapshot)
						break
				except cv2.error:
					if not preview_unavailable_logged:
						print("OpenCV GUI is not available. Continuing without preview windows.")
						preview_unavailable_logged = True
					preview_enabled = False
				display_latency_ms = (time.perf_counter() - display_start) * 1000.0

			performance_tracker.attach_loop_metrics(
				frame_metrics,
				loop_start,
				capture_latency_ms,
				video_write_latency_ms,
				display_latency_ms,
			)
			resource_snapshot = performance_tracker.maybe_sample_resources(
				captured_frame_idx,
				frame_metrics,
				device,
				RESOURCE_SAMPLE_EVERY_N_FRAMES,
			)
			performance_tracker.log_frame(captured_frame_idx, frame_metrics, resource_snapshot)

	except KeyboardInterrupt:
		print("Stopped by user (Ctrl+C).")
	finally:
		if tts_controller is not None:
			tts_controller.stop()
		cap.release()
		cv2.destroyAllWindows()

	performance_tracker.finalize_run()
	print(f"Saved performance logs to: {performance_tracker.run_dir}")


if __name__ == "__main__":
	main()
