import os
import time

import cv2
import torch

from mechanics.depth_estimation import estimate_distance_from_depth
from mechanics.depth_estimation import DepthEstimator
from mechanics.depth_estimation import scan_depth_hazards
from mechanics.navigation_logic import NavigationLogic
from mechanics.object_detection import draw_centered_label
from mechanics.object_detection import ObjectDetector
from mechanics.performance_pipeline import PipelinePerformanceTracker
from mechanics.tts_command_utils import build_short_tts_command
from mechanics.tts_config import DEFAULT_SHORTEN_TTS_COMMANDS
from mechanics.tts_config import DEFAULT_TTS_DIRECT_PLAYBACK
from mechanics.tts_config import DEFAULT_TTS_MIN_INTERVAL_SECONDS
from mechanics.tts_config import DEFAULT_TTS_PHRASE_CACHE_MAXSIZE
from mechanics.tts_config import DEFAULT_TTS_PLAY_AUDIO
from mechanics.tts_config import DEFAULT_TTS_QUEUE_MAXSIZE
from mechanics.tts_config import DEFAULT_TTS_SPEAK_ONCE_PER_EXECUTION
from mechanics.tts_config import DEFAULT_TTS_SPEAK_ON_COMMAND_CHANGE
from mechanics.tts_config import DEFAULT_TTS_USE_PROCESS_WORKER
from mechanics.text_to_speech import TextToSpeech
from mechanics.text_to_speech import TTSRuntimeController
# from nav_slm_augment_logic import NavigationSLMAugmentLogic


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", ".env"))



def env_rel_path(env_key, default_rel_path):
	return os.path.abspath(os.path.join(BASE_DIR, os.getenv(env_key, default_rel_path)))


def load_env_file(env_file_path):
	if not os.path.exists(env_file_path):
		return
	with open(env_file_path, "r", encoding="utf-8") as f:
		for raw_line in f:
			line = raw_line.strip()
			if not line or line.startswith("#") or "=" not in line:
				continue
			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip().strip('"').strip("'")
			if key and key not in os.environ:
				os.environ[key] = value


load_env_file(ENV_FILE)

YOLO_WEIGHTS = env_rel_path(
	"YOLO_WEIGHTS_REL",
	os.path.join("..", "model_training", "object_detection", "best-weights", "YOLOv8n-uni.pt"),
)
DEPTH_MODEL_DIR = env_rel_path(
	"DEPTH_MODEL_DIR_REL",
	# os.path.join("..", "model_training", "depth_estimation", "model_weights"),
	os.path.join(
		"..",
		"model_training",
		"depth_estimation",
		"model_weights",
		"depth_anything_v2_metric_hypersim_vits.pth",
	),
)

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

# 0 = deterministic navigation, 1 = SLM-augmented navigation
NAV_LOGIC_MODE = int(os.getenv("NAV_LOGIC_MODE", "0"))

TTS_ENABLED = os.getenv("TTS_ENABLED", "1") == "1"
# Quality-first defaults for cleaner audio output.
SHORTEN_TTS_COMMANDS = (
	os.getenv("SHORTEN_TTS_COMMANDS", "1" if DEFAULT_SHORTEN_TTS_COMMANDS else "0") == "1"
)
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
TTS_PHRASE_CACHE_MAXSIZE = int(
	os.getenv("TTS_PHRASE_CACHE_MAXSIZE", str(DEFAULT_TTS_PHRASE_CACHE_MAXSIZE))
)
PIPER_EXE = env_rel_path("PIPER_EXE_REL", os.path.join("piper", "piper.exe"))
PIPER_VOICE_MODEL = env_rel_path(
	"PIPER_VOICE_MODEL_REL", os.path.join("piper_voices", "en_US-amy-medium.onnx")
)
PIPER_VOICE_CONFIG = env_rel_path(
	"PIPER_VOICE_CONFIG_REL", os.path.join("piper_voices", "en_US-amy-medium.onnx.json")
)

ZONE_SHADE_OPACITY = float(os.getenv("ZONE_SHADE_OPACITY", "0.1"))
DEPTH_HAZARD_ENABLED = os.getenv("DEPTH_HAZARD_ENABLED", "1") == "1"
DEPTH_DANGER_THRESHOLD_M = float(os.getenv("DEPTH_DANGER_THRESHOLD_M", "1.2"))
DEPTH_WARNING_THRESHOLD_M = float(os.getenv("DEPTH_WARNING_THRESHOLD_M", "2.0"))
DEPTH_HAZARD_MASK_ALPHA = float(os.getenv("DEPTH_HAZARD_MASK_ALPHA", "0.28"))


def sync_cuda_if_needed(device):
	if device == "cuda" and torch.cuda.is_available():
		torch.cuda.synchronize()


def validate_inputs():
	if not os.path.exists(YOLO_WEIGHTS):
		raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
	if not os.path.isfile(DEPTH_MODEL_DIR):
		raise FileNotFoundError(f"Depth model file not found: {DEPTH_MODEL_DIR}")
	if PLOT_STREAM_MODE not in (0, 1):
		raise ValueError("PLOT_STREAM_MODE must be 0 (RGB) or 1 (depth map).")
	if NAV_LOGIC_MODE not in (0, 1):
		raise ValueError("NAV_LOGIC_MODE must be 0 (deterministic) or 1 (SLM).")
	if DEPTH_DANGER_THRESHOLD_M <= 0:
		raise ValueError("DEPTH_DANGER_THRESHOLD_M must be > 0.")
	if DEPTH_WARNING_THRESHOLD_M < DEPTH_DANGER_THRESHOLD_M:
		raise ValueError("DEPTH_WARNING_THRESHOLD_M must be >= DEPTH_DANGER_THRESHOLD_M.")
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


def process_frame(
	frame_idx,
	frame_bgr,
	object_detector,
	depth_estimator,
	device,
	navigation_logic,
	slm_navigation_logic,
	tts_controller,
	performance_tracker,
):
	frame_start = time.perf_counter()
	frame_metrics = performance_tracker.init_frame_metrics(frame_idx, NAV_LOGIC_MODE, WARMUP_FRAMES)

	frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
	yolo_start = time.perf_counter()
	_, frame_boxes = object_detector.predict(frame_bgr)
	sync_cuda_if_needed(device)
	frame_metrics["yolo_latency_ms"] = (time.perf_counter() - yolo_start) * 1000.0
	frame_metrics["detection_count"] = len(frame_boxes)

	depth_start = time.perf_counter()
	depth_float, depth_color = depth_estimator.predict(frame_rgb)
	if GPU_TIMING_STRICT_SYNC:
		sync_cuda_if_needed(device)
	frame_metrics["depth_latency_ms"] = (time.perf_counter() - depth_start) * 1000.0

	hazard_result = None
	hazard_start = time.perf_counter()
	if DEPTH_HAZARD_ENABLED:
		hazard_result = scan_depth_hazards(
			depth_float,
			danger_threshold_m=DEPTH_DANGER_THRESHOLD_M,
			warning_threshold_m=DEPTH_WARNING_THRESHOLD_M,
			return_masks=True,
			return_coords=False,
		)
	frame_metrics["depth_hazard_latency_ms"] = (time.perf_counter() - hazard_start) * 1000.0
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

	plot_frame = frame_bgr.copy() if PLOT_STREAM_MODE == 0 else depth_color.copy()
	if DEPTH_HAZARD_ENABLED and hazard_result is not None and PLOT_STREAM_MODE == 0:
		apply_depth_hazard_overlay(plot_frame, hazard_result)

	spatial_start = time.perf_counter()
	for det in frame_boxes:
		bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]
		object_depth, relative_distance = estimate_distance_from_depth(depth_float, bbox)
		det["depth_relative"] = object_depth
		det["distance_relative"] = relative_distance

		x1, y1, x2, y2 = bbox
		center_x = (x1 + x2) // 2
		center_y = (y1 + y2) // 2
		distance_label = "dist: N/A" if relative_distance is None else f"dist: {relative_distance:.3f}"

		cv2.rectangle(plot_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(
			plot_frame,
			f"{det['class_name']} {det['confidence']:.2f}",
			(x1, max(20, y1 - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.4,
			(0, 255, 0),
			2,
		)
		draw_centered_label(plot_frame, distance_label, center_x, center_y)

	nav_detections = [
		{
			"class": det["class_name"],
			"bbox": [det["x1"], det["y1"], det["x2"], det["y2"]],
			"depth": det.get("depth_relative"),
			"distance": det.get("distance_relative"),
		}
		for det in frame_boxes
	]
	frame_metrics["spatial_latency_ms"] = (time.perf_counter() - spatial_start) * 1000.0

	navigation_start = time.perf_counter()
	deterministic_start = time.perf_counter()
	zone_risks, nav_command = navigation_logic.process_detections(
		nav_detections,
		depth_hazard=hazard_result,
	)
	frame_metrics["deterministic_nav_latency_ms"] = (
		(time.perf_counter() - deterministic_start) * 1000.0
	)
	if NAV_LOGIC_MODE == 1:
		slm_start = time.perf_counter()
		slm_result = slm_navigation_logic.decide_instruction(
			zone_risks,
			nav_detections,
			frame_width=frame_bgr.shape[1],
		)
		nav_command = slm_result["instruction"]
		frame_metrics["slm_nav_latency_ms"] = (time.perf_counter() - slm_start) * 1000.0
	frame_metrics["navigation_latency_ms"] = (time.perf_counter() - navigation_start) * 1000.0
	tts_text = build_short_tts_command(nav_command) if SHORTEN_TTS_COMMANDS else nav_command

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
	return plot_frame, frame_metrics


def main():
	validate_inputs()
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")
	print(f"Plot stream mode: {PLOT_STREAM_MODE} ({'rgb' if PLOT_STREAM_MODE == 0 else 'depth'})")
	print(f"Navigation logic mode: {NAV_LOGIC_MODE} ({'deterministic' if NAV_LOGIC_MODE == 0 else 'slm'})")
	print(f"Max inference duration: {INFERENCE_DURATION_MINUTES} minutes")
	print(f"Warmup frames: {WARMUP_FRAMES}")
	print(f"Resource sampling interval: every {RESOURCE_SAMPLE_EVERY_N_FRAMES} frames")
	print(f"Async TTS enabled: {ENABLE_ASYNC_TTS}")
	print(f"Process TTS worker enabled: {TTS_USE_PROCESS_WORKER}")
	print(f"Shorten TTS commands: {SHORTEN_TTS_COMMANDS}")
	print(f"TTS phrase cache size: {TTS_PHRASE_CACHE_MAXSIZE}")
	print(f"Benchmark display disabled: {DISABLE_DISPLAY_FOR_BENCHMARK}")

	performance_tracker = PipelinePerformanceTracker(
		base_dir=BASE_DIR,
		nav_logic_mode=NAV_LOGIC_MODE,
		plot_stream_mode=PLOT_STREAM_MODE,
		video_source=VIDEO_SOURCE,
		show_windows=SHOW_WINDOWS,
		tts_enabled=TTS_ENABLED,
		yolo_weights=YOLO_WEIGHTS,
		depth_model_dir=DEPTH_MODEL_DIR,
	)

	model_load_start = time.perf_counter()
	object_detector, depth_estimator = load_models(device)
	performance_tracker.set_model_load_ms((time.perf_counter() - model_load_start) * 1000.0)

	cap = cv2.VideoCapture(VIDEO_SOURCE)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

	navigation_logic = None
	slm_navigation_logic = None
	tts_controller = None
	frame_idx = 0

	if TTS_ENABLED:
		tts_engine = TextToSpeech(
			piper_executable=PIPER_EXE,
			voice_model_path=PIPER_VOICE_MODEL,
			voice_config_path=PIPER_VOICE_CONFIG,
			phrase_cache_maxsize=TTS_PHRASE_CACHE_MAXSIZE,
		)
		tts_engine.load_engine()
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

			frame_idx += 1
			if navigation_logic is None:
				navigation_logic = NavigationLogic(frame_width=frame_bgr.shape[1])
				# if NAV_LOGIC_MODE == 1:
				# 	slm_navigation_logic = NavigationSLMAugmentLogic(device=device)

			combined, frame_metrics = process_frame(
				frame_idx,
				frame_bgr,
				object_detector,
				depth_estimator,
				device,
				navigation_logic,
				slm_navigation_logic,
				tts_controller,
				performance_tracker,
			)

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
						performance_tracker.log_frame(frame_idx, frame_metrics, resource_snapshot)
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
				frame_idx,
				frame_metrics,
				device,
				RESOURCE_SAMPLE_EVERY_N_FRAMES,
			)
			performance_tracker.log_frame(frame_idx, frame_metrics, resource_snapshot)

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
