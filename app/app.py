import json
import os
import queue
import threading
import time

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

from image_inference import draw_centered_label
from image_inference import estimate_distance_from_depth
from image_inference import run_depth_inference
from nav_deterministic_logic import NavigationDeterministicLogic
from nav_slm_augment_logic import NavigationSLMAugmentLogic
from nav_tts_piper import PiperTTS
from performance_metrics import PerformanceMetricsLogger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS = os.path.join(
	BASE_DIR,
	"..",
	"model_training",
	"object_detection",
	"best-weights",
	"YOLOv8n-uni.pt",
)
DEPTH_MODEL_DIR = os.path.abspath(
	os.path.join(BASE_DIR, "..", "model_training", "depth_estimation", "model_weights")
)

VIDEO_SOURCE = 1
SHOW_WINDOWS = True

# 0 = plot overlays on RGB frame, 1 = plot overlays on depth map
PLOT_STREAM_MODE = 0

OUTPUT_JSON = os.path.join(BASE_DIR, "bbox_output.json")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "yolo_depth_output.mp4")
INFERENCE_DURATION_MINUTES = 10
WARMUP_FRAMES = 15
RESOURCE_SAMPLE_EVERY_N_FRAMES = 10
DISABLE_DISPLAY_FOR_BENCHMARK = False
ENABLE_ASYNC_TTS = True
GPU_TIMING_STRICT_SYNC = False

# 0 = deterministic navigation, 1 = SLM-augmented navigation
NAV_LOGIC_MODE = 0

TTS_ENABLED = True
TTS_PLAY_AUDIO = False
TTS_DIRECT_PLAYBACK = True
TTS_SPEAK_ONCE_PER_EXECUTION = False
TTS_SPEAK_ON_COMMAND_CHANGE = True
TTS_MIN_INTERVAL_SECONDS = 1.2
PIPER_EXE = os.path.join(BASE_DIR, "piper", "piper.exe")
PIPER_VOICE_MODEL = os.path.join(BASE_DIR, "piper_voices", "en_US-amy-medium.onnx")
PIPER_VOICE_CONFIG = os.path.join(BASE_DIR, "piper_voices", "en_US-amy-medium.onnx.json")

ZONE_SHADE_OPACITY = 0.1


class AsyncTTSWorker:
	def __init__(self, tts_engine, direct_playback=True, play_audio=False):
		self.tts_engine = tts_engine
		self.direct_playback = direct_playback
		self.play_audio = play_audio
		self.queue = queue.Queue(maxsize=5)
		self._stop_event = threading.Event()
		self._thread = threading.Thread(target=self._run, daemon=True)
		self.last_error = None

	def start(self):
		self._thread.start()

	def stop(self):
		self._stop_event.set()
		try:
			self.queue.put_nowait(None)
		except queue.Full:
			pass
		self._thread.join(timeout=2.0)

	def enqueue(self, text):
		try:
			self.queue.put_nowait(text)
			return True
		except queue.Full:
			self.last_error = "tts_queue_full"
			return False

	def _run(self):
		while not self._stop_event.is_set():
			try:
				item = self.queue.get(timeout=0.1)
			except queue.Empty:
				continue
			if item is None:
				self.queue.task_done()
				continue
			try:
				if self.direct_playback:
					self.tts_engine.speak_direct_safe(item, fallback_play_audio=True)
				else:
					self.tts_engine.synthesize(item, play_audio=self.play_audio)
			except Exception as exc:
				self.last_error = str(exc)
			finally:
				self.queue.task_done()


def sync_cuda_if_needed(device):
	if device == "cuda" and torch.cuda.is_available():
		torch.cuda.synchronize()


def validate_inputs():
	if not os.path.exists(YOLO_WEIGHTS):
		raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
	if not os.path.isdir(DEPTH_MODEL_DIR):
		raise FileNotFoundError(f"Depth model directory not found: {DEPTH_MODEL_DIR}")
	if PLOT_STREAM_MODE not in (0, 1):
		raise ValueError("PLOT_STREAM_MODE must be 0 (RGB) or 1 (depth map).")
	if NAV_LOGIC_MODE not in (0, 1):
		raise ValueError("NAV_LOGIC_MODE must be 0 (deterministic) or 1 (SLM).")
	if TTS_ENABLED:
		if not os.path.exists(PIPER_EXE):
			raise FileNotFoundError(f"Piper executable not found: {PIPER_EXE}")
		if not os.path.exists(PIPER_VOICE_MODEL):
			raise FileNotFoundError(f"Piper voice model not found: {PIPER_VOICE_MODEL}")
		if not os.path.exists(PIPER_VOICE_CONFIG):
			raise FileNotFoundError(f"Piper voice config not found: {PIPER_VOICE_CONFIG}")


def load_models(device):
	yolo_model = YOLO(YOLO_WEIGHTS)
	depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_DIR, local_files_only=True)
	depth_model = AutoModelForDepthEstimation.from_pretrained(
		DEPTH_MODEL_DIR,
		local_files_only=True,
	).to(device)
	depth_model.eval()
	return yolo_model, depth_processor, depth_model


def apply_zone_shading(frame, navigation_logic):
	"""Apply subtle translucent shading for left/center/right navigation zones."""
	h, w = frame.shape[:2]
	left_end = int(getattr(navigation_logic, "left_end", int(0.30 * w)))
	center_end = int(getattr(navigation_logic, "center_end", int(0.70 * w)))

	overlay = frame.copy()
	# Red side zones and green center zone.
	cv2.rectangle(overlay, (0, 0), (left_end, h), (0, 0, 255), -1)
	cv2.rectangle(overlay, (left_end, 0), (center_end, h), (0, 255, 0), -1)
	cv2.rectangle(overlay, (center_end, 0), (w, h), (0, 0, 255), -1)

	cv2.addWeighted(overlay, ZONE_SHADE_OPACITY, frame, 1.0 - ZONE_SHADE_OPACITY, 0, frame)


def process_frame(
	frame_idx,
	frame_bgr,
	yolo_model,
	depth_processor,
	depth_model,
	device,
	navigation_logic,
	slm_navigation_logic,
	tts_engine,
	tts_worker,
	tts_state,
):
	frame_start = time.perf_counter()
	frame_metrics = {
		"yolo_latency_ms": None,
		"depth_latency_ms": None,
		"spatial_latency_ms": None,
		"navigation_latency_ms": None,
		"deterministic_nav_latency_ms": None,
		"slm_nav_latency_ms": None,
		"visualization_latency_ms": None,
		"tts_latency_ms": 0.0,
		"tts_should_speak": False,
		"tts_skip_reason": None,
		"tts_mode": None,
		"tts_error": None,
		"detection_count": 0,
		"frame_total_latency_ms": None,
		"nav_mode_name": "deterministic" if NAV_LOGIC_MODE == 0 else "slm",
		"is_warmup": frame_idx <= WARMUP_FRAMES,
		"resource_sampled": False,
		"tts_enqueue_ok": None,
	}

	frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
	yolo_start = time.perf_counter()
	yolo_result = yolo_model(frame_bgr, verbose=False)[0]
	sync_cuda_if_needed(device)
	frame_metrics["yolo_latency_ms"] = (time.perf_counter() - yolo_start) * 1000.0

	frame_boxes = []

	for box in yolo_result.boxes:
		x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
		conf = float(box.conf[0])
		cls_id = int(box.cls[0])
		cls_name = yolo_model.names[cls_id]

		detection = {
			"class_id": cls_id,
			"class_name": cls_name,
			"confidence": conf,
			"x1": x1,
			"y1": y1,
			"x2": x2,
			"y2": y2,
		}
		frame_boxes.append(detection)

	frame_metrics["detection_count"] = len(frame_boxes)

	depth_start = time.perf_counter()
	depth_float, depth_color = run_depth_inference(frame_rgb, depth_processor, depth_model, device)
	if GPU_TIMING_STRICT_SYNC:
		sync_cuda_if_needed(device)
	frame_metrics["depth_latency_ms"] = (time.perf_counter() - depth_start) * 1000.0
	if PLOT_STREAM_MODE == 0:
		plot_frame = frame_bgr.copy()
	else:
		plot_frame = depth_color.copy()

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
	zone_risks, nav_command = navigation_logic.process_detections(nav_detections)
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

	visualization_start = time.perf_counter()

	apply_zone_shading(plot_frame, navigation_logic)
	navigation_logic.draw_overlays(plot_frame, zone_risks, nav_command)
	frame_metrics["visualization_latency_ms"] = (time.perf_counter() - visualization_start) * 1000.0

	if tts_engine is not None:
		tts_start = time.perf_counter()
		now = time.time()
		should_speak = (not TTS_SPEAK_ONCE_PER_EXECUTION) or (not tts_state["spoken"])
		if not should_speak:
			frame_metrics["tts_skip_reason"] = "once_per_execution"
		if should_speak and TTS_SPEAK_ON_COMMAND_CHANGE:
			if tts_state.get("last_command") == nav_command:
				should_speak = False
				frame_metrics["tts_skip_reason"] = "command_unchanged"
		if should_speak and (now - tts_state.get("last_spoken_at", 0.0)) < TTS_MIN_INTERVAL_SECONDS:
			should_speak = False
			frame_metrics["tts_skip_reason"] = "rate_limited"

		frame_metrics["tts_should_speak"] = should_speak

		if should_speak:
			try:
				if ENABLE_ASYNC_TTS and tts_worker is not None:
					enqueue_ok = tts_worker.enqueue(nav_command)
					frame_metrics["tts_enqueue_ok"] = enqueue_ok
					frame_metrics["tts_mode"] = "async_queue"
					if not enqueue_ok and tts_worker.last_error:
						frame_metrics["tts_error"] = tts_worker.last_error
				elif TTS_DIRECT_PLAYBACK:
					playback_mode = tts_engine.speak_direct_safe(nav_command, fallback_play_audio=True)
					print(f"TTS Output: {playback_mode}")
					frame_metrics["tts_mode"] = playback_mode
				else:
					wav_path = tts_engine.synthesize(nav_command, play_audio=TTS_PLAY_AUDIO)
					print(f"TTS Output: {wav_path}")
					frame_metrics["tts_mode"] = "file"

				# Update TTS state at enqueue/dispatch time to preserve anti-chatter behavior.
				tts_state["spoken"] = True
				tts_state["last_command"] = nav_command
				tts_state["last_spoken_at"] = now
			except Exception as exc:
				print(f"TTS error: {exc}")
				frame_metrics["tts_error"] = str(exc)

		frame_metrics["tts_latency_ms"] = (time.perf_counter() - tts_start) * 1000.0

	combined_frame = plot_frame
	frame_metrics["frame_total_latency_ms"] = (time.perf_counter() - frame_start) * 1000.0
	frame_metrics["zone_risks"] = zone_risks
	frame_metrics["nav_command"] = nav_command
	frame_record = {
		"detections": frame_boxes,
		"zone_risks": zone_risks,
		"navigation_command": nav_command,
		"performance": frame_metrics,
	}
	return combined_frame, frame_record, frame_metrics


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
	print(f"Benchmark display disabled: {DISABLE_DISPLAY_FOR_BENCHMARK}")
	profiler = PerformanceMetricsLogger(
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
	yolo_model, depth_processor, depth_model = load_models(device)
	profiler.set_model_load_ms((time.perf_counter() - model_load_start) * 1000.0)

	cap = cv2.VideoCapture(VIDEO_SOURCE)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

	navigation_logic = None
	slm_navigation_logic = None
	tts_engine = None
	tts_worker = None
	tts_state = {"spoken": False, "last_command": None, "last_spoken_at": 0.0}
	video_writer = None
	all_frames = []
	frame_idx = 0

	if TTS_ENABLED:
		tts_engine = PiperTTS(
			piper_executable=PIPER_EXE,
			voice_model_path=PIPER_VOICE_MODEL,
			voice_config_path=PIPER_VOICE_CONFIG,
		)
		if ENABLE_ASYNC_TTS:
			tts_worker = AsyncTTSWorker(
				tts_engine=tts_engine,
				direct_playback=TTS_DIRECT_PLAYBACK,
				play_audio=TTS_PLAY_AUDIO,
			)
			tts_worker.start()

	print("Press 'q' to stop.")
	print("If preview windows are unavailable, press Ctrl+C to stop.")
	run_start_time = time.time()
	max_duration_seconds = max(1, int(INFERENCE_DURATION_MINUTES * 60))

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
				navigation_logic = NavigationDeterministicLogic(frame_width=frame_bgr.shape[1])
				if NAV_LOGIC_MODE == 1:
					slm_navigation_logic = NavigationSLMAugmentLogic(device=device)

			combined, frame_record, frame_metrics = process_frame(
				frame_idx,
				frame_bgr,
				yolo_model,
				depth_processor,
				depth_model,
				device,
				navigation_logic,
				slm_navigation_logic,
				tts_engine,
				tts_worker,
				tts_state,
			)

			frame_record["frame"] = frame_idx
			all_frames.append(frame_record)

			if video_writer is None:
				fps = cap.get(cv2.CAP_PROP_FPS)
				if fps <= 0:
					fps = 20
				h, w = combined.shape[:2]
				fourcc = cv2.VideoWriter_fourcc(*"mp4v")
				video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

			video_write_start = time.perf_counter()
			video_writer.write(combined)
			video_write_latency_ms = (time.perf_counter() - video_write_start) * 1000.0

			display_latency_ms = 0.0

			if SHOW_WINDOWS and (not DISABLE_DISPLAY_FOR_BENCHMARK):
				display_start = time.perf_counter()
				try:
					cv2.imshow("Depth + BBoxes + Navigation", combined)
					if cv2.waitKey(1) & 0xFF == ord("q"):
						display_latency_ms = (time.perf_counter() - display_start) * 1000.0
						frame_metrics["capture_latency_ms"] = capture_latency_ms
						frame_metrics["video_write_latency_ms"] = video_write_latency_ms
						frame_metrics["display_latency_ms"] = display_latency_ms
						frame_metrics["app_loop_latency_ms"] = (time.perf_counter() - loop_start) * 1000.0
						frame_metrics["fps_instant"] = (
							1000.0 / frame_metrics["app_loop_latency_ms"]
							if frame_metrics["app_loop_latency_ms"] > 0
							else None
						)
						resource_snapshot = profiler.sample_resources(device)
						profiler.log_frame(frame_idx, frame_metrics, resource_snapshot)
						break
				except cv2.error:
					print("OpenCV GUI is not available. Continuing without preview windows.")
				display_latency_ms = (time.perf_counter() - display_start) * 1000.0

			frame_metrics["capture_latency_ms"] = capture_latency_ms
			frame_metrics["video_write_latency_ms"] = video_write_latency_ms
			frame_metrics["display_latency_ms"] = display_latency_ms
			frame_metrics["app_loop_latency_ms"] = (time.perf_counter() - loop_start) * 1000.0
			frame_metrics["fps_instant"] = (
				1000.0 / frame_metrics["app_loop_latency_ms"]
				if frame_metrics["app_loop_latency_ms"] > 0
				else None
			)
			resource_snapshot = None
			if frame_idx % max(1, RESOURCE_SAMPLE_EVERY_N_FRAMES) == 0:
				resource_snapshot = profiler.sample_resources(device)
				frame_metrics["resource_sampled"] = True
			profiler.log_frame(frame_idx, frame_metrics, resource_snapshot)

	except KeyboardInterrupt:
		print("Stopped by user (Ctrl+C).")
	finally:
		if tts_worker is not None:
			tts_worker.stop()
		cap.release()
		if video_writer is not None:
			video_writer.release()
		cv2.destroyAllWindows()

	with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
		json.dump(all_frames, f, indent=2)
	profiler.finalize(OUTPUT_JSON, OUTPUT_VIDEO)

	print(f"Saved bounding boxes to: {OUTPUT_JSON}")
	print(f"Saved output video to: {OUTPUT_VIDEO}")
	print(f"Saved performance logs to: {profiler.run_dir}")


if __name__ == "__main__":
	main()
