import json
import os
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

OUTPUT_JSON = os.path.join(BASE_DIR, "bbox_output.json")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "yolo_depth_output.mp4")

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


def validate_inputs():
	if not os.path.exists(YOLO_WEIGHTS):
		raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
	if not os.path.isdir(DEPTH_MODEL_DIR):
		raise FileNotFoundError(f"Depth model directory not found: {DEPTH_MODEL_DIR}")
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
	frame_bgr,
	yolo_model,
	depth_processor,
	depth_model,
	device,
	navigation_logic,
	slm_navigation_logic,
	tts_engine,
	tts_state,
):
	frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
	yolo_result = yolo_model(frame_bgr, verbose=False)[0]

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

	depth_float, depth_color = run_depth_inference(frame_rgb, depth_processor, depth_model, device)
	depth_with_boxes = depth_color.copy()

	for det in frame_boxes:
		bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]
		object_depth, relative_distance = estimate_distance_from_depth(depth_float, bbox)
		det["depth_relative"] = object_depth
		det["distance_relative"] = relative_distance

		x1, y1, x2, y2 = bbox
		center_x = (x1 + x2) // 2
		center_y = (y1 + y2) // 2
		distance_label = "dist: N/A" if relative_distance is None else f"dist: {relative_distance:.3f}"

		cv2.rectangle(depth_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(
			depth_with_boxes,
			f"{det['class_name']} {det['confidence']:.2f}",
			(x1, max(20, y1 - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.4,
			(0, 255, 0),
			2,
		)

		draw_centered_label(depth_with_boxes, distance_label, center_x, center_y)

	nav_detections = [
		{
			"class": det["class_name"],
			"bbox": [det["x1"], det["y1"], det["x2"], det["y2"]],
			"depth": det.get("depth_relative"),
			"distance": det.get("distance_relative"),
		}
		for det in frame_boxes
	]

	zone_risks, nav_command = navigation_logic.process_detections(nav_detections)
	if NAV_LOGIC_MODE == 1:
		slm_result = slm_navigation_logic.decide_instruction(
			zone_risks,
			nav_detections,
			frame_width=frame_bgr.shape[1],
		)
		nav_command = slm_result["instruction"]

	apply_zone_shading(depth_with_boxes, navigation_logic)
	navigation_logic.draw_overlays(depth_with_boxes, zone_risks, nav_command)

	if tts_engine is not None:
		now = time.time()
		should_speak = (not TTS_SPEAK_ONCE_PER_EXECUTION) or (not tts_state["spoken"])
		if should_speak and TTS_SPEAK_ON_COMMAND_CHANGE:
			if tts_state.get("last_command") == nav_command:
				should_speak = False
		if should_speak and (now - tts_state.get("last_spoken_at", 0.0)) < TTS_MIN_INTERVAL_SECONDS:
			should_speak = False

		if should_speak:
			try:
				if TTS_DIRECT_PLAYBACK:
					playback_mode = tts_engine.speak_direct_safe(
						nav_command,
						fallback_play_audio=True,
					)
					print(f"TTS Output: {playback_mode}")
				else:
					wav_path = tts_engine.synthesize(nav_command, play_audio=TTS_PLAY_AUDIO)
					print(f"TTS Output: {wav_path}")
				tts_state["spoken"] = True
				tts_state["last_command"] = nav_command
				tts_state["last_spoken_at"] = now
			except Exception as exc:
				print(f"TTS error: {exc}")

	combined_frame = depth_with_boxes
	frame_record = {
		"detections": frame_boxes,
		"zone_risks": zone_risks,
		"navigation_command": nav_command,
	}
	return combined_frame, frame_record


def main():
	validate_inputs()
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")
	print(f"Navigation logic mode: {NAV_LOGIC_MODE} ({'deterministic' if NAV_LOGIC_MODE == 0 else 'slm'})")

	yolo_model, depth_processor, depth_model = load_models(device)

	cap = cv2.VideoCapture(VIDEO_SOURCE)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

	navigation_logic = None
	slm_navigation_logic = None
	tts_engine = None
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

	print("Press 'q' to stop.")
	print("If preview windows are unavailable, press Ctrl+C to stop.")

	try:
		while True:
			ok, frame_bgr = cap.read()
			if not ok:
				break

			frame_idx += 1

			if navigation_logic is None:
				navigation_logic = NavigationDeterministicLogic(frame_width=frame_bgr.shape[1])
				if NAV_LOGIC_MODE == 1:
					slm_navigation_logic = NavigationSLMAugmentLogic(device=device)

			combined, frame_record = process_frame(
				frame_bgr,
				yolo_model,
				depth_processor,
				depth_model,
				device,
				navigation_logic,
				slm_navigation_logic,
				tts_engine,
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

			video_writer.write(combined)

			if SHOW_WINDOWS:
				try:
					cv2.imshow("Depth + BBoxes + Navigation", combined)
					if cv2.waitKey(1) & 0xFF == ord("q"):
						break
				except cv2.error:
					print("OpenCV GUI is not available. Continuing without preview windows.")

	except KeyboardInterrupt:
		print("Stopped by user (Ctrl+C).")
	finally:
		cap.release()
		if video_writer is not None:
			video_writer.release()
		cv2.destroyAllWindows()

	with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
		json.dump(all_frames, f, indent=2)

	print(f"Saved bounding boxes to: {OUTPUT_JSON}")
	print(f"Saved output video to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
	main()
