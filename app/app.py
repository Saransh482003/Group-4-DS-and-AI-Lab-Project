import json
import os

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

print(torch.__version__)
print(torch.version.cuda)

# Hardcoded settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS = os.path.join(BASE_DIR, "YOLOv8n-uni.pt")
VIDEO_SOURCE = 1  # 0 for webcam, or replace with a video file path string
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
OUTPUT_JSON = os.path.join(BASE_DIR, "bbox_output.json")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "yolo_depth_output.mp4")
SHOW_WINDOWS = True


if not os.path.exists(YOLO_WEIGHTS):
	raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Load models once
yolo_model = YOLO(YOLO_WEIGHTS)
depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device)
depth_model.eval()


cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
	raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")


all_bounding_boxes = []
frame_idx = 0
video_writer = None

print("Press 'q' to stop.")
print("If preview windows are unavailable, press Ctrl+C to stop.")

try:
	while True:
		ok, frame_bgr = cap.read()
		if not ok:
			break

		frame_idx += 1
		frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

		# YOLO inference
		yolo_result = yolo_model(frame_bgr, verbose=False)[0]
		frame_boxes = []

		for box in yolo_result.boxes:
			x1, y1, x2, y2 = box.xyxy[0].tolist()
			conf = float(box.conf[0])
			cls_id = int(box.cls[0])
			cls_name = yolo_model.names[cls_id]

			frame_boxes.append(
				{
					"class_id": cls_id,
					"class_name": cls_name,
					"confidence": conf,
					"x1": float(x1),
					"y1": float(y1),
					"x2": float(x2),
					"y2": float(y2),
				}
			)

			cv2.rectangle(
				frame_bgr,
				(int(x1), int(y1)),
				(int(x2), int(y2)),
				(0, 255, 0),
				2,
			)
			label = f"{cls_name} {conf:.2f}"
			cv2.putText(
				frame_bgr,
				label,
				(int(x1), max(20, int(y1) - 8)),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.6,
				(0, 255, 0),
				2,
			)

		all_bounding_boxes.append({"frame": frame_idx, "detections": frame_boxes})

		# DepthAnythingV2 inference
		depth_inputs = depth_processor(images=frame_rgb, return_tensors="pt").to(device)
		with torch.no_grad():
			depth_pred = depth_model(**depth_inputs).predicted_depth

		depth_pred = torch.nn.functional.interpolate(
			depth_pred.unsqueeze(1),
			size=frame_rgb.shape[:2],
			mode="bicubic",
			align_corners=False,
		).squeeze()

		depth_map = depth_pred.cpu().numpy()
		depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
		depth_map = depth_map.astype(np.uint8)
		depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

		combined_frame = np.hstack((frame_bgr, depth_color))

		if video_writer is None:
			fps = cap.get(cv2.CAP_PROP_FPS)
			if fps <= 0:
				fps = 20
			h, w = combined_frame.shape[:2]
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

		video_writer.write(combined_frame)

		if SHOW_WINDOWS:
			try:
				cv2.imshow("YOLO Detections", frame_bgr)
				cv2.imshow("DepthAnythingV2 Depth Map", depth_color)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
			except cv2.error:
				SHOW_WINDOWS = False
				print("OpenCV GUI is not available. Continuing without preview windows.")

except KeyboardInterrupt:
	print("Stopped by user (Ctrl+C).")


cap.release()
if video_writer is not None:
	video_writer.release()
cv2.destroyAllWindows()

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
	json.dump(all_bounding_boxes, f, indent=2)

print(f"Saved bounding boxes to: {OUTPUT_JSON}")
print(f"Saved output video to: {OUTPUT_VIDEO}")
