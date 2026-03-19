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
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

# Put all your image paths here
IMAGE_PATHS = [
	r"E:\dsai_group4_project\datasets\dsai-unified-dataset\images\train\IMG_000056.jpg",
	# r"E:\dsai_group4_project\datasets\dsai-unified-dataset\images\val\IMG_000056.jpg",
	# r"E:\path\to\another_image.png",
]

OUTPUT_DIR = os.path.join(BASE_DIR, "image_outputs")
OUTPUT_JSON = os.path.join(BASE_DIR, "image_bbox_output.json")


if not os.path.exists(YOLO_WEIGHTS):
	raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")

if len(IMAGE_PATHS) == 0:
	raise ValueError("IMAGE_PATHS is empty. Add at least one image path.")

os.makedirs(OUTPUT_DIR, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Load models once
yolo_model = YOLO(YOLO_WEIGHTS)
depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device)
depth_model.eval()


all_bounding_boxes = []
processed_count = 0

for image_path in IMAGE_PATHS:
	if not os.path.exists(image_path):
		print(f"Skipping missing image: {image_path}")
		continue

	frame_bgr = cv2.imread(image_path)
	if frame_bgr is None:
		print(f"Skipping unreadable image: {image_path}")
		continue

	processed_count += 1

	frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

	# YOLO inference
	yolo_result = yolo_model(frame_bgr, verbose=False)[0]
	image_boxes = []

	for box in yolo_result.boxes:
		x1, y1, x2, y2 = box.xyxy[0].tolist()
		conf = float(box.conf[0])
		cls_id = int(box.cls[0])
		cls_name = yolo_model.names[cls_id]

		image_boxes.append(
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

	all_bounding_boxes.append(
		{
			"image_path": image_path,
			"detections": image_boxes,
		}
	)

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
	depth_with_boxes = depth_color.copy()

	# DepthAnythingV2 gives relative depth, not absolute metric meters.
	# We estimate each object's depth from the median depth value inside its bbox.
	depth_float = depth_pred.cpu().numpy().astype(np.float32)
	print(f"\nImage: {os.path.basename(image_path)}")
	print("Object distances from camera (relative scale):")

	for det in image_boxes:
		h, w = depth_float.shape
		x1 = max(0, min(w - 1, int(det["x1"])))
		y1 = max(0, min(h - 1, int(det["y1"])))
		x2 = max(0, min(w - 1, int(det["x2"])))
		y2 = max(0, min(h - 1, int(det["y2"])))

		if x2 <= x1 or y2 <= y1:
			det["depth_relative"] = None
			det["distance_relative"] = None
			print(f"- {det['class_name']}: bbox invalid")
			continue

		crop = depth_float[y1:y2, x1:x2]
		if crop.size == 0:
			det["depth_relative"] = None
			det["distance_relative"] = None
			print(f"- {det['class_name']}: no depth pixels")
			continue

		object_depth = float(np.median(crop))
		# Convert relative depth to a relative distance-like value.
		# For DepthAnythingV2, depth values are inverse-like; lower depth usually means farther.
		relative_distance = float(1.0 / (object_depth + 1e-6))

		det["depth_relative"] = object_depth
		det["distance_relative"] = relative_distance

		print(
			f"- {det['class_name']}: depth={object_depth:.4f}, distance={relative_distance:.4f}"
		)

	for det in image_boxes:
		x1 = int(det["x1"])
		y1 = int(det["y1"])
		x2 = int(det["x2"])
		y2 = int(det["y2"])
		label = f"{det['class_name']} {det['confidence']:.2f}"

		cv2.rectangle(
			depth_with_boxes,
			(x1, y1),
			(x2, y2),
			(0, 255, 0),
			2,
		)
		cv2.putText(
			depth_with_boxes,
			label,
			(x1, max(20, y1 - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 255, 0),
			2,
		)

	combined = np.hstack((frame_bgr, depth_color))

	base_name = os.path.splitext(os.path.basename(image_path))[0]
	output_path = os.path.join(OUTPUT_DIR, f"{base_name}_yolo_depth.jpg")
	depth_bbox_output_path = os.path.join(OUTPUT_DIR, f"{base_name}_depth_with_boxes.jpg")
	cv2.imwrite(output_path, combined)
	cv2.imwrite(depth_bbox_output_path, depth_with_boxes)
	print(f"Saved: {output_path}")
	print(f"Saved: {depth_bbox_output_path}")


if processed_count == 0:
	raise RuntimeError("No valid images were processed. Check IMAGE_PATHS.")


with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
	json.dump(all_bounding_boxes, f, indent=2)

print(f"Saved bounding boxes to: {OUTPUT_JSON}")
