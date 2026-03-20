import json
import os

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO
from navigation_logic import NavigationLogic


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS = os.path.join(BASE_DIR, "YOLOv8n-uni.pt")

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
IMAGE_PATHS = [
	r"E:\dsai_group4_project\datasets\dsai-unified-dataset\images\train\IMG_000056.jpg",
	# r"E:\dsai_group4_project\datasets\dsai-unified-dataset\images\val\IMG_000056.jpg",
	# r"E:\path\to\another_image.png",
]
OUTPUT_DIR = os.path.join(BASE_DIR, "image_outputs")
OUTPUT_JSON = os.path.join(BASE_DIR, "image_bbox_output.json")


def clamp_bbox(x1, y1, x2, y2, width, height):
	x1 = max(0, min(width - 1, int(x1)))
	y1 = max(0, min(height - 1, int(y1)))
	x2 = max(0, min(width - 1, int(x2)))
	y2 = max(0, min(height - 1, int(y2)))
	return x1, y1, x2, y2


def draw_centered_label(canvas, text, center_x, center_y):
	(text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
	text_x = max(0, min(canvas.shape[1] - text_w, center_x - text_w // 2))
	text_y = max(text_h + baseline, min(canvas.shape[0] - baseline, center_y + text_h // 2))

	cv2.rectangle(
		canvas,
		(text_x - 4, text_y - text_h - 4),
		(text_x + text_w + 4, text_y + baseline + 4),
		(0, 0, 0),
		-1,
	)
	cv2.putText(
		canvas,
		text,
		(text_x, text_y),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.55,
		(0, 255, 255),
		2,
	)


def estimate_distance_from_depth(depth_float, bbox):
	h, w = depth_float.shape
	x1, y1, x2, y2 = clamp_bbox(*bbox, width=w, height=h)
	if x2 <= x1 or y2 <= y1:
		return None, None

	crop = depth_float[y1:y2, x1:x2]
	if crop.size == 0:
		return None, None

	object_depth = float(np.median(crop))
	relative_distance = float(1.0 / (object_depth + 1e-6))
	return object_depth, relative_distance


def validate_inputs():
	if not os.path.exists(YOLO_WEIGHTS):
		raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")
	if not IMAGE_PATHS:
		raise ValueError("IMAGE_PATHS is empty. Add at least one image path.")
	os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_models(device):
	yolo_model = YOLO(YOLO_WEIGHTS)
	depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
	depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(device)
	depth_model.eval()
	return yolo_model, depth_processor, depth_model


def run_depth_inference(frame_rgb, depth_processor, depth_model, device):
	depth_inputs = depth_processor(images=frame_rgb, return_tensors="pt").to(device)
	with torch.no_grad():
		depth_pred = depth_model(**depth_inputs).predicted_depth

	depth_pred = torch.nn.functional.interpolate(
		depth_pred.unsqueeze(1),
		size=frame_rgb.shape[:2],
		mode="bicubic",
		align_corners=False,
	).squeeze()

	depth_float = depth_pred.cpu().numpy().astype(np.float32)
	depth_u8 = cv2.normalize(depth_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
	return depth_float, depth_color


def process_image(image_path, yolo_model, depth_processor, depth_model, device):
	frame_bgr = cv2.imread(image_path)
	if frame_bgr is None:
		print(f"Skipping unreadable image: {image_path}")
		return None

	frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
	yolo_result = yolo_model(frame_bgr, verbose=False)[0]

	image_boxes = []
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
		image_boxes.append(detection)

		cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(
			frame_bgr,
			f"{cls_name} {conf:.2f}",
			(x1, max(20, y1 - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 255, 0),
			2,
		)

	depth_float, depth_color = run_depth_inference(frame_rgb, depth_processor, depth_model, device)
	depth_with_boxes = depth_color.copy()

	print(f"\nImage: {os.path.basename(image_path)}")
	print("Detections (class + bbox):")
	for det in image_boxes:
		print(f"- {det['class_name']}: bbox=({det['x1']}, {det['y1']}, {det['x2']}, {det['y2']})")

	print("Object distances from camera (relative scale):")
	for det in image_boxes:
		bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]
		object_depth, relative_distance = estimate_distance_from_depth(depth_float, bbox)
		det["depth_relative"] = object_depth
		det["distance_relative"] = relative_distance

		if object_depth is None:
			print(f"- {det['class_name']}: invalid bbox/depth")
		else:
			print(f"- {det['class_name']}: depth={object_depth:.4f}, distance={relative_distance:.4f}")

		x1, y1, x2, y2 = bbox
		center_x = (x1 + x2) // 2
		center_y = (y1 + y2) // 2
		distance_label = (
			"dist: N/A" if relative_distance is None else f"dist: {relative_distance:.3f}"
		)

		cv2.rectangle(depth_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(
			depth_with_boxes,
			f"{det['class_name']} {det['confidence']:.2f}",
			(x1, max(20, y1 - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 255, 0),
			2,
		)

		draw_centered_label(frame_bgr, distance_label, center_x, center_y)
		draw_centered_label(depth_with_boxes, distance_label, center_x, center_y)

	nav_detections = [
		{
			"class": det["class_name"],
			"bbox": [det["x1"], det["y1"], det["x2"], det["y2"]],
			"distance": det.get("distance_relative"),
		}
		for det in image_boxes
	]

	navigation_logic = NavigationLogic(frame_width=frame_bgr.shape[1])
	zone_risks, nav_command = navigation_logic.process_detections(nav_detections)
	navigation_logic.draw_overlays(frame_bgr, zone_risks, nav_command)

	print(f"Navigation: L={zone_risks['left']:.2f}, C={zone_risks['center']:.2f}, R={zone_risks['right']:.2f}")
	print(f"Navigation Command: {nav_command}")

	base_name = os.path.splitext(os.path.basename(image_path))[0]
	combined = np.hstack((frame_bgr, depth_color))
	output_path = os.path.join(OUTPUT_DIR, f"{base_name}_yolo_depth.jpg")
	depth_bbox_output_path = os.path.join(OUTPUT_DIR, f"{base_name}_depth_with_boxes.jpg")

	cv2.imwrite(output_path, combined)
	cv2.imwrite(depth_bbox_output_path, depth_with_boxes)
	print(f"Saved: {output_path}")
	print(f"Saved: {depth_bbox_output_path}")

	return {
		"image_path": image_path,
		"detections": image_boxes,
	}


def main():
	validate_inputs()
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")

	yolo_model, depth_processor, depth_model = load_models(device)

	all_bounding_boxes = []
	processed_count = 0
	for image_path in IMAGE_PATHS:
		if not os.path.exists(image_path):
			print(f"Skipping missing image: {image_path}")
			continue

		result = process_image(image_path, yolo_model, depth_processor, depth_model, device)
		if result is None:
			continue

		all_bounding_boxes.append(result)
		processed_count += 1

	if processed_count == 0:
		raise RuntimeError("No valid images were processed. Check IMAGE_PATHS.")

	with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
		json.dump(all_bounding_boxes, f, indent=2)
	print(f"Saved bounding boxes to: {OUTPUT_JSON}")


if __name__ == "__main__":
	main()
