import csv
import json
import os
import random
import re
from datetime import datetime

import cv2
import torch

from mechanics.depth_estimation import DepthEstimator
from mechanics.depth_estimation import estimate_distance_from_depth
from mechanics.navigation_logic import NavigationLogic
from mechanics.object_detection import draw_centered_label


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Choose "eval_set" to run on pipeline_evaluations/eval_set, or "homeobjects" for the previous setup.
DATASET_MODE = os.environ.get("NAV_EVAL_DATASET_MODE", "eval_set").strip().lower()

if DATASET_MODE == "homeobjects":
	DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets", "homeobjects-3K")
	IMAGES_DIR = os.path.join(DATASET_ROOT, "images", "train")
	LABELS_DIR = os.path.join(DATASET_ROOT, "labels", "train")
	DATASET_YAML = os.path.join(DATASET_ROOT, "HomeObjects-3K.yaml")
	CLASS_NAMES_JSON = os.path.join(PROJECT_ROOT, "datasets", "classes.json")
	SAMPLE_RANDOM = True
	NUM_IMAGES = 50
elif DATASET_MODE == "eval_set":
	DATASET_ROOT = os.path.join(PROJECT_ROOT, "pipeline_evaluations", "eval_set")
	IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
	LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
	DATASET_YAML = os.path.join(PROJECT_ROOT, "datasets", "homeobjects-3K", "HomeObjects-3K.yaml")
	CLASS_NAMES_JSON = os.path.join(PROJECT_ROOT, "datasets", "classes.json")
	SAMPLE_RANDOM = False
	NUM_IMAGES = None
else:
	raise ValueError(f"Unsupported NAV_EVAL_DATASET_MODE: {DATASET_MODE}. Use 'eval_set' or 'homeobjects'.")

DEPTH_MODEL_FILE = os.path.join(
	PROJECT_ROOT,
	"model_training",
	"depth_estimation",
	"model_weights",
	"depth_anything_v2_metric_hypersim_vits.pth",
)

OUTPUT_ROOT = os.path.join(BASE_DIR, "navigation_eval_outputs")

RANDOM_SEED = 42
ZONE_SHADE_OPACITY = 0.10


def parse_class_names_from_yaml(yaml_path):
	"""Read class names from YOLO yaml 'names:' mapping without extra dependencies."""
	if not os.path.isfile(yaml_path):
		raise FileNotFoundError(f"Dataset yaml not found: {yaml_path}")

	mapping = {}
	in_names = False
	pattern = re.compile(r"^\s*(\d+)\s*:\s*(.+?)\s*$")

	with open(yaml_path, "r", encoding="utf-8") as f:
		for raw_line in f:
			line = raw_line.rstrip("\n")
			stripped = line.strip()

			if not in_names:
				if stripped == "names:":
					in_names = True
				continue

			if not stripped:
				continue

			match = pattern.match(line)
			if match:
				idx = int(match.group(1))
				name = match.group(2).strip().strip("\"").strip("'")
				mapping[idx] = name
				continue

			if re.match(r"^[A-Za-z_][A-Za-z0-9_\-]*\s*:\s*", stripped):
				break

	if not mapping:
		raise ValueError(f"Could not parse class names from: {yaml_path}")

	max_idx = max(mapping)
	return [mapping.get(i, f"class_{i}") for i in range(max_idx + 1)]


def parse_class_names_from_json(json_path):
	if not os.path.isfile(json_path):
		raise FileNotFoundError(f"Class names json not found: {json_path}")

	with open(json_path, "r", encoding="utf-8") as f:
		class_names = json.load(f)

	if not isinstance(class_names, list) or not class_names:
		raise ValueError(f"Invalid class names json content: {json_path}")

	return [str(name) for name in class_names]


def collect_labeled_image_pairs(images_dir, labels_dir):
	if not os.path.isdir(images_dir):
		raise FileNotFoundError(f"Images directory not found: {images_dir}")
	if not os.path.isdir(labels_dir):
		raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

	image_map = {}
	for file_name in os.listdir(images_dir):
		image_path = os.path.join(images_dir, file_name)
		if not os.path.isfile(image_path):
			continue
		stem, _ = os.path.splitext(file_name)
		image_map[stem] = image_path

	pairs = []
	for file_name in os.listdir(labels_dir):
		if not file_name.lower().endswith(".txt"):
			continue
		label_path = os.path.join(labels_dir, file_name)
		stem, _ = os.path.splitext(file_name)
		image_path = image_map.get(stem)
		if image_path:
			pairs.append((image_path, label_path, stem))

	pairs.sort(key=lambda item: item[2])
	return pairs


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
	detections = []

	with open(label_path, "r", encoding="utf-8") as f:
		for raw_line in f:
			line = raw_line.strip()
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
			detections.append(
				{
					"class_id": class_id,
					"class_name": class_name,
					"bbox": bbox,
				}
			)

	return detections


def apply_zone_shading(frame, navigation_logic):
	h, w = frame.shape[:2]
	left_end = int(getattr(navigation_logic, "left_end", int(0.30 * w)))
	center_end = int(getattr(navigation_logic, "center_end", int(0.70 * w)))

	overlay = frame.copy()
	cv2.rectangle(overlay, (0, 0), (left_end, h), (0, 0, 255), -1)
	cv2.rectangle(overlay, (left_end, 0), (center_end, h), (0, 255, 0), -1)
	cv2.rectangle(overlay, (center_end, 0), (w, h), (0, 0, 255), -1)
	cv2.addWeighted(overlay, ZONE_SHADE_OPACITY, frame, 1.0 - ZONE_SHADE_OPACITY, 0, frame)


def draw_detections_with_distance(frame, detections):
	for det in detections:
		x1, y1, x2, y2 = det["bbox"]
		class_name = det["class_name"]
		distance = det.get("distance")

		cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 255), 2)
		cv2.putText(
			frame,
			class_name,
			(x1, max(20, y1 - 8)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.45,
			(80, 220, 255),
			2,
		)

		center_x = (x1 + x2) // 2
		center_y = (y1 + y2) // 2
		if distance is None:
			distance_label = "dist: N/A"
		else:
			distance_label = f"dist: {distance:.2f}m"
		draw_centered_label(frame, distance_label, center_x, center_y)


def ensure_inputs():
	if not os.path.isfile(DEPTH_MODEL_FILE):
		raise FileNotFoundError(f"Depth model file not found: {DEPTH_MODEL_FILE}")
	if DATASET_MODE == "homeobjects" and not os.path.isfile(DATASET_YAML):
		raise FileNotFoundError(f"Dataset yaml not found: {DATASET_YAML}")
	if not os.path.isfile(CLASS_NAMES_JSON):
		raise FileNotFoundError(f"Class names json not found: {CLASS_NAMES_JSON}")


def main():
	ensure_inputs()

	if DATASET_MODE == "homeobjects":
		class_names = parse_class_names_from_yaml(DATASET_YAML)
	else:
		class_names = parse_class_names_from_json(CLASS_NAMES_JSON)

	labeled_pairs = collect_labeled_image_pairs(IMAGES_DIR, LABELS_DIR)
	if not labeled_pairs:
		raise RuntimeError("No matching image-label pairs found in dataset train split.")

	if SAMPLE_RANDOM:
		rng = random.Random(RANDOM_SEED)
		sample_size = min(NUM_IMAGES, len(labeled_pairs))
		sampled_pairs = rng.sample(labeled_pairs, sample_size)
	else:
		sampled_pairs = labeled_pairs
		sample_size = len(sampled_pairs)

	run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
	run_dir = os.path.join(OUTPUT_ROOT, run_name)
	annotated_dir = os.path.join(run_dir, "annotated_images")
	os.makedirs(annotated_dir, exist_ok=True)

	csv_path = os.path.join(run_dir, "navigation_review.csv")

	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using device: {device}")
	print(f"Dataset mode: {DATASET_MODE}")
	print(f"Images dir: {IMAGES_DIR}")
	print(f"Labels dir: {LABELS_DIR}")
	print(f"Depth model: {DEPTH_MODEL_FILE}")
	print(f"Sample size: {sample_size}")

	depth_estimator = DepthEstimator(DEPTH_MODEL_FILE, device=device)
	depth_estimator.load_model()

	with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
		fieldnames = [
			"sample_id",
			"image_stem",
			"image_path",
			"label_path",
			"output_path",
			"detections_count",
			"left_risk",
			"center_risk",
			"right_risk",
			"nav_command",
			"manual_correct",
			"manual_notes",
		]
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()

		for idx, (image_path, label_path, stem) in enumerate(sampled_pairs, start=1):
			frame_bgr = cv2.imread(image_path)
			if frame_bgr is None:
				print(f"[{idx}/{sample_size}] Skipping unreadable image: {image_path}")
				continue

			h, w = frame_bgr.shape[:2]
			detections = parse_label_file(label_path, w, h, class_names)
			frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
			depth_float, _ = depth_estimator.predict(frame_rgb)

			nav_inputs = []
			for det in detections:
				bbox = det["bbox"]
				depth_value, distance_value = estimate_distance_from_depth(depth_float, bbox)
				det["depth"] = depth_value
				det["distance"] = distance_value

				nav_inputs.append(
					{
						"class": det["class_name"],
						"bbox": bbox,
						"depth": depth_value,
						"distance": distance_value,
					}
				)

			navigation_logic = NavigationLogic(frame_width=w)
			zone_risks, nav_command = navigation_logic.process_detections(nav_inputs)

			vis = frame_bgr.copy()
			apply_zone_shading(vis, navigation_logic)
			draw_detections_with_distance(vis, detections)
			navigation_logic.draw_overlays(vis, zone_risks, nav_command)

			output_name = f"{idx:03d}_{stem}.jpg"
			output_path = os.path.join(annotated_dir, output_name)
			cv2.imwrite(output_path, vis)

			writer.writerow(
				{
					"sample_id": idx,
					"image_stem": stem,
					"image_path": image_path,
					"label_path": label_path,
					"output_path": output_path,
					"detections_count": len(detections),
					"left_risk": f"{zone_risks['left']:.4f}",
					"center_risk": f"{zone_risks['center']:.4f}",
					"right_risk": f"{zone_risks['right']:.4f}",
					"nav_command": nav_command,
					"manual_correct": "",
					"manual_notes": "",
				}
			)

			print(
				f"[{idx}/{sample_size}] Saved: {output_name} | "
				f"risks(L/C/R)=({zone_risks['left']:.2f}/{zone_risks['center']:.2f}/{zone_risks['right']:.2f}) | "
				f"decision={nav_command}"
			)

	print("Navigation evaluation export complete.")
	print(f"Output folder: {run_dir}")
	print(f"Review CSV: {csv_path}")


if __name__ == "__main__":
	main()