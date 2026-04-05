import cv2
from ultralytics import YOLO


class ObjectDetector:
	def __init__(self, weights_path):
		self.weights_path = weights_path
		self.model = None

	def load_model(self):
		import os
		if not os.path.exists(self.weights_path):
			print(f"Warning: Custom weights not found at {self.weights_path}. Falling back to default yolov8n.pt")
			self.model = YOLO("yolov8n.pt")
		else:
			self.model = YOLO(self.weights_path)
		return self.model

	def predict(self, frame_bgr):
		if self.model is None:
			self.load_model()

		result = self.model(frame_bgr, verbose=False)[0]
		detections = []

		for box in result.boxes:
			x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
			conf = float(box.conf[0])
			cls_id = int(box.cls[0])
			cls_name = self.model.names[cls_id]

			detections.append(
				{
					"class_id": cls_id,
					"class_name": cls_name,
					"confidence": conf,
					"x1": x1,
					"y1": y1,
					"x2": x2,
					"y2": y2,
				}
			)

		return result, detections


def draw_centered_label(
	frame,
	text,
	center_x,
	center_y,
	font_scale=0.4,
	text_color=(255, 255, 255),
	bg_color=(0, 0, 0),
	padding=4,
):
	"""Draw text centered around a point with a filled background for readability."""
	font = cv2.FONT_HERSHEY_SIMPLEX
	thickness = 1
	(text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

	x1 = int(center_x - text_w / 2 - padding)
	y1 = int(center_y - text_h / 2 - padding)
	x2 = int(center_x + text_w / 2 + padding)
	y2 = int(center_y + text_h / 2 + padding + baseline)

	h, w = frame.shape[:2]
	x1 = max(0, x1)
	y1 = max(0, y1)
	x2 = min(w - 1, x2)
	y2 = min(h - 1, y2)

	cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
	text_x = int(center_x - text_w / 2)
	text_y = int(center_y + text_h / 2)
	text_x = max(0, min(w - text_w, text_x))
	text_y = max(text_h, min(h - baseline, text_y))
	cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)
