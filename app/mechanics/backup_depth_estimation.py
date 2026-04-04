import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthEstimator:
	def __init__(self, model_dir, device="cpu"):
		self.model_dir = model_dir
		self.device = device
		self.processor = None
		self.model = None

	def load_model(self):
		self.processor = AutoImageProcessor.from_pretrained(self.model_dir, local_files_only=True)
		self.model = AutoModelForDepthEstimation.from_pretrained(
			self.model_dir,
			local_files_only=True,
		).to(self.device)
		self.model.eval()
		return self.processor, self.model

	def predict(self, frame_rgb):
		if self.processor is None or self.model is None:
			self.load_model()

		depth_inputs = self.processor(images=frame_rgb, return_tensors="pt").to(self.device)
		with torch.no_grad():
			depth_pred = self.model(**depth_inputs).predicted_depth

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


def estimate_distance_from_depth(depth_map, bbox):
	"""Estimate object depth from a bounding-box region in the depth map."""
	if depth_map is None or bbox is None or len(bbox) != 4:
		return None, None

	h, w = depth_map.shape[:2]
	x1, y1, x2, y2 = [int(v) for v in bbox]
	x1 = max(0, min(w - 1, x1))
	y1 = max(0, min(h - 1, y1))
	x2 = max(0, min(w, x2))
	y2 = max(0, min(h, y2))

	if x2 <= x1 or y2 <= y1:
		return None, None

	depth_roi = depth_map[y1:y2, x1:x2]
	valid = depth_roi[np.isfinite(depth_roi)]
	if valid.size == 0:
		return None, None

	# Median is robust to noisy pixels in monocular depth outputs.
	depth_value = float(np.median(valid))
	if not np.isfinite(depth_value):
		return None, None

	# Current pipeline treats this as a relative distance signal.
	return depth_value, depth_value
