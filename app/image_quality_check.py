from typing import Dict, Tuple

import cv2
import numpy as np


def is_usable(image: np.ndarray) -> Tuple[bool, Dict[str, float]]:
	"""Check if an RGB image is usable for navigation inference.

	The function evaluates brightness, contrast, and sharpness (Laplacian variance).
	Low contrast plus low sharpness is treated as a likely dirty/smudged lens condition.
	"""
	if image is None or image.size == 0:
		return False, {"error": "empty_image"}

	if image.ndim != 3 or image.shape[2] != 3:
		return False, {"error": "expected_rgb_image"}

	# Tuned for quick camera pre-flight checks on resized RGB frames.
	darkness_threshold = 70.0
	brightness_threshold = 200.0
	laplacian_threshold = 150.0
	contrast_threshold = 45.0

	image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	analysis: Dict[str, float] = {}

	analysis["brightness"] = float(image_gray.mean())
	analysis["contrast"] = float(image_gray.std())
	analysis["laplacian"] = float(cv2.Laplacian(image_gray, cv2.CV_64F).var())

	too_dark_or_bright = (
		analysis["brightness"] < darkness_threshold
		or analysis["brightness"] > brightness_threshold
	)
	low_contrast = analysis["contrast"] < contrast_threshold
	blurry = analysis["laplacian"] < laplacian_threshold

	analysis["likely_dirty_or_smudged"] = float(low_contrast and blurry)

	is_ok = not (too_dark_or_bright or low_contrast or blurry)
	return is_ok, analysis


def is_Usable(image: np.ndarray) -> Tuple[bool, Dict[str, float]]:
	"""Backward-compatible alias for teammates using the original function name."""
	return is_usable(image)
