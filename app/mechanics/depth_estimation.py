import os
import sys

import cv2
import numpy as np
import torch

def get_base_root():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_PATH = os.path.join(get_base_root(), "Depth-Anything-V2")
METRIC_PATH = os.path.join(REPO_PATH, "metric_depth")

if os.path.exists(METRIC_PATH) and METRIC_PATH not in sys.path:
    sys.path.insert(0, METRIC_PATH)
elif REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

from depth_anything_v2.dpt import DepthAnythingV2


class DepthEstimator:
    def __init__(self, model_dir, device="cpu"):
        self.model_dir = model_dir
        self.device = device
        self.model = None

        self.model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        filename = os.path.basename(model_dir).lower()
        self.encoder = "vitl"
        if "vits" in filename:
            self.encoder = "vits"
        elif "vitb" in filename:
            self.encoder = "vitb"

        self.dataset = "vkitti" if "vkitti" in filename else "hypersim"
        self.max_depth = 80 if self.dataset == 'vkitti' else 20

    def load_model(self):
        print(f"Loading Depth-Anything-V2 ({self.encoder}, {self.dataset}) from {self.model_dir}...")
        config = self.model_configs[self.encoder]

        try:
            self.model = DepthAnythingV2(**{**config, "max_depth": self.max_depth})
        except TypeError:
            self.model = DepthAnythingV2(**config)
            self.model.max_depth = self.max_depth

        state_dict = torch.load(self.model_dir, map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        return self.model

    def predict(self, frame_rgb):
        if self.model is None:
            self.load_model()
        with torch.no_grad():
            depth_float = self.model.infer_image(frame_rgb)
            
        depth_clipped = np.clip(depth_float, 0, self.max_depth)
        depth_u8 = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

        return depth_float, depth_color


def estimate_distance_from_depth(depth_map, bbox):
    """Estimate object distance from a bounding-box region in the depth map."""
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

    depth_value = float(np.median(valid))
    if not np.isfinite(depth_value):
        return None, None

    return depth_value, depth_value


def scan_depth_hazards(
    depth_map,
    danger_threshold_m=1.2,
    warning_threshold_m=2.0,
    return_masks=True,
    return_coords=True,
):
    """Scan full depth map for warning/danger pixels and return summaries.

    Levels:
    - danger: depth <= danger_threshold_m
    - warning: danger_threshold_m < depth <= warning_threshold_m
    """
    if depth_map is None:
        return {
            "danger_threshold_m": float(danger_threshold_m),
            "warning_threshold_m": float(warning_threshold_m),
            "danger_mask": None,
            "warning_mask": None,
            "danger_coords_xy": np.empty((0, 2), dtype=np.int32),
            "warning_coords_xy": np.empty((0, 2), dtype=np.int32),
            "zone_summary": {
                "left": {},
                "center": {},
                "right": {},
            },
            "global_summary": {
                "valid_pixel_count": 0,
                "danger_pixel_count": 0,
                "warning_pixel_count": 0,
                "near_pixel_count": 0,
                "danger_ratio": 0.0,
                "warning_ratio": 0.0,
                "near_ratio": 0.0,
                "min_depth_m": None,
                "median_depth_m": None,
            },
        }

    if warning_threshold_m < danger_threshold_m:
        warning_threshold_m = danger_threshold_m

    h, w = depth_map.shape[:2]
    left_end = int(0.30 * w)
    center_end = int(0.70 * w)

    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    danger_mask = valid_mask & (depth_map <= float(danger_threshold_m))
    warning_mask = (
        valid_mask
        & (depth_map > float(danger_threshold_m))
        & (depth_map <= float(warning_threshold_m))
    )
    near_mask = danger_mask | warning_mask

    if return_coords:
        dy, dx = np.where(danger_mask)
        wy, wx = np.where(warning_mask)
        danger_coords_xy = np.column_stack((dx, dy)).astype(np.int32) if dx.size else np.empty((0, 2), dtype=np.int32)
        warning_coords_xy = np.column_stack((wx, wy)).astype(np.int32) if wx.size else np.empty((0, 2), dtype=np.int32)
    else:
        danger_coords_xy = np.empty((0, 2), dtype=np.int32)
        warning_coords_xy = np.empty((0, 2), dtype=np.int32)

    if not return_masks:
        danger_mask_out = None
        warning_mask_out = None
    else:
        danger_mask_out = danger_mask
        warning_mask_out = warning_mask

    valid_count = int(np.sum(valid_mask))
    danger_count = int(np.sum(danger_mask))
    warning_count = int(np.sum(warning_mask))
    near_count = int(np.sum(near_mask))

    near_values = depth_map[near_mask]
    if near_values.size:
        min_depth = float(np.min(near_values))
        median_depth = float(np.median(near_values))
    else:
        min_depth = None
        median_depth = None

    x_coords = np.arange(w)
    left_x = x_coords < left_end
    center_x = (x_coords >= left_end) & (x_coords < center_end)
    right_x = x_coords >= center_end

    zone_map = {
        "left": left_x,
        "center": center_x,
        "right": right_x,
    }

    zone_summary = {}
    for zone_name, zone_x_mask in zone_map.items():
        zone_valid_mask = valid_mask[:, zone_x_mask]
        zone_danger_mask = danger_mask[:, zone_x_mask]
        zone_warning_mask = warning_mask[:, zone_x_mask]
        zone_near_mask = near_mask[:, zone_x_mask]

        zone_valid_count = int(np.sum(zone_valid_mask))
        zone_danger_count = int(np.sum(zone_danger_mask))
        zone_warning_count = int(np.sum(zone_warning_mask))
        zone_near_count = int(np.sum(zone_near_mask))

        if zone_near_count > 0:
            zone_near_values = depth_map[:, zone_x_mask][zone_near_mask]
            zone_min_depth = float(np.min(zone_near_values))
            zone_median_depth = float(np.median(zone_near_values))
        else:
            zone_min_depth = None
            zone_median_depth = None

        if zone_valid_count > 0:
            zone_danger_ratio = zone_danger_count / zone_valid_count
            zone_warning_ratio = zone_warning_count / zone_valid_count
            zone_near_ratio = zone_near_count / zone_valid_count
        else:
            zone_danger_ratio = 0.0
            zone_warning_ratio = 0.0
            zone_near_ratio = 0.0

        zone_summary[zone_name] = {
            "valid_pixel_count": zone_valid_count,
            "danger_pixel_count": zone_danger_count,
            "warning_pixel_count": zone_warning_count,
            "near_pixel_count": zone_near_count,
            "danger_ratio": float(zone_danger_ratio),
            "warning_ratio": float(zone_warning_ratio),
            "near_ratio": float(zone_near_ratio),
            "min_depth_m": zone_min_depth,
            "median_depth_m": zone_median_depth,
        }

    if valid_count > 0:
        danger_ratio = danger_count / valid_count
        warning_ratio = warning_count / valid_count
        near_ratio = near_count / valid_count
    else:
        danger_ratio = 0.0
        warning_ratio = 0.0
        near_ratio = 0.0

    return {
        "danger_threshold_m": float(danger_threshold_m),
        "warning_threshold_m": float(warning_threshold_m),
        "danger_mask": danger_mask_out,
        "warning_mask": warning_mask_out,
        "danger_coords_xy": danger_coords_xy,
        "warning_coords_xy": warning_coords_xy,
        "zone_summary": zone_summary,
        "global_summary": {
            "valid_pixel_count": valid_count,
            "danger_pixel_count": danger_count,
            "warning_pixel_count": warning_count,
            "near_pixel_count": near_count,
            "danger_ratio": float(danger_ratio),
            "warning_ratio": float(warning_ratio),
            "near_ratio": float(near_ratio),
            "min_depth_m": min_depth,
            "median_depth_m": median_depth,
        },
    }