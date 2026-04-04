import cv2
import numpy as np
import torch
import sys
import os

# 1. Point Python to the cloned GitHub repository
REPO_PATH = r"E:\dsai_group4_project\Depth-Anything-V2"

# The metric depth logic has its own specialized files inside the 'metric_depth' folder.
# We insert it at the front of sys.path (index 0) so Python prioritizes it over the base model.
METRIC_PATH = os.path.join(REPO_PATH, "metric_depth")
if os.path.exists(METRIC_PATH) and METRIC_PATH not in sys.path:
    sys.path.insert(0, METRIC_PATH)
elif REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

from depth_anything_v2.dpt import DepthAnythingV2

class DepthEstimator:
    def __init__(self, model_dir, device="cpu"):
        # model_dir is expected to be the full path to the .pth file
        self.model_dir = model_dir
        self.device = device
        self.model = None
        
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }

        # Auto-detect encoder and dataset from the filename to prevent hardcoding errors
        filename = os.path.basename(model_dir).lower()
        self.encoder = 'vitl'
        if 'vits' in filename: self.encoder = 'vits'
        elif 'vitb' in filename: self.encoder = 'vitb'
        
        self.dataset = 'vkitti' if 'vkitti' in filename else 'hypersim'
        self.max_depth = 80 if self.dataset == 'vkitti' else 20

    def load_model(self):
        print(f"Loading Depth-Anything-V2 ({self.encoder}, {self.dataset}) from {self.model_dir}...")
        
        config = self.model_configs[self.encoder]
        
        try:
            # Try initializing with max_depth (expected by the metric dpt.py)
            self.model = DepthAnythingV2(**{**config, 'max_depth': self.max_depth})
        except TypeError:
            # Fallback if Python picks up the base dpt.py which doesn't take max_depth
            self.model = DepthAnythingV2(**config)
            self.model.max_depth = self.max_depth
            
        # Load weights
        state_dict = torch.load(self.model_dir, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        
        return self.model

    def predict(self, frame_rgb):
        if self.model is None:
            self.load_model()

        # The V2 infer_image method handles all processing and resizing automatically
        with torch.no_grad():
            depth_float = self.model.infer_image(frame_rgb) # Returns HxW numpy array in meters

        # Create a visual colormap normalized from 0 to max_depth
        # We cap the visual normalization at max_depth so close objects look distinct
        depth_clipped = np.clip(depth_float, 0, self.max_depth)
        depth_u8 = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        
        return depth_float, depth_color


def estimate_distance_from_depth(depth_map, bbox):
    """Estimate object depth from a bounding-box region in the depth map.
       Now returns physical distance in meters.
    """
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

    # The depth_value is now actual distance in meters, not a relative signal!
    return depth_value, depth_value