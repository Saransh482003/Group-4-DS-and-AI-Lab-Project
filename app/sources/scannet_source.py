import cv2
import os
import glob
from typing import Tuple, Any, Optional, List
from app.sources.base import FrameSource

class ScanNetSource(FrameSource):
    """
    Reads frames from a ScanNet dataset structure, supporting per-scene 
    navigation through sequential color video frames.
    """

    def __init__(self, root_dir: str, include_depth: bool = False):
        self.root_dir = root_dir
        self.include_depth = include_depth
        self.image_files: List[Tuple[str, str, Optional[str]]] = []
        self.current_idx = 0

    def open(self) -> bool:
        if not os.path.isdir(self.root_dir):
            print(f"ScanNet root directory not found: {self.root_dir}")
            return False

        self.image_files = []
        
        # Scan for scene folders
        scenes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        if not scenes:
            # Maybe the root_dir IS the scene folder
            if os.path.isdir(os.path.join(self.root_dir, "color")):
                scenes = ["."]
            else:
                print(f"No scene folders or 'color' subfolder found in {self.root_dir}")
                return False

        for scene in sorted(scenes):
            scene_path = os.path.join(self.root_dir, scene)
            color_path = os.path.join(scene_path, "color")
            depth_path = os.path.join(scene_path, "depth")

            if not os.path.isdir(color_path):
                continue

            # Get all jpg/png files in color/ folder
            images = sorted(glob.glob(os.path.join(color_path, "*.jpg")) + glob.glob(os.path.join(color_path, "*.png")))
            
            for img_path in images:
                # Try to find matching depth file
                depth_file = None
                if self.include_depth:
                    base_name = os.path.basename(img_path).split('.')[0]
                    # ScanNet depth is usually .png
                    potential_depth = os.path.join(depth_path, f"{base_name}.png")
                    if os.path.exists(potential_depth):
                        depth_file = potential_depth
                
                self.image_files.append((scene, img_path, depth_file))

        self.current_idx = 0
        return len(self.image_files) > 0

    def read(self) -> Tuple[bool, Any, Optional[str]]:
        if self.current_idx >= len(self.image_files):
            return False, None, None

        scene_id, color_path, depth_path = self.image_files[self.current_idx]
        self.current_idx += 1

        frame_bgr = cv2.imread(color_path)
        if frame_bgr is None:
            return self.read()

        source_id = f"scannet/{scene_id}/{os.path.basename(color_path)}"
        
        # If we want to return depth too, we'd need to modify the FrameSource interface 
        # or store it in the context later. For now, we follow the interface.
        return True, frame_bgr, source_id

    def close(self) -> None:
        self.image_files = []
        self.current_idx = 0
