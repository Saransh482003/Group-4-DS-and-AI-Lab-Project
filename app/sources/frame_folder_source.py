import cv2
import os
from typing import Tuple, Any, Optional, List, Dict
from app.sources.base import FrameSource

class FrameFolderSource(FrameSource):
    """
    Reads and sequences image frames from a directory, supporting both flat 
    folder structures and nested sequence subfolders.
    """

    SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        # list of (sequence_id, image_filepath)
        self.image_files: List[Tuple[str, str]] = []
        self.current_idx = 0

    def _is_image_file(self, filename: str) -> bool:
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.SUPPORTED_EXTS

    def open(self) -> bool:
        if not os.path.isdir(self.root_dir):
            return False

        self.image_files = []

        # We'll use os.walk to find all image files
        for root, dirs, files in os.walk(self.root_dir):
            # Sort files to ensure frames are in order
            images = [f for f in files if self._is_image_file(f)]
            if images:
                # Use the relative path from root_dir to the current folder as the sequence_id
                rel_path = os.path.relpath(root, self.root_dir)
                seq_id = "root" if rel_path == "." else rel_path
                
                for img in sorted(images):
                    self.image_files.append((seq_id, os.path.join(root, img)))

        # Sort all image files by sequence and then filename to maintain overall order
        self.image_files.sort(key=lambda x: (x[0], x[1]))

        self.current_idx = 0
        return len(self.image_files) > 0

    def read(self) -> Tuple[bool, Any, Optional[str]]:
        if self.current_idx >= len(self.image_files):
            return False, None, None

        seq_id, filepath = self.image_files[self.current_idx]
        self.current_idx += 1

        frame_bgr = cv2.imread(filepath)
        if frame_bgr is None:
            # Skip unreadable images
            return self.read()

        # We can expose both sequence id and filename as source_id
        source_id = f"{seq_id}/{os.path.basename(filepath)}"
        return True, frame_bgr, source_id

    def close(self) -> None:
        self.image_files = []
        self.current_idx = 0
