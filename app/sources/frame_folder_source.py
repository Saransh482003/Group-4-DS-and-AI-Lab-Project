import cv2
import os
from typing import Tuple, Any, Optional, List, Dict
from app.sources.base import FrameSource

class FrameFolderSource(FrameSource):
    """
    Reads frames from a directory of images.
    Supports either a flat folder of images, or a folder containing sequence subfolders.
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

        # Check if the root directory itself contains subfolders
        entries = os.listdir(self.root_dir)
        subdirs = [e for e in entries if os.path.isdir(os.path.join(self.root_dir, e))]

        if subdirs:
            # Multi-sequence mode
            # Sort subdirectories to maintain predictable order
            for subdir in sorted(subdirs):
                seq_dir = os.path.join(self.root_dir, subdir)
                images = [f for f in os.listdir(seq_dir) if self._is_image_file(f)]
                for img in sorted(images):
                    self.image_files.append((subdir, os.path.join(seq_dir, img)))
        else:
            # Flat folder mode
            images = [f for f in entries if self._is_image_file(f)]
            for img in sorted(images):
                self.image_files.append(("seq_0", os.path.join(self.root_dir, img)))

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
