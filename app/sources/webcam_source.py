import cv2
from typing import Tuple, Any, Optional
from app.sources.base import FrameSource

class WebcamFrameSource(FrameSource):
    """Reads frames from a local webcam or camera device."""

    def __init__(self, source_path: str):
        # We try to interpret source_path as an integer first
        try:
            self.source_id = int(source_path)
        except ValueError:
            self.source_id = source_path

        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.source_id)
        return self.cap.isOpened()

    def read(self) -> Tuple[bool, Any, Optional[str]]:
        if self.cap is None:
            return False, None, None

        ok, frame_bgr = self.cap.read()
        return ok, frame_bgr, str(self.source_id)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
