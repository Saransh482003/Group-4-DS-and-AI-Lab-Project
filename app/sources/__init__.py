from .base import FrameSource
from .webcam_source import WebcamFrameSource
from .video_source import VideoFileFrameSource
from .frame_folder_source import FrameFolderSource

__all__ = ["FrameSource", "WebcamFrameSource", "VideoFileFrameSource", "FrameFolderSource"]
