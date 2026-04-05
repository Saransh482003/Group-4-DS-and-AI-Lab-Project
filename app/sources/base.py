import abc
from typing import Tuple, Any, Optional

class FrameSource(abc.ABC):
    """Base interface for all frame sources."""

    @abc.abstractmethod
    def open(self) -> bool:
        """Initialize and open the source. Return True if successful."""
        pass

    @abc.abstractmethod
    def read(self) -> Tuple[bool, Any, Optional[str]]:
        """
        Read the next frame.
        Returns:
            (success: bool, frame_bgr: np.ndarray, source_id: str | None)
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Release resources."""
        pass
