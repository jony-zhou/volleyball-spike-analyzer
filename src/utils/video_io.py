"""
Video I/O operations module.

This module provides functionality for reading and writing video files,
with support for various formats and frame manipulation operations.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """
    Read video files and extract frames.

    This class provides methods for reading video files and extracting
    frame information efficiently.

    Attributes:
        video_path: Path to the video file.
        cap: OpenCV VideoCapture object.
    """

    def __init__(self, video_path: str) -> None:
        """
        Initialize the VideoReader.

        Args:
            video_path: Path to the input video file.

        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If video file cannot be opened.
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        logger.info(f"VideoReader initialized for: {video_path}")

    def get_properties(self) -> dict:
        """
        Get video properties.

        Returns:
            Dictionary containing:
                - 'width': Frame width
                - 'height': Frame height
                - 'fps': Frames per second
                - 'frame_count': Total number of frames
                - 'duration': Duration in seconds
        """
        properties = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        properties['duration'] = properties['frame_count'] / properties['fps']

        return properties

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read the next frame.

        Returns:
            Frame as numpy array in BGR format, or None if no more frames.
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def read_all_frames(self, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Read all frames from the video.

        Args:
            max_frames: Maximum number of frames to read (None for all).

        Returns:
            List of frames as numpy arrays.

        Raises:
            ValueError: If max_frames is less than 1.
        """
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be at least 1")

        frames = []
        frame_count = 0

        while True:
            frame = self.read_frame()
            if frame is None:
                break

            frames.append(frame)
            frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

            if frame_count % 100 == 0:
                logger.debug(f"Read {frame_count} frames")

        logger.info(f"Read {frame_count} frames total")
        return frames

    def seek_frame(self, frame_number: int) -> bool:
        """
        Seek to a specific frame number.

        Args:
            frame_number: Frame number to seek to (0-indexed).

        Returns:
            True if seek was successful, False otherwise.

        Raises:
            ValueError: If frame_number is negative.
        """
        if frame_number < 0:
            raise ValueError("frame_number must be non-negative")

        success = self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return bool(success)

    def reset(self) -> None:
        """Reset video to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self) -> None:
        """Release video capture resources."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            logger.info("VideoReader closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()


class VideoWriter:
    """
    Write video files with frames.

    This class provides methods for creating video files and writing
    frames with various codecs and settings.

    Attributes:
        output_path: Path to the output video file.
        fps: Frames per second.
        frame_size: (width, height) tuple.
        writer: OpenCV VideoWriter object.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = 'mp4v'
    ) -> None:
        """
        Initialize the VideoWriter.

        Args:
            output_path: Path to the output video file.
            fps: Frames per second.
            frame_size: (width, height) tuple.
            codec: FourCC codec code (e.g., 'mp4v', 'XVID', 'H264').

        Raises:
            ValueError: If parameters are invalid.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        if len(frame_size) != 2 or frame_size[0] <= 0 or frame_size[1] <= 0:
            raise ValueError(f"Invalid frame_size: {frame_size}")

        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size

        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            frame_size
        )

        if not self.writer.isOpened():
            raise ValueError(f"Could not initialize VideoWriter for: {output_path}")

        self.frame_count = 0
        logger.info(f"VideoWriter initialized for: {output_path}")

    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a single frame to the video.

        Args:
            frame: Frame as numpy array in BGR format.

        Raises:
            ValueError: If frame dimensions don't match.
        """
        if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
            raise ValueError(
                f"Frame size {(frame.shape[1], frame.shape[0])} doesn't match "
                f"expected size {self.frame_size}"
            )

        self.writer.write(frame)
        self.frame_count += 1

        if self.frame_count % 100 == 0:
            logger.debug(f"Written {self.frame_count} frames")

    def write_frames(self, frames: List[np.ndarray]) -> None:
        """
        Write multiple frames to the video.

        Args:
            frames: List of frames as numpy arrays.

        Raises:
            ValueError: If any frame dimensions don't match.
        """
        for frame in frames:
            self.write_frame(frame)

        logger.info(f"Written {len(frames)} frames")

    def close(self) -> None:
        """Release video writer resources."""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.release()
            logger.info(f"VideoWriter closed. Total frames written: {self.frame_count}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()


def resize_frame(
    frame: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    scale: Optional[float] = None
) -> np.ndarray:
    """
    Resize a frame to target size or by scale factor.

    Args:
        frame: Input frame.
        target_size: Target (width, height), or None to use scale.
        scale: Scale factor, or None to use target_size.

    Returns:
        Resized frame.

    Raises:
        ValueError: If neither or both target_size and scale are provided.
    """
    if (target_size is None) == (scale is None):
        raise ValueError("Must provide either target_size or scale, but not both")

    if target_size is not None:
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR
        )

    return resized


def validate_video_file(file_path: str, supported_formats: List[str]) -> bool:
    """
    Validate if a video file exists and has supported format.

    Args:
        file_path: Path to video file.
        supported_formats: List of supported extensions (e.g., ['.mp4', '.avi']).

    Returns:
        True if valid, False otherwise.
    """
    path = Path(file_path)

    if not path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return False

    if path.suffix.lower() not in supported_formats:
        logger.warning(f"Unsupported format: {path.suffix}")
        return False

    return True
