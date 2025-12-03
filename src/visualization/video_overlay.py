"""
Video overlay module for 2D skeleton visualization.

This module provides functionality to draw 2D skeleton overlays on video
frames using OpenCV.
"""

import logging
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class VideoOverlay:
    """
    Draw 2D skeleton overlay on video frames.

    This class provides methods for drawing pose landmarks and connections
    on video frames using MediaPipe's connection definitions.

    Attributes:
        landmark_color: RGB color for landmarks.
        connection_color: RGB color for connections.
        landmark_radius: Radius of landmark circles.
        connection_thickness: Thickness of connection lines.
    """

    def __init__(
        self,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),
        connection_color: Tuple[int, int, int] = (255, 0, 0),
        landmark_radius: int = 5,
        connection_thickness: int = 2
    ) -> None:
        """
        Initialize the VideoOverlay.

        Args:
            landmark_color: RGB color for landmarks (default: green).
            connection_color: RGB color for connections (default: red).
            landmark_radius: Radius of landmark circles.
            connection_thickness: Thickness of connection lines.

        Raises:
            ValueError: If parameters are out of valid range.
        """
        if landmark_radius < 1:
            raise ValueError("landmark_radius must be at least 1")
        if connection_thickness < 1:
            raise ValueError("connection_thickness must be at least 1")

        self.landmark_color = landmark_color
        self.connection_color = connection_color
        self.landmark_radius = landmark_radius
        self.connection_thickness = connection_thickness

        # Get MediaPipe pose connections
        self.mp_pose = mp.solutions.pose
        self.connections = self.mp_pose.POSE_CONNECTIONS

        logger.info("VideoOverlay initialized")

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks_2d: np.ndarray,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Draw pose landmarks on a frame.

        Args:
            frame: Input frame in BGR format.
            landmarks_2d: 2D landmarks array of shape (33, 3) with (x, y, visibility).
            draw_connections: Whether to draw connections between landmarks.

        Returns:
            Frame with landmarks drawn (copy of original).

        Raises:
            ValueError: If frame or landmarks are invalid.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is invalid or empty")

        if landmarks_2d.shape[0] != 33 or landmarks_2d.shape[1] < 2:
            raise ValueError(
                f"Expected landmarks shape (33, 2+), got {landmarks_2d.shape}"
            )

        # Create a copy to avoid modifying the original
        output_frame = frame.copy()

        height, width = frame.shape[:2]

        # Convert normalized coordinates to pixel coordinates
        pixel_coords = []
        for i in range(33):
            x = int(landmarks_2d[i, 0] * width)
            y = int(landmarks_2d[i, 1] * height)
            visibility = landmarks_2d[i, 2] if landmarks_2d.shape[1] > 2 else 1.0
            pixel_coords.append((x, y, visibility))

        # Draw connections first (so they appear behind landmarks)
        if draw_connections:
            for connection in self.connections:
                start_idx = connection[0]
                end_idx = connection[1]

                start_x, start_y, start_vis = pixel_coords[start_idx]
                end_x, end_y, end_vis = pixel_coords[end_idx]

                # Only draw if both landmarks are visible
                if start_vis > 0.5 and end_vis > 0.5:
                    cv2.line(
                        output_frame,
                        (start_x, start_y),
                        (end_x, end_y),
                        self.connection_color,
                        self.connection_thickness
                    )

        # Draw landmarks
        for x, y, visibility in pixel_coords:
            if visibility > 0.5:  # Only draw visible landmarks
                cv2.circle(
                    output_frame,
                    (x, y),
                    self.landmark_radius,
                    self.landmark_color,
                    -1  # Filled circle
                )

        return output_frame

    def draw_landmarks_on_video(
        self,
        frames: List[np.ndarray],
        landmarks_sequence: List[Optional[np.ndarray]],
        draw_connections: bool = True
    ) -> List[np.ndarray]:
        """
        Draw landmarks on a sequence of video frames.

        Args:
            frames: List of input frames in BGR format.
            landmarks_sequence: List of 2D landmarks arrays (can contain None).
            draw_connections: Whether to draw connections between landmarks.

        Returns:
            List of frames with landmarks drawn.

        Raises:
            ValueError: If lengths don't match.
        """
        if len(frames) != len(landmarks_sequence):
            raise ValueError(
                f"Number of frames ({len(frames)}) doesn't match number of "
                f"landmark sets ({len(landmarks_sequence)})"
            )

        output_frames = []

        for frame, landmarks in zip(frames, landmarks_sequence):
            if landmarks is None:
                # If no landmarks detected, just use original frame
                output_frames.append(frame.copy())
            else:
                output_frames.append(
                    self.draw_landmarks(frame, landmarks, draw_connections)
                )

        logger.info(f"Drew landmarks on {len(output_frames)} frames")

        return output_frames

    def draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int] = (10, 30),
        font_scale: float = 1.0,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw text on a frame.

        Args:
            frame: Input frame.
            text: Text to draw.
            position: (x, y) position for text.
            font_scale: Font scale factor.
            color: RGB color for text.
            thickness: Text thickness.

        Returns:
            Frame with text drawn (copy of original).
        """
        output_frame = frame.copy()

        cv2.putText(
            output_frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )

        return output_frame

    def draw_angle_info(
        self,
        frame: np.ndarray,
        angles: dict,
        position: Tuple[int, int] = (10, 30),
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        Draw joint angle information on a frame.

        Args:
            frame: Input frame.
            angles: Dictionary of joint names to angles.
            position: Starting (x, y) position for text.
            font_scale: Font scale factor.

        Returns:
            Frame with angle information drawn.
        """
        output_frame = frame.copy()

        y_offset = position[1]
        line_height = int(30 * font_scale)

        for joint, angle in angles.items():
            text = f"{joint}: {angle:.1f}°"
            cv2.putText(
                output_frame,
                text,
                (position[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            y_offset += line_height

        return output_frame

    def highlight_landmark(
        self,
        frame: np.ndarray,
        landmarks_2d: np.ndarray,
        landmark_index: int,
        highlight_color: Tuple[int, int, int] = (0, 0, 255),
        highlight_radius: int = 10
    ) -> np.ndarray:
        """
        Highlight a specific landmark on a frame.

        Args:
            frame: Input frame.
            landmarks_2d: 2D landmarks array.
            landmark_index: Index of landmark to highlight (0-32).
            highlight_color: RGB color for highlight (default: red).
            highlight_radius: Radius of highlight circle.

        Returns:
            Frame with highlighted landmark.

        Raises:
            ValueError: If landmark_index is out of range.
        """
        if not 0 <= landmark_index < 33:
            raise ValueError(f"landmark_index must be 0-32, got {landmark_index}")

        output_frame = frame.copy()

        height, width = frame.shape[:2]

        x = int(landmarks_2d[landmark_index, 0] * width)
        y = int(landmarks_2d[landmark_index, 1] * height)
        visibility = landmarks_2d[landmark_index, 2] if landmarks_2d.shape[1] > 2 else 1.0

        if visibility > 0.5:
            cv2.circle(
                output_frame,
                (x, y),
                highlight_radius,
                highlight_color,
                2  # Outline only
            )

        return output_frame
