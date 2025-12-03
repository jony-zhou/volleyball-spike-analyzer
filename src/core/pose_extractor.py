"""
Pose extraction module using MediaPipe Pose.

This module provides functionality to extract 2D and 3D skeleton data
from video frames using Google's MediaPipe Pose solution.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)


class PoseExtractor:
    """
    Extracts pose landmarks from video frames using MediaPipe Pose.

    This class handles the initialization and processing of MediaPipe Pose
    to extract both 2D and 3D coordinates of body landmarks.

    Attributes:
        model_complexity: Complexity of pose model (0, 1, or 2).
        min_detection_confidence: Minimum confidence for detection.
        min_tracking_confidence: Minimum confidence for tracking.
        enable_segmentation: Whether to enable segmentation.
        smooth_landmarks: Whether to smooth landmarks.
    """

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_segmentation: bool = False,
        smooth_landmarks: bool = True
    ) -> None:
        """
        Initialize the PoseExtractor.

        Args:
            model_complexity: Model complexity (0, 1, or 2). Higher is more accurate but slower.
            min_detection_confidence: Minimum detection confidence threshold (0.0-1.0).
            min_tracking_confidence: Minimum tracking confidence threshold (0.0-1.0).
            enable_segmentation: Whether to enable segmentation mask.
            smooth_landmarks: Whether to apply landmark smoothing.

        Raises:
            ValueError: If parameters are out of valid range.
        """
        if not 0 <= model_complexity <= 2:
            raise ValueError("model_complexity must be 0, 1, or 2")
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0.0 and 1.0")
        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be between 0.0 and 1.0")

        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_segmentation = enable_segmentation
        self.smooth_landmarks = smooth_landmarks

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        logger.info(f"PoseExtractor initialized with model_complexity={model_complexity}")

    def extract_pose(self, frame: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract pose landmarks from a single frame.

        Args:
            frame: Input frame in BGR format (OpenCV format).

        Returns:
            Dictionary containing:
                - 'landmarks_2d': 2D landmarks (x, y, visibility) shape (33, 3)
                - 'landmarks_3d': 3D landmarks (x, y, z, visibility) shape (33, 4)
            Returns None if no pose detected.

        Raises:
            ValueError: If frame is invalid or empty.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is invalid or empty")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        try:
            results = self.pose.process(frame_rgb)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None

        # Check if pose was detected
        if not results.pose_landmarks:
            logger.debug("No pose detected in frame")
            return None

        # Extract 2D landmarks
        landmarks_2d = np.array([
            [lm.x, lm.y, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ])

        # Extract 3D landmarks
        landmarks_3d = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_world_landmarks.landmark
        ])

        return {
            'landmarks_2d': landmarks_2d,
            'landmarks_3d': landmarks_3d
        }

    def extract_from_video(
        self,
        video_path: str,
        frame_skip: int = 1
    ) -> List[Dict[str, np.ndarray]]:
        """
        Extract pose landmarks from all frames in a video.

        Args:
            video_path: Path to input video file.
            frame_skip: Process every Nth frame (1 = process all frames).

        Returns:
            List of dictionaries, each containing landmarks for a frame.

        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If frame_skip is less than 1.
        """
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")

        all_landmarks = []
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if needed
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue

                # Extract pose
                landmarks = self.extract_pose(frame)
                all_landmarks.append(landmarks)

                frame_count += 1

                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")

        finally:
            cap.release()

        logger.info(f"Extraction complete. Processed {frame_count} frames total")
        return all_landmarks

    def get_landmark_names(self) -> List[str]:
        """
        Get list of all landmark names.

        Returns:
            List of 33 landmark names in order.
        """
        return [landmark.name for landmark in self.mp_pose.PoseLandmark]

    def close(self) -> None:
        """Release resources used by the pose detector."""
        if hasattr(self, 'pose'):
            self.pose.close()
            logger.info("PoseExtractor closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
