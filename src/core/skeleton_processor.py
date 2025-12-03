"""
Skeleton data processing module.

This module provides functionality to process raw pose landmarks,
including calculating joint angles, velocities, and smoothing trajectories.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


class SkeletonProcessor:
    """
    Process skeleton data extracted from pose estimation.

    This class provides methods for calculating joint angles, velocities,
    and smoothing trajectories for biomechanical analysis.

    Attributes:
        smoothing_window: Window size for trajectory smoothing.
        calculate_angles: Whether to calculate joint angles.
        calculate_velocities: Whether to calculate velocities.
    """

    # MediaPipe landmark indices
    LANDMARK_INDICES = {
        'nose': 0,
        'left_eye_inner': 1,
        'left_eye': 2,
        'left_eye_outer': 3,
        'right_eye_inner': 4,
        'right_eye': 5,
        'right_eye_outer': 6,
        'left_ear': 7,
        'right_ear': 8,
        'mouth_left': 9,
        'mouth_right': 10,
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_pinky': 17,
        'right_pinky': 18,
        'left_index': 19,
        'right_index': 20,
        'left_thumb': 21,
        'right_thumb': 22,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
        'left_heel': 29,
        'right_heel': 30,
        'left_foot_index': 31,
        'right_foot_index': 32
    }

    def __init__(
        self,
        smoothing_window: int = 5,
        calculate_angles: bool = True,
        calculate_velocities: bool = True
    ) -> None:
        """
        Initialize the SkeletonProcessor.

        Args:
            smoothing_window: Window size for smoothing (odd number recommended).
            calculate_angles: Whether to calculate joint angles.
            calculate_velocities: Whether to calculate joint velocities.

        Raises:
            ValueError: If smoothing_window is less than 1.
        """
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be at least 1")

        self.smoothing_window = smoothing_window
        self.calculate_angles = calculate_angles
        self.calculate_velocities = calculate_velocities

        logger.info(f"SkeletonProcessor initialized with smoothing_window={smoothing_window}")

    def smooth_trajectories(
        self,
        landmarks_sequence: np.ndarray
    ) -> np.ndarray:
        """
        Smooth landmark trajectories over time using uniform filter.

        Args:
            landmarks_sequence: Array of shape (num_frames, num_landmarks, num_coords).

        Returns:
            Smoothed landmarks with same shape as input.

        Raises:
            ValueError: If input shape is invalid.
        """
        if landmarks_sequence.ndim != 3:
            raise ValueError(
                f"Expected 3D array (frames, landmarks, coords), got shape {landmarks_sequence.shape}"
            )

        if len(landmarks_sequence) < self.smoothing_window:
            logger.warning(
                f"Sequence length {len(landmarks_sequence)} is shorter than smoothing window "
                f"{self.smoothing_window}. Returning unsmoothed data."
            )
            return landmarks_sequence

        # Apply smoothing along time axis (axis=0)
        smoothed = uniform_filter1d(
            landmarks_sequence,
            size=self.smoothing_window,
            axis=0,
            mode='nearest'
        )

        return smoothed

    def calculate_angle(
        self,
        point1: np.ndarray,
        point2: np.ndarray,
        point3: np.ndarray
    ) -> float:
        """
        Calculate angle at point2 formed by three points.

        Args:
            point1: First point (x, y, z).
            point2: Vertex point (x, y, z).
            point3: Third point (x, y, z).

        Returns:
            Angle in degrees (0-180).

        Raises:
            ValueError: If points are not 3D.
        """
        if point1.shape[0] < 2 or point2.shape[0] < 2 or point3.shape[0] < 2:
            raise ValueError("Points must have at least 2 coordinates")

        # Calculate vectors
        vector1 = point1[:3] - point2[:3]
        vector2 = point3[:3] - point2[:3]

        # Calculate angle using dot product
        cos_angle = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-8
        )

        # Clip to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Convert to degrees
        angle = np.degrees(np.arccos(cos_angle))

        return float(angle)

    def calculate_joint_angles(
        self,
        landmarks_3d: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate key joint angles for spike analysis.

        Args:
            landmarks_3d: 3D landmarks array of shape (33, 3 or 4).

        Returns:
            Dictionary mapping joint names to angles in degrees.

        Raises:
            ValueError: If landmarks shape is invalid.
        """
        if landmarks_3d.shape[0] != 33:
            raise ValueError(f"Expected 33 landmarks, got {landmarks_3d.shape[0]}")

        angles = {}

        # Right arm angles
        try:
            angles['right_elbow'] = self.calculate_angle(
                landmarks_3d[self.LANDMARK_INDICES['right_shoulder']],
                landmarks_3d[self.LANDMARK_INDICES['right_elbow']],
                landmarks_3d[self.LANDMARK_INDICES['right_wrist']]
            )

            angles['right_shoulder'] = self.calculate_angle(
                landmarks_3d[self.LANDMARK_INDICES['right_hip']],
                landmarks_3d[self.LANDMARK_INDICES['right_shoulder']],
                landmarks_3d[self.LANDMARK_INDICES['right_elbow']]
            )
        except Exception as e:
            logger.warning(f"Could not calculate right arm angles: {e}")

        # Left arm angles
        try:
            angles['left_elbow'] = self.calculate_angle(
                landmarks_3d[self.LANDMARK_INDICES['left_shoulder']],
                landmarks_3d[self.LANDMARK_INDICES['left_elbow']],
                landmarks_3d[self.LANDMARK_INDICES['left_wrist']]
            )

            angles['left_shoulder'] = self.calculate_angle(
                landmarks_3d[self.LANDMARK_INDICES['left_hip']],
                landmarks_3d[self.LANDMARK_INDICES['left_shoulder']],
                landmarks_3d[self.LANDMARK_INDICES['left_elbow']]
            )
        except Exception as e:
            logger.warning(f"Could not calculate left arm angles: {e}")

        # Right leg angles
        try:
            angles['right_knee'] = self.calculate_angle(
                landmarks_3d[self.LANDMARK_INDICES['right_hip']],
                landmarks_3d[self.LANDMARK_INDICES['right_knee']],
                landmarks_3d[self.LANDMARK_INDICES['right_ankle']]
            )

            angles['right_hip'] = self.calculate_angle(
                landmarks_3d[self.LANDMARK_INDICES['right_shoulder']],
                landmarks_3d[self.LANDMARK_INDICES['right_hip']],
                landmarks_3d[self.LANDMARK_INDICES['right_knee']]
            )
        except Exception as e:
            logger.warning(f"Could not calculate right leg angles: {e}")

        # Left leg angles
        try:
            angles['left_knee'] = self.calculate_angle(
                landmarks_3d[self.LANDMARK_INDICES['left_hip']],
                landmarks_3d[self.LANDMARK_INDICES['left_knee']],
                landmarks_3d[self.LANDMARK_INDICES['left_ankle']]
            )

            angles['left_hip'] = self.calculate_angle(
                landmarks_3d[self.LANDMARK_INDICES['left_shoulder']],
                landmarks_3d[self.LANDMARK_INDICES['left_hip']],
                landmarks_3d[self.LANDMARK_INDICES['left_knee']]
            )
        except Exception as e:
            logger.warning(f"Could not calculate left leg angles: {e}")

        return angles

    def calculate_velocity(
        self,
        landmarks_sequence: np.ndarray,
        fps: float = 30.0
    ) -> np.ndarray:
        """
        Calculate velocity of landmarks over time.

        Args:
            landmarks_sequence: Array of shape (num_frames, num_landmarks, num_coords).
            fps: Frames per second of the video.

        Returns:
            Velocity array of shape (num_frames-1, num_landmarks).

        Raises:
            ValueError: If input shape is invalid or fps is non-positive.
        """
        if landmarks_sequence.ndim != 3:
            raise ValueError(
                f"Expected 3D array (frames, landmarks, coords), got shape {landmarks_sequence.shape}"
            )

        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        # Calculate displacement between consecutive frames
        displacement = np.diff(landmarks_sequence, axis=0)

        # Calculate magnitude of displacement for each landmark
        velocity = np.linalg.norm(displacement, axis=2) * fps

        return velocity

    def get_landmark_position(
        self,
        landmarks: np.ndarray,
        landmark_name: str
    ) -> np.ndarray:
        """
        Get position of a specific landmark by name.

        Args:
            landmarks: Landmarks array of shape (33, 3 or 4).
            landmark_name: Name of the landmark (e.g., 'left_wrist').

        Returns:
            Position array (x, y, z) or (x, y, z, visibility).

        Raises:
            KeyError: If landmark name is invalid.
            ValueError: If landmarks shape is invalid.
        """
        if landmark_name not in self.LANDMARK_INDICES:
            raise KeyError(f"Unknown landmark name: {landmark_name}")

        if landmarks.shape[0] != 33:
            raise ValueError(f"Expected 33 landmarks, got {landmarks.shape[0]}")

        index = self.LANDMARK_INDICES[landmark_name]
        return landmarks[index]

    def process_sequence(
        self,
        landmarks_sequence: List[Optional[Dict[str, np.ndarray]]],
        fps: float = 30.0
    ) -> Dict[str, np.ndarray]:
        """
        Process a complete sequence of landmarks.

        Args:
            landmarks_sequence: List of landmark dictionaries from PoseExtractor.
            fps: Video frame rate.

        Returns:
            Dictionary containing:
                - 'landmarks_3d_smooth': Smoothed 3D landmarks
                - 'angles': Joint angles over time (if calculate_angles=True)
                - 'velocities': Landmark velocities (if calculate_velocities=True)

        Raises:
            ValueError: If sequence is empty or fps is invalid.
        """
        if not landmarks_sequence:
            raise ValueError("landmarks_sequence is empty")

        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        # Filter out None values and extract 3D landmarks
        valid_landmarks = [
            lm['landmarks_3d'] for lm in landmarks_sequence if lm is not None
        ]

        if not valid_landmarks:
            raise ValueError("No valid landmarks found in sequence")

        # Stack into array (num_frames, 33, 4)
        landmarks_array = np.stack(valid_landmarks, axis=0)

        result = {}

        # Smooth trajectories
        smoothed = self.smooth_trajectories(landmarks_array)
        result['landmarks_3d_smooth'] = smoothed

        # Calculate angles for each frame
        if self.calculate_angles:
            angles_sequence = []
            for frame_landmarks in smoothed:
                angles = self.calculate_joint_angles(frame_landmarks)
                angles_sequence.append(angles)
            result['angles'] = angles_sequence

        # Calculate velocities
        if self.calculate_velocities:
            velocities = self.calculate_velocity(landmarks_array, fps)
            result['velocities'] = velocities

        logger.info(f"Processed sequence of {len(valid_landmarks)} frames")

        return result
