"""
Joint angle calculation module for volleyball spike analysis.

This module provides functionality to calculate key joint angles
for biomechanical analysis of volleyball spike techniques.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class JointAngleCalculator:
    """
    Calculate joint angles for biomechanical analysis.

    This class provides methods to calculate various joint angles including:
    - Shoulder abduction
    - Shoulder horizontal abduction
    - Elbow flexion
    - Torso rotation
    - Torso lean

    Attributes:
        min_visibility: Minimum visibility threshold for landmarks.
    """

    # MediaPipe landmark indices
    LANDMARK_INDICES = {
        'right_shoulder': 12,
        'left_shoulder': 11,
        'right_elbow': 14,
        'left_elbow': 13,
        'right_wrist': 16,
        'left_wrist': 15,
        'right_hip': 24,
        'left_hip': 23
    }

    def __init__(self, min_visibility: float = 0.5) -> None:
        """
        Initialize the JointAngleCalculator.

        Args:
            min_visibility: Minimum visibility threshold (0.0-1.0).

        Raises:
            ValueError: If min_visibility is out of range.
        """
        if not 0.0 <= min_visibility <= 1.0:
            raise ValueError("min_visibility must be between 0.0 and 1.0")

        self.min_visibility = min_visibility
        logger.info(f"JointAngleCalculator initialized with min_visibility={min_visibility}")

    def calculate_angle(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
        point_c: np.ndarray
    ) -> float:
        """
        Calculate angle at point_b formed by three points.

        Args:
            point_a: First point (x, y, z).
            point_b: Vertex point (x, y, z).
            point_c: Third point (x, y, z).

        Returns:
            Angle in degrees (0-180).

        Raises:
            ValueError: If points are not 3D or have invalid dimensions.
        """
        # Validate input
        if point_a.size < 3 or point_b.size < 3 or point_c.size < 3:
            raise ValueError("All points must have at least 3 coordinates (x, y, z)")

        # Extract only x, y, z coordinates (ignore visibility if present)
        a = point_a[:3].astype(float)
        b = point_b[:3].astype(float)
        c = point_c[:3].astype(float)

        # Calculate vectors
        vector_ba = a - b
        vector_bc = c - b

        # Calculate angle using dot product
        dot_product = np.dot(vector_ba, vector_bc)
        magnitude_ba = np.linalg.norm(vector_ba)
        magnitude_bc = np.linalg.norm(vector_bc)

        # Avoid division by zero
        if magnitude_ba < 1e-8 or magnitude_bc < 1e-8:
            logger.warning("Vector magnitude too small, returning 0.0")
            return 0.0

        # Calculate cosine of angle
        cos_angle = dot_product / (magnitude_ba * magnitude_bc)

        # Clip to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Convert to degrees
        angle = np.degrees(np.arccos(cos_angle))

        return float(angle)

    def _check_visibility(self, landmarks: np.ndarray, indices: List[int]) -> bool:
        """
        Check if landmarks are visible enough for calculation.

        Args:
            landmarks: Landmarks array of shape (33, 3 or 4).
            indices: List of landmark indices to check.

        Returns:
            True if all landmarks have sufficient visibility.
        """
        if landmarks.shape[1] < 4:
            # No visibility information, assume visible
            return True

        for idx in indices:
            if landmarks[idx, 3] < self.min_visibility:
                return False
        return True

    def calculate_shoulder_abduction(self, landmarks: np.ndarray) -> float:
        """
        Calculate shoulder abduction angle.

        The angle between the vertical axis and the shoulder-elbow vector.

        Args:
            landmarks: 3D landmarks array of shape (33, 3 or 4).

        Returns:
            Shoulder abduction angle in degrees.
        """
        # Get landmark positions
        shoulder = landmarks[self.LANDMARK_INDICES['right_shoulder'], :3]
        elbow = landmarks[self.LANDMARK_INDICES['right_elbow'], :3]
        hip_right = landmarks[self.LANDMARK_INDICES['right_hip'], :3]
        hip_left = landmarks[self.LANDMARK_INDICES['left_hip'], :3]

        # Calculate hip midpoint
        hip_mid = (hip_right + hip_left) / 2

        # Create a point directly above the shoulder (vertical reference)
        point_above = shoulder.copy()
        point_above[1] += 1.0  # Y-axis is up

        # Calculate angle: hip_mid -> shoulder -> elbow
        angle = self.calculate_angle(hip_mid, shoulder, elbow)

        return angle

    def calculate_shoulder_horizontal_abduction(self, landmarks: np.ndarray) -> float:
        """
        Calculate shoulder horizontal abduction angle.

        The angle between the shoulder-elbow vector and the body midline
        in the horizontal plane (XZ plane).

        Args:
            landmarks: 3D landmarks array of shape (33, 3 or 4).

        Returns:
            Horizontal abduction angle in degrees.
        """
        # Get landmark positions
        shoulder = landmarks[self.LANDMARK_INDICES['right_shoulder'], :3]
        elbow = landmarks[self.LANDMARK_INDICES['right_elbow'], :3]
        hip_right = landmarks[self.LANDMARK_INDICES['right_hip'], :3]
        hip_left = landmarks[self.LANDMARK_INDICES['left_hip'], :3]

        # Calculate hip midpoint (body center)
        hip_mid = (hip_right + hip_left) / 2

        # Project to horizontal plane (XZ plane, Y=0)
        shoulder_proj = shoulder.copy()
        shoulder_proj[1] = 0
        elbow_proj = elbow.copy()
        elbow_proj[1] = 0
        hip_mid_proj = hip_mid.copy()
        hip_mid_proj[1] = 0

        # Create a point along the body midline (forward direction)
        forward_point = hip_mid_proj.copy()
        forward_point[2] += 1.0  # Z-axis is forward

        # Calculate angle: forward -> shoulder -> elbow (in horizontal plane)
        angle = self.calculate_angle(forward_point, shoulder_proj, elbow_proj)

        return angle

    def calculate_elbow_flexion(self, landmarks: np.ndarray) -> float:
        """
        Calculate elbow flexion angle.

        The angle between the shoulder-elbow and elbow-wrist vectors.

        Args:
            landmarks: 3D landmarks array of shape (33, 3 or 4).

        Returns:
            Elbow flexion angle in degrees.
        """
        # Get landmark positions
        shoulder = landmarks[self.LANDMARK_INDICES['right_shoulder'], :3]
        elbow = landmarks[self.LANDMARK_INDICES['right_elbow'], :3]
        wrist = landmarks[self.LANDMARK_INDICES['right_wrist'], :3]

        # Calculate angle: shoulder -> elbow -> wrist
        angle = self.calculate_angle(shoulder, elbow, wrist)

        return angle

    def calculate_torso_rotation(self, landmarks: np.ndarray) -> float:
        """
        Calculate torso rotation angle.

        The angle between the shoulder line and hip line in the horizontal plane.

        Args:
            landmarks: 3D landmarks array of shape (33, 3 or 4).

        Returns:
            Torso rotation angle in degrees.
        """
        # Get landmark positions
        shoulder_right = landmarks[self.LANDMARK_INDICES['right_shoulder'], :3]
        shoulder_left = landmarks[self.LANDMARK_INDICES['left_shoulder'], :3]
        hip_right = landmarks[self.LANDMARK_INDICES['right_hip'], :3]
        hip_left = landmarks[self.LANDMARK_INDICES['left_hip'], :3]

        # Project to horizontal plane (XZ plane)
        shoulder_right_proj = shoulder_right.copy()
        shoulder_right_proj[1] = 0
        shoulder_left_proj = shoulder_left.copy()
        shoulder_left_proj[1] = 0
        hip_right_proj = hip_right.copy()
        hip_right_proj[1] = 0
        hip_left_proj = hip_left.copy()
        hip_left_proj[1] = 0

        # Calculate shoulder and hip vectors
        shoulder_vector = shoulder_right_proj - shoulder_left_proj
        hip_vector = hip_right_proj - hip_left_proj

        # Calculate angle between vectors
        dot_product = np.dot(shoulder_vector, hip_vector)
        magnitude_shoulder = np.linalg.norm(shoulder_vector)
        magnitude_hip = np.linalg.norm(hip_vector)

        if magnitude_shoulder < 1e-8 or magnitude_hip < 1e-8:
            logger.warning("Vector magnitude too small for torso rotation, returning 0.0")
            return 0.0

        cos_angle = dot_product / (magnitude_shoulder * magnitude_hip)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def calculate_torso_lean(self, landmarks: np.ndarray) -> float:
        """
        Calculate torso lean angle.

        The angle between the torso (shoulder midpoint to hip midpoint)
        and the vertical axis.

        Args:
            landmarks: 3D landmarks array of shape (33, 3 or 4).

        Returns:
            Torso lean angle in degrees (0 = vertical, 90 = horizontal).
        """
        # Get landmark positions
        shoulder_right = landmarks[self.LANDMARK_INDICES['right_shoulder'], :3]
        shoulder_left = landmarks[self.LANDMARK_INDICES['left_shoulder'], :3]
        hip_right = landmarks[self.LANDMARK_INDICES['right_hip'], :3]
        hip_left = landmarks[self.LANDMARK_INDICES['left_hip'], :3]

        # Calculate midpoints
        shoulder_mid = (shoulder_right + shoulder_left) / 2
        hip_mid = (hip_right + hip_left) / 2

        # Calculate torso vector
        torso_vector = shoulder_mid - hip_mid

        # Calculate vertical reference vector (Y-axis)
        vertical_vector = np.array([0.0, 1.0, 0.0])

        # Calculate angle
        dot_product = np.dot(torso_vector, vertical_vector)
        magnitude_torso = np.linalg.norm(torso_vector)

        if magnitude_torso < 1e-8:
            logger.warning("Torso vector magnitude too small, returning 0.0")
            return 0.0

        cos_angle = dot_product / magnitude_torso
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def calculate_all_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculate all joint angles for a single frame.

        Args:
            landmarks: 3D landmarks array of shape (33, 3 or 4).

        Returns:
            Dictionary mapping angle names to values in degrees:
            {
                'shoulder_abduction': float,
                'shoulder_horizontal_abduction': float,
                'elbow_flexion': float,
                'torso_rotation': float,
                'torso_lean': float
            }

        Raises:
            ValueError: If landmarks shape is invalid.
        """
        if landmarks.shape[0] != 33:
            raise ValueError(f"Expected 33 landmarks, got {landmarks.shape[0]}")

        # Check visibility of key landmarks
        key_indices = [
            self.LANDMARK_INDICES['right_shoulder'],
            self.LANDMARK_INDICES['right_elbow'],
            self.LANDMARK_INDICES['right_wrist'],
            self.LANDMARK_INDICES['right_hip'],
            self.LANDMARK_INDICES['left_hip']
        ]

        if not self._check_visibility(landmarks, key_indices):
            logger.warning("Key landmarks not visible enough, returning NaN values")
            return {
                'shoulder_abduction': np.nan,
                'shoulder_horizontal_abduction': np.nan,
                'elbow_flexion': np.nan,
                'torso_rotation': np.nan,
                'torso_lean': np.nan
            }

        angles = {}

        try:
            angles['shoulder_abduction'] = self.calculate_shoulder_abduction(landmarks)
        except Exception as e:
            logger.warning(f"Could not calculate shoulder abduction: {e}")
            angles['shoulder_abduction'] = np.nan

        try:
            angles['shoulder_horizontal_abduction'] = self.calculate_shoulder_horizontal_abduction(landmarks)
        except Exception as e:
            logger.warning(f"Could not calculate shoulder horizontal abduction: {e}")
            angles['shoulder_horizontal_abduction'] = np.nan

        try:
            angles['elbow_flexion'] = self.calculate_elbow_flexion(landmarks)
        except Exception as e:
            logger.warning(f"Could not calculate elbow flexion: {e}")
            angles['elbow_flexion'] = np.nan

        try:
            angles['torso_rotation'] = self.calculate_torso_rotation(landmarks)
        except Exception as e:
            logger.warning(f"Could not calculate torso rotation: {e}")
            angles['torso_rotation'] = np.nan

        try:
            angles['torso_lean'] = self.calculate_torso_lean(landmarks)
        except Exception as e:
            logger.warning(f"Could not calculate torso lean: {e}")
            angles['torso_lean'] = np.nan

        return angles

    def calculate_angles_timeseries(
        self,
        skeleton_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate angles for entire time series.

        Args:
            skeleton_df: DataFrame with columns ['frame', 'time', 'landmarks_3d'].
                landmarks_3d should be np.ndarray of shape (33, 3 or 4).

        Returns:
            DataFrame with columns:
            ['frame', 'time', 'shoulder_abduction', 'shoulder_horizontal_abduction',
             'elbow_flexion', 'torso_rotation', 'torso_lean']

        Raises:
            ValueError: If input DataFrame is invalid.
        """
        # Validate input
        required_columns = ['frame', 'time', 'landmarks_3d']
        if not all(col in skeleton_df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        # Calculate angles for each frame
        angles_list = []
        for idx, row in skeleton_df.iterrows():
            angles = self.calculate_all_angles(row['landmarks_3d'])
            angles['frame'] = row['frame']
            angles['time'] = row['time']
            angles_list.append(angles)

        # Create DataFrame
        angles_df = pd.DataFrame(angles_list)

        # Reorder columns
        column_order = [
            'frame', 'time', 'shoulder_abduction', 'shoulder_horizontal_abduction',
            'elbow_flexion', 'torso_rotation', 'torso_lean'
        ]
        angles_df = angles_df[column_order]

        logger.info(f"Calculated angles for {len(angles_df)} frames")

        return angles_df
