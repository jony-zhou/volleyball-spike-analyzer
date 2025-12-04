"""
Spatial metrics calculation module for volleyball spike analysis.

This module provides functionality to calculate spatial metrics including
jump height, contact height, flight time, and horizontal displacement.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SpatialMetricsCalculator:
    """
    Calculate spatial metrics for spike analysis.

    This class provides methods to calculate:
    - Jump height (using multiple methods)
    - Contact point height
    - Flight time
    - Horizontal displacement
    - Center of mass trajectory

    Attributes:
        gravity: Gravitational acceleration constant (m/s²).
    """

    # Physical constants
    GRAVITY = 9.81  # m/s²

    # MediaPipe landmark indices
    LANDMARK_INDICES = {
        'right_wrist': 16,
        'right_elbow': 14,
        'right_shoulder': 12,
        'left_shoulder': 11,
        'right_hip': 24,
        'left_hip': 23,
        'right_ankle': 28,
        'left_ankle': 27
    }

    def __init__(self, gravity: float = GRAVITY) -> None:
        """
        Initialize the SpatialMetricsCalculator.

        Args:
            gravity: Gravitational acceleration (m/s²). Default is 9.81.

        Raises:
            ValueError: If gravity is non-positive.
        """
        if gravity <= 0:
            raise ValueError("gravity must be positive")

        self.gravity = gravity
        logger.info(f"SpatialMetricsCalculator initialized with gravity={gravity} m/s²")

    def _extract_landmark_position(
        self,
        landmarks: np.ndarray,
        landmark_idx: int
    ) -> np.ndarray:
        """
        Extract 3D position of a specific landmark.

        Args:
            landmarks: Array of shape (33, 3 or 4).
            landmark_idx: Index of the landmark.

        Returns:
            Array of shape (3,) containing x, y, z coordinates.
        """
        return landmarks[landmark_idx, :3]

    def _calculate_hip_center(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate center point of hips.

        Args:
            landmarks: Array of shape (33, 3 or 4).

        Returns:
            Array of shape (3,) containing hip center coordinates.
        """
        right_hip = self._extract_landmark_position(landmarks, self.LANDMARK_INDICES['right_hip'])
        left_hip = self._extract_landmark_position(landmarks, self.LANDMARK_INDICES['left_hip'])
        return (right_hip + left_hip) / 2

    def calculate_baseline_height(
        self,
        skeleton_df: pd.DataFrame,
        phase_info: dict,
        landmark_name: str = 'hip_center'
    ) -> float:
        """
        Calculate baseline height before jump.

        Args:
            skeleton_df: DataFrame with 'landmarks_3d' column.
            phase_info: Dictionary containing phase boundaries.
            landmark_name: Name of landmark ('hip_center', 'right_ankle', etc.).

        Returns:
            Baseline height in meters.

        Raises:
            ValueError: If phase_info doesn't contain 'approach' phase.
        """
        if 'approach' not in phase_info:
            raise ValueError("phase_info must contain 'approach' phase")

        approach_start = phase_info['approach']['start']
        approach_end = phase_info['approach']['end']

        # Extract heights during approach phase
        heights = []
        for idx in range(approach_start, min(approach_end + 1, len(skeleton_df))):
            landmarks = skeleton_df.iloc[idx]['landmarks_3d']

            if landmark_name == 'hip_center':
                position = self._calculate_hip_center(landmarks)
            else:
                if landmark_name not in self.LANDMARK_INDICES:
                    raise ValueError(f"Unknown landmark: {landmark_name}")
                landmark_idx = self.LANDMARK_INDICES[landmark_name]
                position = self._extract_landmark_position(landmarks, landmark_idx)

            heights.append(position[1])  # Y coordinate is height

        # Use median to avoid outliers
        baseline_height = float(np.median(heights))
        logger.info(f"Baseline height for {landmark_name}: {baseline_height:.3f} m")

        return baseline_height

    def calculate_peak_height(
        self,
        skeleton_df: pd.DataFrame,
        phase_info: dict,
        landmark_name: str = 'hip_center'
    ) -> Tuple[float, int]:
        """
        Calculate peak height during jump.

        Args:
            skeleton_df: DataFrame with 'landmarks_3d' column.
            phase_info: Dictionary containing phase boundaries.
            landmark_name: Name of landmark ('hip_center', 'right_wrist', etc.).

        Returns:
            Tuple of (peak_height, frame_index).

        Raises:
            ValueError: If phase_info doesn't contain required phases.
        """
        # Search for peak between takeoff and landing
        if 'takeoff' not in phase_info or 'landing' not in phase_info:
            raise ValueError("phase_info must contain 'takeoff' and 'landing' phases")

        search_start = phase_info['takeoff']['start']
        search_end = phase_info['landing']['end']

        heights = []
        frame_indices = []

        for idx in range(search_start, min(search_end + 1, len(skeleton_df))):
            landmarks = skeleton_df.iloc[idx]['landmarks_3d']

            if landmark_name == 'hip_center':
                position = self._calculate_hip_center(landmarks)
            else:
                if landmark_name not in self.LANDMARK_INDICES:
                    raise ValueError(f"Unknown landmark: {landmark_name}")
                landmark_idx = self.LANDMARK_INDICES[landmark_name]
                position = self._extract_landmark_position(landmarks, landmark_idx)

            heights.append(position[1])
            frame_indices.append(idx)

        if not heights:
            raise ValueError("No valid height data found")

        peak_idx = int(np.argmax(heights))
        peak_height = float(heights[peak_idx])
        peak_frame = frame_indices[peak_idx]

        logger.info(f"Peak height for {landmark_name}: {peak_height:.3f} m at frame {peak_frame}")

        return peak_height, peak_frame

    def calculate_jump_height_method1(
        self,
        skeleton_df: pd.DataFrame,
        phase_info: dict
    ) -> float:
        """
        Calculate jump height using hip displacement method.

        Args:
            skeleton_df: DataFrame with 'landmarks_3d' column.
            phase_info: Dictionary containing phase boundaries.

        Returns:
            Jump height in meters.
        """
        baseline = self.calculate_baseline_height(skeleton_df, phase_info, 'hip_center')
        peak, _ = self.calculate_peak_height(skeleton_df, phase_info, 'hip_center')

        jump_height = peak - baseline
        logger.info(f"Jump height (Method 1 - Hip displacement): {jump_height:.3f} m")

        return float(jump_height)

    def calculate_flight_time(
        self,
        skeleton_df: pd.DataFrame,
        phase_info: dict
    ) -> float:
        """
        Calculate flight time (time in air).

        Args:
            skeleton_df: DataFrame with 'time' column.
            phase_info: Dictionary containing phase boundaries.

        Returns:
            Flight time in seconds.

        Raises:
            ValueError: If phase_info doesn't contain required phases.
        """
        if 'takeoff' not in phase_info or 'landing' not in phase_info:
            raise ValueError("phase_info must contain 'takeoff' and 'landing' phases")

        takeoff_frame = phase_info['takeoff']['end']
        landing_frame = phase_info['landing']['start']

        takeoff_time = skeleton_df.iloc[takeoff_frame]['time']
        landing_time = skeleton_df.iloc[landing_frame]['time']

        flight_time = landing_time - takeoff_time
        logger.info(f"Flight time: {flight_time:.3f} s")

        return float(flight_time)

    def calculate_jump_height_method2(
        self,
        skeleton_df: pd.DataFrame,
        phase_info: dict
    ) -> float:
        """
        Calculate jump height using flight time method.

        Uses the physics formula: h = 0.5 * g * (t/2)²
        where t is the total flight time.

        Args:
            skeleton_df: DataFrame with 'time' column.
            phase_info: Dictionary containing phase boundaries.

        Returns:
            Jump height in meters.
        """
        flight_time = self.calculate_flight_time(skeleton_df, phase_info)

        # h = 0.5 * g * (t/2)²
        # Time to reach peak is half of flight time
        half_time = flight_time / 2
        jump_height = 0.5 * self.gravity * (half_time ** 2)

        logger.info(f"Jump height (Method 2 - Flight time): {jump_height:.3f} m")

        return float(jump_height)

    def calculate_jump_height(
        self,
        skeleton_df: pd.DataFrame,
        phase_info: dict
    ) -> dict:
        """
        Calculate jump height using multiple methods.

        Args:
            skeleton_df: DataFrame with columns ['time', 'landmarks_3d'].
            phase_info: Dictionary containing phase boundaries.

        Returns:
            Dictionary containing:
            {
                'method_1_hip_displacement': float,
                'method_2_flight_time': float,
                'flight_time': float,
                'recommended': float (average or more reliable method)
            }
        """
        result = {}

        try:
            result['method_1_hip_displacement'] = self.calculate_jump_height_method1(
                skeleton_df, phase_info
            )
        except Exception as e:
            logger.warning(f"Method 1 failed: {e}")
            result['method_1_hip_displacement'] = np.nan

        try:
            result['flight_time'] = self.calculate_flight_time(skeleton_df, phase_info)
            result['method_2_flight_time'] = self.calculate_jump_height_method2(
                skeleton_df, phase_info
            )
        except Exception as e:
            logger.warning(f"Method 2 failed: {e}")
            result['method_2_flight_time'] = np.nan
            result['flight_time'] = np.nan

        # Calculate recommended value (average of valid methods)
        valid_methods = [
            v for k, v in result.items()
            if k.startswith('method_') and not np.isnan(v)
        ]

        if valid_methods:
            result['recommended'] = float(np.mean(valid_methods))
        else:
            result['recommended'] = np.nan

        logger.info(f"Jump height calculation complete: recommended={result['recommended']:.3f} m")

        return result

    def calculate_contact_height(
        self,
        skeleton_df: pd.DataFrame,
        contact_frame: int,
        landmark_name: str = 'right_wrist'
    ) -> float:
        """
        Calculate contact point height at ball contact.

        Args:
            skeleton_df: DataFrame with 'landmarks_3d' column.
            contact_frame: Frame index at contact.
            landmark_name: Name of landmark (typically 'right_wrist').

        Returns:
            Contact height in meters.

        Raises:
            ValueError: If contact_frame is out of range or landmark unknown.
        """
        if contact_frame < 0 or contact_frame >= len(skeleton_df):
            raise ValueError(f"contact_frame {contact_frame} out of range [0, {len(skeleton_df)})")

        landmarks = skeleton_df.iloc[contact_frame]['landmarks_3d']

        if landmark_name not in self.LANDMARK_INDICES:
            raise ValueError(f"Unknown landmark: {landmark_name}")

        landmark_idx = self.LANDMARK_INDICES[landmark_name]
        position = self._extract_landmark_position(landmarks, landmark_idx)

        contact_height = float(position[1])
        logger.info(f"Contact height ({landmark_name}): {contact_height:.3f} m")

        return contact_height

    def calculate_horizontal_displacement(
        self,
        skeleton_df: pd.DataFrame,
        phase_info: dict,
        landmark_name: str = 'hip_center'
    ) -> dict:
        """
        Calculate horizontal displacement from takeoff to contact.

        Args:
            skeleton_df: DataFrame with 'landmarks_3d' column.
            phase_info: Dictionary containing phase boundaries.
            landmark_name: Name of landmark to track.

        Returns:
            Dictionary containing:
            {
                'forward_displacement': float (Z-axis),
                'lateral_displacement': float (X-axis),
                'total_displacement': float (magnitude)
            }

        Raises:
            ValueError: If phase_info doesn't contain required phases.
        """
        if 'takeoff' not in phase_info or 'contact' not in phase_info:
            raise ValueError("phase_info must contain 'takeoff' and 'contact' phases")

        takeoff_frame = phase_info['takeoff']['end']
        contact_frame = phase_info['contact']['start']

        # Get positions
        landmarks_takeoff = skeleton_df.iloc[takeoff_frame]['landmarks_3d']
        landmarks_contact = skeleton_df.iloc[contact_frame]['landmarks_3d']

        if landmark_name == 'hip_center':
            pos_takeoff = self._calculate_hip_center(landmarks_takeoff)
            pos_contact = self._calculate_hip_center(landmarks_contact)
        else:
            if landmark_name not in self.LANDMARK_INDICES:
                raise ValueError(f"Unknown landmark: {landmark_name}")
            landmark_idx = self.LANDMARK_INDICES[landmark_name]
            pos_takeoff = self._extract_landmark_position(landmarks_takeoff, landmark_idx)
            pos_contact = self._extract_landmark_position(landmarks_contact, landmark_idx)

        # Calculate displacements
        displacement = pos_contact - pos_takeoff

        result = {
            'lateral_displacement': float(displacement[0]),  # X-axis
            'forward_displacement': float(displacement[2]),  # Z-axis
            'total_displacement': float(np.linalg.norm(displacement[[0, 2]]))  # Horizontal magnitude
        }

        logger.info(f"Horizontal displacement: {result['total_displacement']:.3f} m")

        return result

    def calculate_center_of_mass_trajectory(
        self,
        skeleton_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculate center of mass trajectory over time.

        Approximates COM as hip center.

        Args:
            skeleton_df: DataFrame with 'landmarks_3d' column.

        Returns:
            Array of shape (num_frames, 3) containing COM positions.
        """
        com_trajectory = np.array([
            self._calculate_hip_center(landmarks)
            for landmarks in skeleton_df['landmarks_3d'].values
        ])

        logger.info(f"Calculated COM trajectory for {len(com_trajectory)} frames")

        return com_trajectory

    def calculate_spatial_profile(
        self,
        skeleton_df: pd.DataFrame,
        phase_info: dict
    ) -> dict:
        """
        Calculate all spatial metrics.

        Args:
            skeleton_df: DataFrame with columns ['time', 'landmarks_3d'].
            phase_info: Dictionary containing phase boundaries.

        Returns:
            Dictionary containing all spatial metrics:
            {
                'jump_height': {...},
                'contact_height': float,
                'horizontal_displacement': {...},
                'com_trajectory': np.ndarray
            }
        """
        result = {}

        # Calculate jump height
        try:
            result['jump_height'] = self.calculate_jump_height(skeleton_df, phase_info)
        except Exception as e:
            logger.error(f"Failed to calculate jump height: {e}")
            result['jump_height'] = {
                'method_1_hip_displacement': np.nan,
                'method_2_flight_time': np.nan,
                'flight_time': np.nan,
                'recommended': np.nan
            }

        # Calculate contact height
        try:
            contact_frame = phase_info['contact']['start']
            result['contact_height'] = self.calculate_contact_height(
                skeleton_df, contact_frame
            )
        except Exception as e:
            logger.error(f"Failed to calculate contact height: {e}")
            result['contact_height'] = np.nan

        # Calculate horizontal displacement
        try:
            result['horizontal_displacement'] = self.calculate_horizontal_displacement(
                skeleton_df, phase_info
            )
        except Exception as e:
            logger.error(f"Failed to calculate horizontal displacement: {e}")
            result['horizontal_displacement'] = {
                'lateral_displacement': np.nan,
                'forward_displacement': np.nan,
                'total_displacement': np.nan
            }

        # Calculate COM trajectory
        try:
            result['com_trajectory'] = self.calculate_center_of_mass_trajectory(skeleton_df)
        except Exception as e:
            logger.error(f"Failed to calculate COM trajectory: {e}")
            result['com_trajectory'] = None

        logger.info("Spatial profile calculation complete")

        return result
