"""
Velocity calculation module for volleyball spike analysis.

This module provides functionality to calculate linear velocities,
angular velocities, and accelerations for biomechanical analysis.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


class VelocityCalculator:
    """
    Calculate velocity and acceleration metrics for spike analysis.

    This class provides methods to calculate:
    - Linear velocities of key joints (wrist, elbow, shoulder)
    - Angular velocities of joints
    - Linear accelerations

    Attributes:
        smooth_window: Window size for Savitzky-Golay smoothing.
        polyorder: Polynomial order for smoothing filter.
    """

    # MediaPipe landmark indices
    LANDMARK_INDICES = {
        'right_wrist': 16,
        'right_elbow': 14,
        'right_shoulder': 12,
        'left_wrist': 15,
        'left_elbow': 13,
        'left_shoulder': 11,
        'right_hip': 24,
        'left_hip': 23
    }

    def __init__(
        self,
        smooth_window: int = 5,
        polyorder: int = 2
    ) -> None:
        """
        Initialize the VelocityCalculator.

        Args:
            smooth_window: Window size for smoothing (must be odd and >= 3).
            polyorder: Polynomial order for Savitzky-Golay filter (must be < smooth_window).

        Raises:
            ValueError: If parameters are invalid.
        """
        if smooth_window < 3 or smooth_window % 2 == 0:
            raise ValueError("smooth_window must be odd and >= 3")
        if polyorder >= smooth_window:
            raise ValueError("polyorder must be less than smooth_window")
        if polyorder < 0:
            raise ValueError("polyorder must be non-negative")

        self.smooth_window = smooth_window
        self.polyorder = polyorder

        logger.info(f"VelocityCalculator initialized with smooth_window={smooth_window}, polyorder={polyorder}")

    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to smooth signal.

        Args:
            signal: Input signal array.

        Returns:
            Smoothed signal.
        """
        if len(signal) < self.smooth_window:
            logger.warning(f"Signal length {len(signal)} < smooth_window {self.smooth_window}, returning original")
            return signal

        try:
            smoothed = savgol_filter(signal, self.smooth_window, self.polyorder)
            return smoothed
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}, returning original signal")
            return signal

    def calculate_linear_velocity(
        self,
        positions: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        """
        Calculate 3D linear velocity from position data.

        Args:
            positions: Array of shape (num_frames, 3) containing x, y, z positions.
            times: Array of shape (num_frames,) containing timestamps.

        Returns:
            Array of shape (num_frames,) containing velocity magnitudes in m/s.

        Raises:
            ValueError: If input shapes are invalid.
        """
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (n, 3), got {positions.shape}")
        if times.ndim != 1:
            raise ValueError(f"times must be 1D array, got shape {times.shape}")
        if len(positions) != len(times):
            raise ValueError(f"positions and times must have same length")

        # Calculate velocity components using gradient
        velocity_components = np.gradient(positions, times, axis=0)

        # Calculate velocity magnitude
        velocity_magnitude = np.linalg.norm(velocity_components, axis=1)

        # Smooth the velocity signal
        velocity_smooth = self._smooth_signal(velocity_magnitude)

        return velocity_smooth

    def calculate_linear_acceleration(
        self,
        velocities: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        """
        Calculate linear acceleration from velocity data.

        Args:
            velocities: Array of shape (num_frames,) containing velocity magnitudes.
            times: Array of shape (num_frames,) containing timestamps.

        Returns:
            Array of shape (num_frames,) containing acceleration magnitudes in m/s².

        Raises:
            ValueError: If input shapes are invalid.
        """
        if velocities.ndim != 1:
            raise ValueError(f"velocities must be 1D array, got shape {velocities.shape}")
        if times.ndim != 1:
            raise ValueError(f"times must be 1D array, got shape {times.shape}")
        if len(velocities) != len(times):
            raise ValueError(f"velocities and times must have same length")

        # Calculate acceleration using gradient
        acceleration = np.gradient(velocities, times)

        # Smooth the acceleration signal
        acceleration_smooth = self._smooth_signal(acceleration)

        return acceleration_smooth

    def calculate_angular_velocity(
        self,
        angles: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        """
        Calculate angular velocity from angle data.

        Args:
            angles: Array of shape (num_frames,) containing angles in degrees.
            times: Array of shape (num_frames,) containing timestamps.

        Returns:
            Array of shape (num_frames,) containing angular velocities in deg/s.

        Raises:
            ValueError: If input shapes are invalid.
        """
        if angles.ndim != 1:
            raise ValueError(f"angles must be 1D array, got shape {angles.shape}")
        if times.ndim != 1:
            raise ValueError(f"times must be 1D array, got shape {times.shape}")
        if len(angles) != len(times):
            raise ValueError(f"angles and times must have same length")

        # Calculate angular velocity using gradient
        angular_velocity = np.gradient(angles, times)

        # Smooth the angular velocity signal
        angular_velocity_smooth = self._smooth_signal(angular_velocity)

        return angular_velocity_smooth

    def _extract_landmark_positions(
        self,
        skeleton_df: pd.DataFrame,
        landmark_name: str
    ) -> np.ndarray:
        """
        Extract positions of a specific landmark over time.

        Args:
            skeleton_df: DataFrame with 'landmarks_3d' column.
            landmark_name: Name of the landmark (e.g., 'right_wrist').

        Returns:
            Array of shape (num_frames, 3) containing x, y, z positions.

        Raises:
            KeyError: If landmark name is invalid.
        """
        if landmark_name not in self.LANDMARK_INDICES:
            raise KeyError(f"Unknown landmark name: {landmark_name}")

        landmark_idx = self.LANDMARK_INDICES[landmark_name]
        positions = np.array([
            landmarks[landmark_idx, :3]
            for landmarks in skeleton_df['landmarks_3d'].values
        ])

        return positions

    def analyze_velocity_profile(
        self,
        skeleton_df: pd.DataFrame,
        angles_df: Optional[pd.DataFrame] = None,
        phase_info: Optional[dict] = None
    ) -> dict:
        """
        Analyze velocity profile for the entire motion.

        Args:
            skeleton_df: DataFrame with columns ['frame', 'time', 'landmarks_3d'].
            angles_df: Optional DataFrame with joint angles.
            phase_info: Optional dictionary of phase boundaries.

        Returns:
            Dictionary containing velocity metrics:
            {
                'wrist_velocity': {
                    'values': np.ndarray,
                    'max': float,
                    'mean': float,
                    'at_contact': float (if phase_info provided)
                },
                'elbow_velocity': {...},
                'shoulder_velocity': {...},
                'wrist_acceleration': {...},
                'elbow_acceleration': {...},
                'shoulder_angular_velocity': {...} (if angles_df provided),
                'elbow_angular_velocity': {...} (if angles_df provided)
            }

        Raises:
            ValueError: If input DataFrame is invalid.
        """
        if 'time' not in skeleton_df.columns or 'landmarks_3d' not in skeleton_df.columns:
            raise ValueError("skeleton_df must contain 'time' and 'landmarks_3d' columns")

        times = skeleton_df['time'].values
        result = {}

        # Calculate linear velocities for key joints
        joints = ['right_wrist', 'right_elbow', 'right_shoulder']
        joint_display_names = {
            'right_wrist': 'wrist_velocity',
            'right_elbow': 'elbow_velocity',
            'right_shoulder': 'shoulder_velocity'
        }

        for joint in joints:
            try:
                positions = self._extract_landmark_positions(skeleton_df, joint)
                velocities = self.calculate_linear_velocity(positions, times)

                display_name = joint_display_names[joint]
                result[display_name] = {
                    'values': velocities,
                    'max': float(np.max(velocities)),
                    'mean': float(np.mean(velocities)),
                    'std': float(np.std(velocities))
                }

                # Add velocity at contact if phase info provided
                if phase_info and 'contact' in phase_info:
                    contact_frame = phase_info['contact']['start']
                    if 0 <= contact_frame < len(velocities):
                        result[display_name]['at_contact'] = float(velocities[contact_frame])

                # Calculate acceleration for wrist and elbow
                if joint in ['right_wrist', 'right_elbow']:
                    accelerations = self.calculate_linear_acceleration(velocities, times)
                    accel_name = display_name.replace('velocity', 'acceleration')
                    result[accel_name] = {
                        'values': accelerations,
                        'max': float(np.max(np.abs(accelerations))),
                        'mean': float(np.mean(np.abs(accelerations))),
                        'std': float(np.std(accelerations))
                    }

            except Exception as e:
                logger.warning(f"Could not calculate velocity for {joint}: {e}")

        # Calculate angular velocities if angles provided
        if angles_df is not None:
            angle_columns = {
                'shoulder_abduction': 'shoulder_angular_velocity',
                'elbow_flexion': 'elbow_angular_velocity'
            }

            for angle_col, result_name in angle_columns.items():
                if angle_col in angles_df.columns:
                    try:
                        angles = angles_df[angle_col].values
                        # Remove NaN values
                        valid_mask = ~np.isnan(angles)
                        if np.sum(valid_mask) > self.smooth_window:
                            angles_valid = angles[valid_mask]
                            times_valid = angles_df['time'].values[valid_mask]

                            angular_vel = self.calculate_angular_velocity(angles_valid, times_valid)

                            # Create full array with NaN for invalid frames
                            angular_vel_full = np.full_like(angles, np.nan)
                            angular_vel_full[valid_mask] = angular_vel

                            result[result_name] = {
                                'values': angular_vel_full,
                                'max': float(np.nanmax(np.abs(angular_vel))),
                                'mean': float(np.nanmean(np.abs(angular_vel))),
                                'std': float(np.nanstd(angular_vel))
                            }

                            # Add angular velocity at contact
                            if phase_info and 'contact' in phase_info:
                                contact_frame = phase_info['contact']['start']
                                if 0 <= contact_frame < len(angular_vel_full):
                                    contact_value = angular_vel_full[contact_frame]
                                    if not np.isnan(contact_value):
                                        result[result_name]['at_contact'] = float(contact_value)

                    except Exception as e:
                        logger.warning(f"Could not calculate angular velocity for {angle_col}: {e}")

        logger.info(f"Velocity analysis complete with {len(result)} metrics")
        return result

    def get_peak_velocity_frame(
        self,
        skeleton_df: pd.DataFrame,
        landmark_name: str = 'right_wrist'
    ) -> int:
        """
        Find the frame with peak velocity for a specific landmark.

        Args:
            skeleton_df: DataFrame with columns ['frame', 'time', 'landmarks_3d'].
            landmark_name: Name of the landmark to analyze.

        Returns:
            Frame index with peak velocity.

        Raises:
            ValueError: If no valid velocity data found.
        """
        times = skeleton_df['time'].values
        positions = self._extract_landmark_positions(skeleton_df, landmark_name)
        velocities = self.calculate_linear_velocity(positions, times)

        if len(velocities) == 0:
            raise ValueError("No velocity data available")

        peak_frame = int(np.argmax(velocities))
        logger.info(f"Peak {landmark_name} velocity at frame {peak_frame}: {velocities[peak_frame]:.2f} m/s")

        return peak_frame

    def calculate_velocity_timeseries(
        self,
        skeleton_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate velocities for all key joints and return as DataFrame.

        Args:
            skeleton_df: DataFrame with columns ['frame', 'time', 'landmarks_3d'].

        Returns:
            DataFrame with columns:
            ['frame', 'time', 'wrist_velocity', 'elbow_velocity',
             'shoulder_velocity', 'wrist_acceleration', 'elbow_acceleration']

        Raises:
            ValueError: If input DataFrame is invalid.
        """
        if 'frame' not in skeleton_df.columns or 'time' not in skeleton_df.columns:
            raise ValueError("skeleton_df must contain 'frame' and 'time' columns")

        times = skeleton_df['time'].values
        result_data = {
            'frame': skeleton_df['frame'].values,
            'time': times
        }

        # Calculate velocities for each joint
        joints = {
            'right_wrist': 'wrist_velocity',
            'right_elbow': 'elbow_velocity',
            'right_shoulder': 'shoulder_velocity'
        }

        for joint, col_name in joints.items():
            try:
                positions = self._extract_landmark_positions(skeleton_df, joint)
                velocities = self.calculate_linear_velocity(positions, times)
                result_data[col_name] = velocities

                # Calculate acceleration for wrist and elbow
                if joint in ['right_wrist', 'right_elbow']:
                    accelerations = self.calculate_linear_acceleration(velocities, times)
                    accel_col = col_name.replace('velocity', 'acceleration')
                    result_data[accel_col] = accelerations

            except Exception as e:
                logger.warning(f"Could not calculate velocity for {joint}: {e}")
                result_data[col_name] = np.full(len(times), np.nan)

        velocity_df = pd.DataFrame(result_data)
        logger.info(f"Calculated velocity timeseries for {len(velocity_df)} frames")

        return velocity_df
