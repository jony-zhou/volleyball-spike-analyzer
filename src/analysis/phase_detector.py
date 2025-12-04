"""
Phase detection module for volleyball spike motion analysis.

This module provides classes for detecting different phases of volleyball
spike motion, including full motion phases and detailed arm swing phases.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

logger = logging.getLogger(__name__)


class FullMotionPhaseDetector:
    """
    Detects five phases of complete volleyball spike motion.

    Phases:
        1. Approach: Running phase with high horizontal velocity
        2. Takeoff: Jump initiation with peak vertical acceleration
        3. Arm Swing: Arm movement from start to peak velocity
        4. Contact: Ball contact phase around peak hand velocity
        5. Landing: Descent and ground contact

    Attributes:
        min_approach_velocity: Minimum horizontal velocity for approach phase (m/s).
        contact_window: Time window around peak velocity for contact (seconds).
        smooth_window: Window size for Savitzky-Golay smoothing.
    """

    # MediaPipe landmark indices
    RIGHT_WRIST = 16
    RIGHT_ANKLE = 28
    RIGHT_HIP = 24
    LEFT_HIP = 23

    def __init__(
        self,
        min_approach_velocity: float = 1.0,
        contact_window: float = 0.1,
        smooth_window: int = 5
    ) -> None:
        """
        Initialize the FullMotionPhaseDetector.

        Args:
            min_approach_velocity: Minimum horizontal velocity threshold for approach (m/s).
            contact_window: Time window around peak velocity for contact phase (seconds).
            smooth_window: Window size for smoothing (must be odd and >= 3).

        Raises:
            ValueError: If parameters are invalid.
        """
        if min_approach_velocity <= 0:
            raise ValueError("min_approach_velocity must be positive")
        if contact_window <= 0:
            raise ValueError("contact_window must be positive")
        if smooth_window < 3 or smooth_window % 2 == 0:
            raise ValueError("smooth_window must be odd and >= 3")

        self.min_approach_velocity = min_approach_velocity
        self.contact_window = contact_window
        self.smooth_window = smooth_window

        logger.info(f"FullMotionPhaseDetector initialized with min_approach_velocity={min_approach_velocity}")

    def _extract_landmark_position(
        self,
        landmarks_3d: np.ndarray,
        landmark_idx: int
    ) -> np.ndarray:
        """
        Extract 3D position of a specific landmark.

        Args:
            landmarks_3d: Array of shape (33, 3 or 4).
            landmark_idx: Index of the landmark.

        Returns:
            Array of shape (3,) containing x, y, z coordinates.
        """
        return landmarks_3d[landmark_idx, :3]

    def _calculate_velocity(
        self,
        positions: np.ndarray,
        fps: float
    ) -> np.ndarray:
        """
        Calculate velocity from position time series.

        Args:
            positions: Array of shape (num_frames, 3).
            fps: Frames per second.

        Returns:
            Velocity array of shape (num_frames,).
        """
        # Calculate displacement
        displacement = np.gradient(positions, axis=0)
        # Calculate magnitude
        velocity = np.linalg.norm(displacement, axis=1) * fps
        return velocity

    def _calculate_acceleration(
        self,
        velocity: np.ndarray,
        fps: float
    ) -> np.ndarray:
        """
        Calculate acceleration from velocity time series.

        Args:
            velocity: Array of shape (num_frames,).
            fps: Frames per second.

        Returns:
            Acceleration array of shape (num_frames,).
        """
        acceleration = np.gradient(velocity) * fps
        return acceleration

    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to smooth signal.

        Args:
            signal: Input signal array.

        Returns:
            Smoothed signal.
        """
        if len(signal) < self.smooth_window:
            logger.warning("Signal too short for smoothing, returning original")
            return signal

        try:
            smoothed = savgol_filter(signal, self.smooth_window, polyorder=2)
            return smoothed
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}, returning original signal")
            return signal

    def detect_phases(
        self,
        skeleton_df: pd.DataFrame
    ) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Detect all five phases of volleyball spike motion.

        Args:
            skeleton_df: DataFrame with columns ['frame', 'time', 'landmarks_3d'].
                landmarks_3d should be np.ndarray of shape (33, 3 or 4).

        Returns:
            Dictionary containing phase boundaries:
            {
                'approach': {'start': frame_idx, 'end': frame_idx},
                'takeoff': {'start': frame_idx, 'end': frame_idx},
                'arm_swing': {'start': frame_idx, 'end': frame_idx},
                'contact': {'start': frame_idx, 'end': frame_idx},
                'landing': {'start': frame_idx, 'end': frame_idx}
            }
            Returns None if detection fails.

        Raises:
            ValueError: If input DataFrame is invalid.
        """
        # Validate input
        required_columns = ['frame', 'time', 'landmarks_3d']
        if not all(col in skeleton_df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        if len(skeleton_df) < self.smooth_window:
            logger.warning(f"Sequence too short ({len(skeleton_df)} frames)")
            return None

        # Calculate FPS
        time_diff = skeleton_df['time'].diff().mean()
        fps = 1.0 / time_diff if time_diff > 0 else 30.0

        num_frames = len(skeleton_df)

        # Extract key landmarks over time
        wrist_positions = np.array([
            self._extract_landmark_position(lm, self.RIGHT_WRIST)
            for lm in skeleton_df['landmarks_3d'].values
        ])

        ankle_positions = np.array([
            self._extract_landmark_position(lm, self.RIGHT_ANKLE)
            for lm in skeleton_df['landmarks_3d'].values
        ])

        hip_positions = np.array([
            (self._extract_landmark_position(lm, self.RIGHT_HIP) +
             self._extract_landmark_position(lm, self.LEFT_HIP)) / 2
            for lm in skeleton_df['landmarks_3d'].values
        ])

        # Calculate velocities and accelerations
        wrist_velocity = self._calculate_velocity(wrist_positions, fps)
        ankle_horizontal_velocity = self._calculate_velocity(ankle_positions[:, [0, 2]], fps)
        hip_vertical_velocity = np.gradient(hip_positions[:, 1]) * fps
        hip_vertical_acceleration = self._calculate_acceleration(hip_vertical_velocity, fps)

        # Smooth signals
        wrist_velocity_smooth = self._smooth_signal(wrist_velocity)
        ankle_h_vel_smooth = self._smooth_signal(ankle_horizontal_velocity)
        hip_v_accel_smooth = self._smooth_signal(hip_vertical_acceleration)
        wrist_height_smooth = self._smooth_signal(wrist_positions[:, 1])
        ankle_height_smooth = self._smooth_signal(ankle_positions[:, 1])

        try:
            # 1. Detect APPROACH phase
            # Start: when ankle horizontal velocity exceeds threshold
            approach_start = 0
            approach_candidates = np.where(ankle_h_vel_smooth > self.min_approach_velocity)[0]
            if len(approach_candidates) > 0:
                approach_start = approach_candidates[0]

            # 2. Detect TAKEOFF phase
            # Start: peak vertical acceleration of hip
            # End: when ankle leaves ground (height increases significantly)
            takeoff_start = approach_start
            accel_peaks, _ = find_peaks(hip_v_accel_smooth, distance=int(fps * 0.2))
            if len(accel_peaks) > 0:
                # Find the first major peak after approach
                valid_peaks = accel_peaks[accel_peaks > approach_start]
                if len(valid_peaks) > 0:
                    takeoff_start = valid_peaks[0]

            # Ankle leaves ground when height increases
            ankle_baseline = np.median(ankle_height_smooth[:max(10, num_frames // 10)])
            ankle_off_ground = np.where(ankle_height_smooth > ankle_baseline * 1.1)[0]
            takeoff_end = takeoff_start + int(fps * 0.3)  # Default 0.3s
            if len(ankle_off_ground) > 0:
                candidates = ankle_off_ground[ankle_off_ground > takeoff_start]
                if len(candidates) > 0:
                    takeoff_end = min(candidates[0], num_frames - 1)

            # 3. Detect ARM SWING phase
            # Start: wrist starts rising (after takeoff)
            # End: wrist reaches peak velocity
            arm_swing_start = takeoff_end
            wrist_rising = np.where(np.gradient(wrist_height_smooth) > 0)[0]
            if len(wrist_rising) > 0:
                candidates = wrist_rising[wrist_rising >= takeoff_end]
                if len(candidates) > 0:
                    arm_swing_start = candidates[0]

            # Find peak wrist velocity
            wrist_vel_peaks, _ = find_peaks(wrist_velocity_smooth, distance=int(fps * 0.1))
            arm_swing_end = arm_swing_start + int(fps * 0.5)  # Default 0.5s
            if len(wrist_vel_peaks) > 0:
                candidates = wrist_vel_peaks[wrist_vel_peaks > arm_swing_start]
                if len(candidates) > 0:
                    arm_swing_end = candidates[0]

            # 4. Detect CONTACT phase
            # Around the peak wrist velocity (±contact_window)
            contact_center = arm_swing_end
            contact_frames = int(self.contact_window * fps)
            contact_start = max(0, contact_center - contact_frames)
            contact_end = min(num_frames - 1, contact_center + contact_frames)

            # 5. Detect LANDING phase
            # Start: after contact
            # End: ankle returns to baseline height
            landing_start = contact_end
            ankle_baseline_height = np.median(ankle_height_smooth[:max(10, num_frames // 10)])
            landing_candidates = np.where(
                (ankle_height_smooth <= ankle_baseline_height * 1.1) &
                (np.arange(num_frames) > landing_start)
            )[0]
            landing_end = num_frames - 1
            if len(landing_candidates) > 0:
                landing_end = landing_candidates[0]

            # Construct result
            phases = {
                'approach': {'start': int(approach_start), 'end': int(takeoff_start)},
                'takeoff': {'start': int(takeoff_start), 'end': int(takeoff_end)},
                'arm_swing': {'start': int(arm_swing_start), 'end': int(arm_swing_end)},
                'contact': {'start': int(contact_start), 'end': int(contact_end)},
                'landing': {'start': int(landing_start), 'end': int(landing_end)}
            }

            logger.info(f"Successfully detected all 5 phases")
            return phases

        except Exception as e:
            logger.error(f"Phase detection failed: {e}", exc_info=True)
            return None


class ArmSwingPhaseDetector:
    """
    Detects three sub-phases within the arm swing phase.

    Based on research literature, the arm swing is divided into:
        - Phase I (Initiation): Wrist and elbow start rising
        - Phase II (Wind-up): Wrist reaches maximum height
        - Phase III (Final Cocking): Wrist descends and accelerates

    Attributes:
        smooth_window: Window size for Savitzky-Golay smoothing.
        peak_prominence: Minimum prominence for peak detection.
    """

    # MediaPipe landmark indices
    RIGHT_WRIST = 16
    RIGHT_ELBOW = 14

    def __init__(
        self,
        smooth_window: int = 5,
        peak_prominence: float = 0.05
    ) -> None:
        """
        Initialize the ArmSwingPhaseDetector.

        Args:
            smooth_window: Window size for smoothing (must be odd and >= 3).
            peak_prominence: Minimum prominence for peak detection.

        Raises:
            ValueError: If parameters are invalid.
        """
        if smooth_window < 3 or smooth_window % 2 == 0:
            raise ValueError("smooth_window must be odd and >= 3")
        if peak_prominence <= 0:
            raise ValueError("peak_prominence must be positive")

        self.smooth_window = smooth_window
        self.peak_prominence = peak_prominence

        logger.info(f"ArmSwingPhaseDetector initialized")

    def _extract_landmark_position(
        self,
        landmarks_3d: np.ndarray,
        landmark_idx: int
    ) -> np.ndarray:
        """
        Extract 3D position of a specific landmark.

        Args:
            landmarks_3d: Array of shape (33, 3 or 4).
            landmark_idx: Index of the landmark.

        Returns:
            Array of shape (3,) containing x, y, z coordinates.
        """
        return landmarks_3d[landmark_idx, :3]

    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to smooth signal.

        Args:
            signal: Input signal array.

        Returns:
            Smoothed signal.
        """
        if len(signal) < self.smooth_window:
            logger.warning("Signal too short for smoothing, returning original")
            return signal

        try:
            smoothed = savgol_filter(signal, self.smooth_window, polyorder=2)
            return smoothed
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}, returning original signal")
            return signal

    def detect_sub_phases(
        self,
        skeleton_df: pd.DataFrame,
        arm_swing_start_frame: int,
        arm_swing_end_frame: int
    ) -> Optional[Dict[str, Dict[str, int]]]:
        """
        Detect three sub-phases within the arm swing phase.

        Args:
            skeleton_df: DataFrame with columns ['frame', 'time', 'landmarks_3d'].
            arm_swing_start_frame: Start frame of arm swing phase.
            arm_swing_end_frame: End frame of arm swing phase.

        Returns:
            Dictionary containing sub-phase boundaries:
            {
                'phase_i': {'start': frame_idx, 'end': frame_idx},
                'phase_ii': {'start': frame_idx, 'end': frame_idx},
                'phase_iii': {'start': frame_idx, 'end': frame_idx}
            }
            Returns None if detection fails.

        Raises:
            ValueError: If input is invalid.
        """
        # Validate input
        required_columns = ['frame', 'time', 'landmarks_3d']
        if not all(col in skeleton_df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        if arm_swing_start_frame >= arm_swing_end_frame:
            raise ValueError("arm_swing_start_frame must be less than arm_swing_end_frame")

        if arm_swing_end_frame >= len(skeleton_df):
            raise ValueError("arm_swing_end_frame exceeds DataFrame length")

        # Extract arm swing segment
        segment_df = skeleton_df.iloc[arm_swing_start_frame:arm_swing_end_frame + 1].copy()
        segment_df.reset_index(drop=True, inplace=True)

        if len(segment_df) < self.smooth_window:
            logger.warning(f"Arm swing segment too short ({len(segment_df)} frames)")
            return None

        try:
            # Calculate FPS
            time_diff = segment_df['time'].diff().mean()
            fps = 1.0 / time_diff if time_diff > 0 else 30.0

            # Extract wrist positions
            wrist_positions = np.array([
                self._extract_landmark_position(lm, self.RIGHT_WRIST)
                for lm in segment_df['landmarks_3d'].values
            ])

            # Extract Y coordinate (height) and smooth
            wrist_height = wrist_positions[:, 1]
            wrist_height_smooth = self._smooth_signal(wrist_height)

            # Calculate velocity (first derivative)
            wrist_velocity = np.gradient(wrist_height_smooth)
            wrist_velocity_smooth = self._smooth_signal(wrist_velocity)

            # Calculate acceleration (second derivative)
            wrist_acceleration = np.gradient(wrist_velocity_smooth)
            wrist_acceleration_smooth = self._smooth_signal(wrist_acceleration)

            # Phase I (Initiation): wrist starts rising
            # Start: first frame (beginning of arm swing)
            # End: when wrist velocity becomes consistently positive
            phase_i_start = 0

            # Find when velocity becomes positive
            positive_vel = np.where(wrist_velocity_smooth > 0)[0]
            phase_i_end = len(segment_df) // 3  # Default: first third
            if len(positive_vel) > 0:
                phase_i_end = positive_vel[0] if positive_vel[0] > 0 else phase_i_end

            # Phase II (Wind-up): wrist reaches maximum height
            # Start: end of Phase I
            # End: peak height (local maximum)
            phase_ii_start = phase_i_end

            # Find peak height
            peaks, properties = find_peaks(
                wrist_height_smooth,
                prominence=self.peak_prominence,
                distance=int(fps * 0.1)
            )

            phase_ii_end = len(segment_df) * 2 // 3  # Default: two thirds
            if len(peaks) > 0:
                # Take the most prominent peak
                if 'prominences' in properties:
                    most_prominent = peaks[np.argmax(properties['prominences'])]
                else:
                    most_prominent = peaks[0]
                phase_ii_end = most_prominent

            # Phase III (Final Cocking): wrist descends and accelerates
            # Start: end of Phase II (peak height)
            # End: end of arm swing phase
            phase_iii_start = phase_ii_end
            phase_iii_end = len(segment_df) - 1

            # Convert to absolute frame indices
            phases = {
                'phase_i': {
                    'start': int(arm_swing_start_frame + phase_i_start),
                    'end': int(arm_swing_start_frame + phase_i_end)
                },
                'phase_ii': {
                    'start': int(arm_swing_start_frame + phase_ii_start),
                    'end': int(arm_swing_start_frame + phase_ii_end)
                },
                'phase_iii': {
                    'start': int(arm_swing_start_frame + phase_iii_start),
                    'end': int(arm_swing_start_frame + phase_iii_end)
                }
            }

            logger.info(f"Successfully detected 3 arm swing sub-phases")
            return phases

        except Exception as e:
            logger.error(f"Sub-phase detection failed: {e}", exc_info=True)
            return None
