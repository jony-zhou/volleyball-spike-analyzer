"""
Arm swing classification module for volleyball spike analysis.

This module provides functionality to classify arm swing motions into
five types based on biomechanical characteristics defined in research literature.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ArmSwingClassifier:
    """
    Classify arm swing motion into five types.

    Based on research literature, arm swing motions are classified into:
    1. Straight: High arc with stopping motion
    2. Bow-Arrow High (BA-High): Very high arc with stopping motion
    3. Bow-Arrow Low (BA-Low): Medium arc with stopping motion
    4. Snap: Horizontal motion with stopping motion
    5. Circular: Low continuous motion without stopping

    Attributes:
        height_threshold: Threshold for height comparison (meters).
        velocity_threshold: Threshold for detecting stopping motion (fraction of peak).
        stopping_duration: Minimum frames for stopping motion detection.
    """

    # MediaPipe landmark indices
    LANDMARK_INDICES = {
        'nose': 0,
        'right_wrist': 16,
        'right_elbow': 14,
        'right_shoulder': 12,
        'left_shoulder': 11,
        'right_hip': 24,
        'left_hip': 23
    }

    # Classification types
    MOTION_TYPES = {
        'STRAIGHT': 'Straight',
        'BA_HIGH': 'Bow-Arrow High',
        'BA_LOW': 'Bow-Arrow Low',
        'SNAP': 'Snap',
        'CIRCULAR': 'Circular'
    }

    def __init__(
        self,
        height_threshold: float = 0.05,
        velocity_threshold: float = 0.2,
        stopping_duration: int = 3
    ) -> None:
        """
        Initialize the ArmSwingClassifier.

        Args:
            height_threshold: Threshold for height comparison (meters).
            velocity_threshold: Velocity threshold for stopping detection (fraction of peak).
            stopping_duration: Minimum frames for stopping motion.

        Raises:
            ValueError: If parameters are invalid.
        """
        if height_threshold <= 0:
            raise ValueError("height_threshold must be positive")
        if not 0 < velocity_threshold < 1:
            raise ValueError("velocity_threshold must be between 0 and 1")
        if stopping_duration < 1:
            raise ValueError("stopping_duration must be at least 1")

        self.height_threshold = height_threshold
        self.velocity_threshold = velocity_threshold
        self.stopping_duration = stopping_duration

        logger.info(f"ArmSwingClassifier initialized")

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

    def _calculate_forehead_position(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Approximate forehead position.

        Uses weighted average of nose and shoulder midpoint.

        Args:
            landmarks: Array of shape (33, 3 or 4).

        Returns:
            Array of shape (3,) containing forehead position.
        """
        nose = self._extract_landmark_position(landmarks, self.LANDMARK_INDICES['nose'])
        right_shoulder = self._extract_landmark_position(landmarks, self.LANDMARK_INDICES['right_shoulder'])
        left_shoulder = self._extract_landmark_position(landmarks, self.LANDMARK_INDICES['left_shoulder'])

        shoulder_mid = (right_shoulder + left_shoulder) / 2

        # Forehead is approximately 70% from shoulder_mid to nose
        forehead = shoulder_mid * 0.3 + nose * 0.7

        return forehead

    def _check_height_relative_to_landmark(
        self,
        point: np.ndarray,
        reference: np.ndarray,
        threshold: Optional[float] = None
    ) -> str:
        """
        Check height of point relative to reference landmark.

        Args:
            point: Point position (x, y, z).
            reference: Reference position (x, y, z).
            threshold: Height threshold (uses instance default if None).

        Returns:
            'above', 'at', or 'below'.
        """
        if threshold is None:
            threshold = self.height_threshold

        height_diff = point[1] - reference[1]  # Y-axis is height

        if height_diff > threshold:
            return 'above'
        elif height_diff < -threshold:
            return 'below'
        else:
            return 'at'

    def _detect_stopping_motion(
        self,
        velocity_data: np.ndarray,
        phase_iii_frames: Tuple[int, int]
    ) -> bool:
        """
        Detect if there's stopping motion in Phase III.

        Stopping is defined as velocity dropping below threshold
        for a sustained period.

        Args:
            velocity_data: Array of velocity values.
            phase_iii_frames: Tuple of (start_frame, end_frame) for Phase III.

        Returns:
            True if stopping motion detected, False otherwise.
        """
        start_frame, end_frame = phase_iii_frames

        if start_frame >= len(velocity_data) or end_frame > len(velocity_data):
            logger.warning(f"Phase III frames out of range")
            return False

        # Get velocity in Phase III
        phase_velocity = velocity_data[start_frame:end_frame + 1]

        if len(phase_velocity) == 0:
            return False

        # Find peak velocity in entire sequence
        peak_velocity = np.max(velocity_data)

        if peak_velocity < 1e-6:
            return False

        # Check if velocity drops below threshold
        threshold_velocity = peak_velocity * self.velocity_threshold
        below_threshold = phase_velocity < threshold_velocity

        # Count consecutive frames below threshold
        max_consecutive = 0
        current_consecutive = 0

        for is_below in below_threshold:
            if is_below:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        has_stopping = max_consecutive >= self.stopping_duration

        logger.debug(f"Stopping detection: max_consecutive={max_consecutive}, has_stopping={has_stopping}")

        return has_stopping

    def _analyze_phase_characteristics(
        self,
        skeleton_df: pd.DataFrame,
        arm_swing_phases: dict
    ) -> dict:
        """
        Analyze characteristics of each arm swing phase.

        Args:
            skeleton_df: DataFrame with skeleton data.
            arm_swing_phases: Dictionary with phase boundaries.

        Returns:
            Dictionary with phase characteristics.
        """
        characteristics = {}

        for phase_name in ['phase_i', 'phase_ii', 'phase_iii']:
            if phase_name not in arm_swing_phases:
                continue

            start_frame = arm_swing_phases[phase_name]['start']
            end_frame = arm_swing_phases[phase_name]['end']

            # Get middle frame of the phase
            mid_frame = (start_frame + end_frame) // 2

            if mid_frame >= len(skeleton_df):
                continue

            landmarks = skeleton_df.iloc[mid_frame]['landmarks_3d']

            # Extract key landmarks
            wrist = self._extract_landmark_position(landmarks, self.LANDMARK_INDICES['right_wrist'])
            elbow = self._extract_landmark_position(landmarks, self.LANDMARK_INDICES['right_elbow'])
            shoulder = self._extract_landmark_position(landmarks, self.LANDMARK_INDICES['right_shoulder'])
            forehead = self._calculate_forehead_position(landmarks)

            # Analyze heights
            wrist_vs_shoulder = self._check_height_relative_to_landmark(wrist, shoulder)
            wrist_vs_forehead = self._check_height_relative_to_landmark(wrist, forehead)
            elbow_vs_shoulder = self._check_height_relative_to_landmark(elbow, shoulder)

            # Check if wrist is between forehead and shoulder
            wrist_between = (wrist[1] < forehead[1] and wrist[1] > shoulder[1])

            # Check if wrist is below elbow
            wrist_below_elbow = wrist[1] < elbow[1]

            characteristics[phase_name] = {
                'wrist_vs_shoulder': wrist_vs_shoulder,
                'wrist_vs_forehead': wrist_vs_forehead,
                'elbow_vs_shoulder': elbow_vs_shoulder,
                'wrist_between_forehead_shoulder': wrist_between,
                'wrist_below_elbow': wrist_below_elbow
            }

        return characteristics

    def _classify_by_rules(
        self,
        characteristics: dict,
        has_stopping: bool
    ) -> Tuple[str, float, List[str]]:
        """
        Classify motion type based on rules.

        Args:
            characteristics: Phase characteristics.
            has_stopping: Whether stopping motion is detected.

        Returns:
            Tuple of (motion_type, confidence, matched_rules).
        """
        matched_rules = []
        confidence_scores = {}

        # Get phase characteristics
        phase_i = characteristics.get('phase_i', {})
        phase_ii = characteristics.get('phase_ii', {})
        phase_iii = characteristics.get('phase_iii', {})

        # Rule 1: Circular - Continuous motion, low position
        if not has_stopping:
            matched_rules.append("Continuous motion without stopping")
            confidence = 0.7

            # Additional checks for Circular
            if (phase_i.get('wrist_vs_shoulder') in ['at', 'below'] and
                phase_ii.get('wrist_below_elbow', False)):
                matched_rules.append("Phase I: Wrist at/below shoulder")
                matched_rules.append("Phase II: Wrist moves down below elbow")
                confidence += 0.2

            if (phase_iii.get('wrist_vs_shoulder') in ['at', 'below'] and
                phase_iii.get('elbow_vs_shoulder') in ['at', 'below']):
                matched_rules.append("Phase III: Both wrist and elbow at/below shoulder")
                confidence += 0.1

            confidence_scores['CIRCULAR'] = min(1.0, confidence)

        # Rules for motions with stopping
        if has_stopping:
            matched_rules.append("Stopping motion detected in Phase III")

            # Rule 2: Straight - High position with stopping
            if (phase_i.get('wrist_vs_shoulder') == 'above' and
                phase_i.get('elbow_vs_shoulder') == 'above' and
                phase_ii.get('wrist_vs_forehead') == 'above' and
                phase_iii.get('elbow_vs_shoulder') == 'above'):

                confidence_scores['STRAIGHT'] = 0.85
                if 'STRAIGHT' not in matched_rules:
                    matched_rules.append("Phase I: Wrist and elbow above shoulder")
                    matched_rules.append("Phase II: Wrist above forehead")
                    matched_rules.append("Phase III: Elbow above shoulder")

            # Rule 3: BA-High - Very high position with stopping
            if (phase_i.get('wrist_vs_shoulder') == 'above' and
                phase_i.get('elbow_vs_shoulder') == 'above' and
                phase_ii.get('wrist_vs_forehead') == 'above' and
                phase_iii.get('wrist_vs_forehead') == 'above'):

                confidence_scores['BA_HIGH'] = 0.90
                if 'BA_HIGH' not in matched_rules:
                    matched_rules.append("Phase I: Wrist and elbow above shoulder")
                    matched_rules.append("Phase II & III: Wrist consistently above forehead")

            # Rule 4: BA-Low - Medium position with stopping
            if (phase_i.get('wrist_vs_shoulder') == 'above' and
                phase_ii.get('wrist_between_forehead_shoulder', False) and
                phase_iii.get('elbow_vs_shoulder') in ['at', 'above']):

                confidence_scores['BA_LOW'] = 0.80
                if 'BA_LOW' not in matched_rules:
                    matched_rules.append("Phase I: Wrist above shoulder")
                    matched_rules.append("Phase II: Wrist between forehead and shoulder")
                    matched_rules.append("Phase III: Elbow at/above shoulder")

            # Rule 5: Snap - Horizontal position with stopping
            if (phase_i.get('wrist_vs_shoulder') in ['at', 'above'] and
                phase_i.get('elbow_vs_shoulder') == 'at' and
                phase_ii.get('wrist_vs_shoulder') == 'at' and
                phase_iii.get('elbow_vs_shoulder') == 'at'):

                confidence_scores['SNAP'] = 0.85
                if 'SNAP' not in matched_rules:
                    matched_rules.append("Phase I: Elbow at shoulder height")
                    matched_rules.append("Phase II: Wrist at shoulder height")
                    matched_rules.append("Phase III: Horizontal motion pattern")

        # Select the type with highest confidence
        if confidence_scores:
            motion_type = max(confidence_scores, key=confidence_scores.get)
            confidence = confidence_scores[motion_type]
        else:
            # Default to most common pattern if no rules matched
            motion_type = 'STRAIGHT'
            confidence = 0.5
            matched_rules.append("Default classification (no clear pattern matched)")

        return motion_type, confidence, matched_rules

    def classify_arm_swing(
        self,
        skeleton_df: pd.DataFrame,
        phase_info: dict,
        arm_swing_phases: dict,
        velocity_df: Optional[pd.DataFrame] = None
    ) -> dict:
        """
        Classify arm swing motion type.

        Args:
            skeleton_df: DataFrame with skeleton data.
            phase_info: Dictionary with main phase boundaries.
            arm_swing_phases: Dictionary with arm swing sub-phase boundaries.
            velocity_df: Optional DataFrame with velocity data.

        Returns:
            Dictionary containing:
            {
                'type': str,
                'type_display': str,
                'confidence': float,
                'features': dict,
                'matched_rules': list,
                'has_stopping_motion': bool
            }

        Raises:
            ValueError: If required data is missing.
        """
        if arm_swing_phases is None or 'phase_iii' not in arm_swing_phases:
            raise ValueError("arm_swing_phases must contain Phase III information")

        # Analyze phase characteristics
        characteristics = self._analyze_phase_characteristics(skeleton_df, arm_swing_phases)

        # Detect stopping motion
        has_stopping = False
        if velocity_df is not None and 'wrist_velocity' in velocity_df.columns:
            velocity_data = velocity_df['wrist_velocity'].values
            phase_iii_frames = (
                arm_swing_phases['phase_iii']['start'],
                arm_swing_phases['phase_iii']['end']
            )
            has_stopping = self._detect_stopping_motion(velocity_data, phase_iii_frames)
        else:
            logger.warning("Velocity data not provided, cannot detect stopping motion accurately")

        # Classify based on rules
        motion_type, confidence, matched_rules = self._classify_by_rules(
            characteristics, has_stopping
        )

        # Build result
        result = {
            'type': motion_type,
            'type_display': self.MOTION_TYPES[motion_type],
            'confidence': float(confidence),
            'features': characteristics,
            'matched_rules': matched_rules,
            'has_stopping_motion': has_stopping
        }

        logger.info(
            f"Classification result: {result['type_display']} "
            f"(confidence: {confidence:.2%})"
        )

        return result

    def compare_with_standard(
        self,
        skeleton_df: pd.DataFrame,
        angles_df: pd.DataFrame,
        phase_info: dict,
        motion_type: str
    ) -> dict:
        """
        Compare motion with standard patterns for the classified type.

        Args:
            skeleton_df: DataFrame with skeleton data.
            angles_df: DataFrame with joint angles.
            phase_info: Dictionary with phase boundaries.
            motion_type: Classified motion type.

        Returns:
            Dictionary with comparison metrics:
            {
                'elbow_angle_deviation': float,
                'shoulder_angle_deviation': float,
                'timing_similarity': float,
                'trajectory_similarity': float
            }
        """
        # Define standard angle ranges for each motion type
        standard_ranges = {
            'STRAIGHT': {
                'elbow_flexion': {'min': 130, 'max': 160, 'optimal': 145},
                'shoulder_abduction': {'min': 150, 'max': 180, 'optimal': 165}
            },
            'BA_HIGH': {
                'elbow_flexion': {'min': 140, 'max': 170, 'optimal': 155},
                'shoulder_abduction': {'min': 160, 'max': 180, 'optimal': 170}
            },
            'BA_LOW': {
                'elbow_flexion': {'min': 110, 'max': 140, 'optimal': 125},
                'shoulder_abduction': {'min': 130, 'max': 160, 'optimal': 145}
            },
            'SNAP': {
                'elbow_flexion': {'min': 120, 'max': 150, 'optimal': 135},
                'shoulder_abduction': {'min': 100, 'max': 130, 'optimal': 115}
            },
            'CIRCULAR': {
                'elbow_flexion': {'min': 90, 'max': 120, 'optimal': 105},
                'shoulder_abduction': {'min': 80, 'max': 110, 'optimal': 95}
            }
        }

        if motion_type not in standard_ranges:
            logger.warning(f"Unknown motion type: {motion_type}")
            return {}

        standard = standard_ranges[motion_type]
        comparison = {}

        # Calculate average angles during contact phase
        if 'contact' in phase_info:
            contact_start = phase_info['contact']['start']
            contact_end = phase_info['contact']['end']

            contact_angles = angles_df.iloc[contact_start:contact_end + 1]

            # Elbow angle deviation
            if 'elbow_flexion' in contact_angles.columns:
                avg_elbow = contact_angles['elbow_flexion'].mean()
                optimal_elbow = standard['elbow_flexion']['optimal']
                comparison['elbow_angle_deviation'] = float(avg_elbow - optimal_elbow)

            # Shoulder angle deviation
            if 'shoulder_abduction' in contact_angles.columns:
                avg_shoulder = contact_angles['shoulder_abduction'].mean()
                optimal_shoulder = standard['shoulder_abduction']['optimal']
                comparison['shoulder_angle_deviation'] = float(avg_shoulder - optimal_shoulder)

        # Calculate timing similarity (simplified)
        # Compare phase durations with typical patterns
        if 'arm_swing' in phase_info:
            arm_swing_duration = (
                skeleton_df.iloc[phase_info['arm_swing']['end']]['time'] -
                skeleton_df.iloc[phase_info['arm_swing']['start']]['time']
            )

            # Typical arm swing duration is 0.8-1.2 seconds
            typical_duration = 1.0
            timing_diff = abs(arm_swing_duration - typical_duration)
            timing_similarity = max(0, 1.0 - timing_diff / typical_duration)
            comparison['timing_similarity'] = float(timing_similarity)

        logger.info(f"Comparison complete: {comparison}")

        return comparison

    def get_motion_description(self, motion_type: str) -> str:
        """
        Get detailed description of motion type.

        Args:
            motion_type: Motion type code.

        Returns:
            Detailed description string.
        """
        descriptions = {
            'STRAIGHT': (
                "Straight arm swing is characterized by a high arc motion with "
                "the wrist and elbow consistently above the shoulder level. "
                "There is a distinct stopping motion in the final cocking phase."
            ),
            'BA_HIGH': (
                "Bow-Arrow High features a very high arc with the wrist remaining "
                "above the forehead throughout most phases. This creates maximum "
                "power generation but requires excellent timing."
            ),
            'BA_LOW': (
                "Bow-Arrow Low uses a medium arc with the wrist positioned between "
                "forehead and shoulder. This balanced approach offers good power "
                "with easier execution."
            ),
            'SNAP': (
                "Snap motion is characterized by a horizontal arm path at shoulder "
                "height. This quick, compact motion is often used for deception "
                "and quick attacks."
            ),
            'CIRCULAR': (
                "Circular motion features continuous movement without stopping, "
                "with the arm moving in a low, circular path. This creates a "
                "whip-like effect and is often the fastest swing type."
            )
        }

        return descriptions.get(motion_type, "Unknown motion type")
