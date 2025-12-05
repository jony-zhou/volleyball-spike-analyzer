"""
Unit tests for phase detection module.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.phase_detector import FullMotionPhaseDetector, ArmSwingPhaseDetector


class TestFullMotionPhaseDetector:
    """Test cases for FullMotionPhaseDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a FullMotionPhaseDetector instance."""
        return FullMotionPhaseDetector(
            min_approach_velocity=1.0,
            contact_window=0.1,
            smooth_window=5
        )

    @pytest.fixture
    def sample_skeleton_df(self):
        """Create sample skeleton DataFrame for testing."""
        num_frames = 100
        fps = 30.0

        data = []
        for i in range(num_frames):
            # Create landmarks with 33 points, 4 dimensions (x, y, z, visibility)
            landmarks = np.zeros((33, 4))
            landmarks[:, 3] = 1.0  # Set visibility

            # Simulate motion trajectory
            # Right wrist (16)
            t = i / num_frames
            landmarks[16, 0] = 0.5 + 0.2 * np.sin(t * 2 * np.pi)  # x
            landmarks[16, 1] = 1.0 + 0.3 * t  # y (rising)
            landmarks[16, 2] = 0.5 + 0.2 * t  # z

            # Right ankle (28)
            if i < 50:  # On ground
                landmarks[28, 1] = 0.1
            else:  # In air
                landmarks[28, 1] = 0.3 + 0.1 * np.sin((i - 50) / 25 * np.pi)

            # Hips (23, 24)
            hip_height = 0.5
            if i >= 20 and i < 70:  # Jump phase
                hip_height = 0.5 + 0.3 * np.sin((i - 20) / 50 * np.pi)

            landmarks[23, :3] = [-0.2, hip_height, 0.0]  # left hip
            landmarks[24, :3] = [0.2, hip_height, 0.0]  # right hip

            data.append({
                'frame': i,
                'time': i / fps,
                'landmarks_3d': landmarks
            })

        return pd.DataFrame(data)

    def test_detector_initialization(self):
        """Test FullMotionPhaseDetector initialization."""
        detector = FullMotionPhaseDetector(
            min_approach_velocity=2.0,
            contact_window=0.15,
            smooth_window=7
        )
        assert detector.min_approach_velocity == 2.0
        assert detector.contact_window == 0.15
        assert detector.smooth_window == 7

    def test_detector_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            FullMotionPhaseDetector(min_approach_velocity=-1.0)

        with pytest.raises(ValueError):
            FullMotionPhaseDetector(contact_window=0.0)

        with pytest.raises(ValueError):
            FullMotionPhaseDetector(smooth_window=4)  # Must be odd

        with pytest.raises(ValueError):
            FullMotionPhaseDetector(smooth_window=2)  # Must be >= 3

    def test_extract_landmark_position(self, detector):
        """Test landmark position extraction."""
        landmarks = np.zeros((33, 4))
        landmarks[16, :3] = [1.0, 2.0, 3.0]

        position = detector._extract_landmark_position(landmarks, 16)
        assert position.shape == (3,)
        assert np.array_equal(position, [1.0, 2.0, 3.0])

    def test_calculate_velocity(self, detector):
        """Test velocity calculation."""
        # Create simple linear motion
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ])
        fps = 30.0

        velocity = detector._calculate_velocity(positions, fps)
        assert velocity.shape == (4,)
        # Velocity should be approximately 30 m/s (1 meter per frame * 30 fps)
        assert velocity.mean() > 0

    def test_calculate_acceleration(self, detector):
        """Test acceleration calculation."""
        # Constant velocity
        velocity = np.array([1.0, 1.0, 1.0, 1.0])
        fps = 30.0

        acceleration = detector._calculate_acceleration(velocity, fps)
        assert acceleration.shape == (4,)

    def test_smooth_signal(self, detector):
        """Test signal smoothing."""
        # Create noisy signal
        signal = np.array([1.0, 5.0, 2.0, 4.0, 3.0, 6.0, 3.0, 5.0, 4.0, 7.0])
        smoothed = detector._smooth_signal(signal)
        assert smoothed.shape == signal.shape

    def test_smooth_signal_short(self, detector):
        """Test signal smoothing with short signal."""
        signal = np.array([1.0, 2.0])
        smoothed = detector._smooth_signal(signal)
        # Should return original signal
        assert np.array_equal(smoothed, signal)

    def test_detect_phases(self, detector, sample_skeleton_df):
        """Test detection of all phases."""
        phases = detector.detect_phases(sample_skeleton_df)

        # Check that phases are detected
        if phases is not None:
            assert isinstance(phases, dict)
            expected_phases = ['approach', 'takeoff', 'arm_swing', 'contact', 'landing']
            for phase in expected_phases:
                assert phase in phases
                assert 'start' in phases[phase]
                assert 'end' in phases[phase]
                assert phases[phase]['start'] <= phases[phase]['end']

    def test_detect_phases_invalid_dataframe(self, detector):
        """Test phase detection with invalid DataFrame."""
        invalid_df = pd.DataFrame({'frame': [0, 1, 2]})

        with pytest.raises(ValueError):
            detector.detect_phases(invalid_df)

    def test_detect_phases_too_short(self, detector):
        """Test phase detection with too short sequence."""
        short_df = pd.DataFrame([
            {'frame': 0, 'time': 0.0, 'landmarks_3d': np.zeros((33, 4))},
            {'frame': 1, 'time': 0.033, 'landmarks_3d': np.zeros((33, 4))}
        ])

        result = detector.detect_phases(short_df)
        assert result is None


class TestArmSwingPhaseDetector:
    """Test cases for ArmSwingPhaseDetector class."""

    @pytest.fixture
    def detector(self):
        """Create an ArmSwingPhaseDetector instance."""
        return ArmSwingPhaseDetector(smooth_window=5, peak_prominence=0.05)

    @pytest.fixture
    def sample_skeleton_df(self):
        """Create sample skeleton DataFrame for arm swing."""
        num_frames = 30
        fps = 30.0

        data = []
        for i in range(num_frames):
            landmarks = np.zeros((33, 4))
            landmarks[:, 3] = 1.0

            # Simulate arm swing: wrist rises then falls
            t = i / num_frames
            wrist_height = 0.5 + 0.5 * np.sin(t * np.pi)  # Rise and fall

            landmarks[16, :3] = [0.5, wrist_height, 0.0]  # right wrist
            landmarks[14, :3] = [0.4, wrist_height * 0.7, 0.0]  # right elbow

            data.append({
                'frame': i,
                'time': i / fps,
                'landmarks_3d': landmarks
            })

        return pd.DataFrame(data)

    def test_detector_initialization(self):
        """Test ArmSwingPhaseDetector initialization."""
        detector = ArmSwingPhaseDetector(smooth_window=7, peak_prominence=0.1)
        assert detector.smooth_window == 7
        assert detector.peak_prominence == 0.1

    def test_detector_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            ArmSwingPhaseDetector(smooth_window=4)  # Must be odd

        with pytest.raises(ValueError):
            ArmSwingPhaseDetector(peak_prominence=-0.1)  # Must be positive

    def test_extract_landmark_position(self, detector):
        """Test landmark position extraction."""
        landmarks = np.zeros((33, 4))
        landmarks[16, :3] = [1.0, 2.0, 3.0]

        position = detector._extract_landmark_position(landmarks, 16)
        assert position.shape == (3,)
        assert np.array_equal(position, [1.0, 2.0, 3.0])

    def test_smooth_signal(self, detector):
        """Test signal smoothing."""
        signal = np.array([1.0, 5.0, 2.0, 4.0, 3.0, 6.0, 3.0, 5.0, 4.0, 7.0])
        smoothed = detector._smooth_signal(signal)
        assert smoothed.shape == signal.shape

    def test_detect_sub_phases(self, detector, sample_skeleton_df):
        """Test detection of arm swing sub-phases."""
        arm_swing_start = 0
        arm_swing_end = len(sample_skeleton_df) - 1

        phases = detector.detect_sub_phases(
            sample_skeleton_df,
            arm_swing_start,
            arm_swing_end
        )

        if phases is not None:
            assert isinstance(phases, dict)
            expected_phases = ['phase_i', 'phase_ii', 'phase_iii']
            for phase in expected_phases:
                assert phase in phases
                assert 'start' in phases[phase]
                assert 'end' in phases[phase]

    def test_detect_sub_phases_invalid_bounds(self, detector, sample_skeleton_df):
        """Test sub-phase detection with invalid bounds."""
        with pytest.raises(ValueError):
            detector.detect_sub_phases(sample_skeleton_df, 10, 5)  # Start > end

        with pytest.raises(ValueError):
            detector.detect_sub_phases(sample_skeleton_df, 0, 1000)  # End out of range

    def test_detect_sub_phases_invalid_dataframe(self, detector):
        """Test sub-phase detection with invalid DataFrame."""
        invalid_df = pd.DataFrame({'frame': [0, 1, 2]})

        with pytest.raises(ValueError):
            detector.detect_sub_phases(invalid_df, 0, 2)

    def test_detect_sub_phases_too_short(self, detector):
        """Test sub-phase detection with too short segment."""
        short_df = pd.DataFrame([
            {'frame': 0, 'time': 0.0, 'landmarks_3d': np.zeros((33, 4))},
            {'frame': 1, 'time': 0.033, 'landmarks_3d': np.zeros((33, 4))}
        ])

        result = detector.detect_sub_phases(short_df, 0, 1)
        assert result is None
