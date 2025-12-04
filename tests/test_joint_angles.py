"""
Unit tests for joint angle calculation module.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.joint_angles import JointAngleCalculator


class TestJointAngleCalculator:
    """Test cases for JointAngleCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a JointAngleCalculator instance."""
        return JointAngleCalculator(min_visibility=0.5)

    @pytest.fixture
    def sample_landmarks(self):
        """Create sample landmarks for testing."""
        # Create 33 landmarks with x, y, z, visibility
        landmarks = np.zeros((33, 4))

        # Set visibility to 1.0 for all landmarks
        landmarks[:, 3] = 1.0

        # Right shoulder (12)
        landmarks[12, :3] = [0.5, 1.0, 0.0]

        # Left shoulder (11)
        landmarks[11, :3] = [-0.5, 1.0, 0.0]

        # Right elbow (14)
        landmarks[14, :3] = [0.7, 0.7, 0.2]

        # Left elbow (13)
        landmarks[13, :3] = [-0.7, 0.7, 0.2]

        # Right wrist (16)
        landmarks[16, :3] = [0.8, 0.5, 0.3]

        # Left wrist (15)
        landmarks[15, :3] = [-0.8, 0.5, 0.3]

        # Right hip (24)
        landmarks[24, :3] = [0.3, 0.0, 0.0]

        # Left hip (23)
        landmarks[23, :3] = [-0.3, 0.0, 0.0]

        return landmarks

    def test_calculator_initialization(self):
        """Test JointAngleCalculator initialization."""
        calc = JointAngleCalculator(min_visibility=0.6)
        assert calc.min_visibility == 0.6

        # Test invalid min_visibility
        with pytest.raises(ValueError):
            JointAngleCalculator(min_visibility=-0.1)

        with pytest.raises(ValueError):
            JointAngleCalculator(min_visibility=1.5)

    def test_calculate_angle_basic(self, calculator):
        """Test basic angle calculation."""
        # Create a right angle (90 degrees)
        point_a = np.array([1.0, 0.0, 0.0])
        point_b = np.array([0.0, 0.0, 0.0])
        point_c = np.array([0.0, 1.0, 0.0])

        angle = calculator.calculate_angle(point_a, point_b, point_c)
        assert np.isclose(angle, 90.0, atol=1e-6)

    def test_calculate_angle_straight_line(self, calculator):
        """Test angle calculation for straight line (180 degrees)."""
        point_a = np.array([1.0, 0.0, 0.0])
        point_b = np.array([0.0, 0.0, 0.0])
        point_c = np.array([-1.0, 0.0, 0.0])

        angle = calculator.calculate_angle(point_a, point_b, point_c)
        assert np.isclose(angle, 180.0, atol=1e-6)

    def test_calculate_angle_zero(self, calculator):
        """Test angle calculation when points are collinear (0 degrees)."""
        point_a = np.array([1.0, 0.0, 0.0])
        point_b = np.array([0.0, 0.0, 0.0])
        point_c = np.array([1.0, 0.0, 0.0])

        angle = calculator.calculate_angle(point_a, point_b, point_c)
        assert np.isclose(angle, 0.0, atol=1e-6)

    def test_calculate_angle_45_degrees(self, calculator):
        """Test angle calculation for 45 degrees."""
        point_a = np.array([1.0, 0.0, 0.0])
        point_b = np.array([0.0, 0.0, 0.0])
        point_c = np.array([1.0, 1.0, 0.0])

        angle = calculator.calculate_angle(point_a, point_b, point_c)
        assert np.isclose(angle, 45.0, atol=1e-6)

    def test_calculate_angle_with_visibility(self, calculator):
        """Test angle calculation with visibility channel."""
        # Points with 4 dimensions (x, y, z, visibility)
        point_a = np.array([1.0, 0.0, 0.0, 1.0])
        point_b = np.array([0.0, 0.0, 0.0, 1.0])
        point_c = np.array([0.0, 1.0, 0.0, 1.0])

        angle = calculator.calculate_angle(point_a, point_b, point_c)
        assert np.isclose(angle, 90.0, atol=1e-6)

    def test_calculate_angle_invalid_points(self, calculator):
        """Test angle calculation with invalid points."""
        # Points with insufficient dimensions
        point_a = np.array([1.0, 0.0])
        point_b = np.array([0.0, 0.0, 0.0])
        point_c = np.array([0.0, 1.0, 0.0])

        with pytest.raises(ValueError):
            calculator.calculate_angle(point_a, point_b, point_c)

    def test_calculate_shoulder_abduction(self, calculator, sample_landmarks):
        """Test shoulder abduction angle calculation."""
        angle = calculator.calculate_shoulder_abduction(sample_landmarks)
        assert isinstance(angle, float)
        assert 0.0 <= angle <= 180.0

    def test_calculate_shoulder_horizontal_abduction(self, calculator, sample_landmarks):
        """Test shoulder horizontal abduction angle calculation."""
        angle = calculator.calculate_shoulder_horizontal_abduction(sample_landmarks)
        assert isinstance(angle, float)
        assert 0.0 <= angle <= 180.0

    def test_calculate_elbow_flexion(self, calculator, sample_landmarks):
        """Test elbow flexion angle calculation."""
        angle = calculator.calculate_elbow_flexion(sample_landmarks)
        assert isinstance(angle, float)
        assert 0.0 <= angle <= 180.0

    def test_calculate_torso_rotation(self, calculator, sample_landmarks):
        """Test torso rotation angle calculation."""
        angle = calculator.calculate_torso_rotation(sample_landmarks)
        assert isinstance(angle, float)
        assert 0.0 <= angle <= 180.0

    def test_calculate_torso_lean(self, calculator, sample_landmarks):
        """Test torso lean angle calculation."""
        angle = calculator.calculate_torso_lean(sample_landmarks)
        assert isinstance(angle, float)
        assert 0.0 <= angle <= 180.0

    def test_calculate_all_angles(self, calculator, sample_landmarks):
        """Test calculation of all angles at once."""
        angles = calculator.calculate_all_angles(sample_landmarks)

        assert isinstance(angles, dict)
        assert 'shoulder_abduction' in angles
        assert 'shoulder_horizontal_abduction' in angles
        assert 'elbow_flexion' in angles
        assert 'torso_rotation' in angles
        assert 'torso_lean' in angles

        # Check all values are floats (or NaN)
        for key, value in angles.items():
            assert isinstance(value, float) or np.isnan(value)

    def test_calculate_all_angles_invalid_shape(self, calculator):
        """Test calculate_all_angles with invalid landmark shape."""
        invalid_landmarks = np.zeros((30, 4))  # Wrong number of landmarks

        with pytest.raises(ValueError):
            calculator.calculate_all_angles(invalid_landmarks)

    def test_calculate_all_angles_low_visibility(self, calculator, sample_landmarks):
        """Test calculate_all_angles with low visibility landmarks."""
        # Set visibility to below threshold
        sample_landmarks[:, 3] = 0.3

        angles = calculator.calculate_all_angles(sample_landmarks)

        # All angles should be NaN due to low visibility
        for key, value in angles.items():
            assert np.isnan(value)

    def test_calculate_angles_timeseries(self, calculator, sample_landmarks):
        """Test calculation of angles over time series."""
        # Create a DataFrame with multiple frames
        num_frames = 10
        data = []

        for i in range(num_frames):
            data.append({
                'frame': i,
                'time': i / 30.0,  # 30 FPS
                'landmarks_3d': sample_landmarks.copy()
            })

        skeleton_df = pd.DataFrame(data)

        # Calculate angles
        angles_df = calculator.calculate_angles_timeseries(skeleton_df)

        # Check output
        assert isinstance(angles_df, pd.DataFrame)
        assert len(angles_df) == num_frames

        # Check columns
        expected_columns = [
            'frame', 'time', 'shoulder_abduction', 'shoulder_horizontal_abduction',
            'elbow_flexion', 'torso_rotation', 'torso_lean'
        ]
        for col in expected_columns:
            assert col in angles_df.columns

        # Check that all angle values are numeric
        for col in expected_columns[2:]:  # Skip 'frame' and 'time'
            assert angles_df[col].dtype in [np.float64, np.float32]

    def test_calculate_angles_timeseries_invalid_dataframe(self, calculator):
        """Test calculate_angles_timeseries with invalid DataFrame."""
        # Missing required columns
        invalid_df = pd.DataFrame({'frame': [0, 1, 2]})

        with pytest.raises(ValueError):
            calculator.calculate_angles_timeseries(invalid_df)

    def test_check_visibility_sufficient(self, calculator, sample_landmarks):
        """Test visibility check with sufficient visibility."""
        indices = [12, 14, 16]  # right shoulder, elbow, wrist
        result = calculator._check_visibility(sample_landmarks, indices)
        assert result is True

    def test_check_visibility_insufficient(self, calculator, sample_landmarks):
        """Test visibility check with insufficient visibility."""
        # Set some landmarks to low visibility
        sample_landmarks[12, 3] = 0.3
        indices = [12, 14, 16]
        result = calculator._check_visibility(sample_landmarks, indices)
        assert result is False

    def test_check_visibility_no_visibility_channel(self, calculator):
        """Test visibility check when no visibility channel present."""
        landmarks = np.zeros((33, 3))  # No visibility channel
        indices = [12, 14, 16]
        result = calculator._check_visibility(landmarks, indices)
        assert result is True  # Should assume visible

    def test_angle_symmetry(self, calculator):
        """Test that angle calculation is symmetric."""
        # Create three points
        point_a = np.array([1.0, 0.0, 0.0])
        point_b = np.array([0.0, 0.0, 0.0])
        point_c = np.array([0.0, 1.0, 0.0])

        # Calculate angle in both directions
        angle1 = calculator.calculate_angle(point_a, point_b, point_c)
        angle2 = calculator.calculate_angle(point_c, point_b, point_a)

        # Should be the same
        assert np.isclose(angle1, angle2, atol=1e-6)

    def test_calculate_angle_3d(self, calculator):
        """Test angle calculation in 3D space."""
        # Create points in 3D
        point_a = np.array([1.0, 1.0, 1.0])
        point_b = np.array([0.0, 0.0, 0.0])
        point_c = np.array([1.0, -1.0, 1.0])

        angle = calculator.calculate_angle(point_a, point_b, point_c)

        # Verify angle is within valid range
        assert 0.0 <= angle <= 180.0
        assert isinstance(angle, float)

    def test_torso_lean_vertical(self, calculator):
        """Test torso lean calculation when torso is vertical."""
        landmarks = np.zeros((33, 4))
        landmarks[:, 3] = 1.0

        # Set shoulders directly above hips (vertical torso)
        landmarks[12, :3] = [0.5, 1.0, 0.0]  # right shoulder
        landmarks[11, :3] = [-0.5, 1.0, 0.0]  # left shoulder
        landmarks[24, :3] = [0.5, 0.0, 0.0]  # right hip
        landmarks[23, :3] = [-0.5, 0.0, 0.0]  # left hip

        angle = calculator.calculate_torso_lean(landmarks)

        # Should be close to 0 degrees (vertical)
        assert np.isclose(angle, 0.0, atol=5.0)

    def test_torso_lean_horizontal(self, calculator):
        """Test torso lean calculation when torso is horizontal."""
        landmarks = np.zeros((33, 4))
        landmarks[:, 3] = 1.0

        # Set shoulders horizontal to hips
        landmarks[12, :3] = [0.5, 0.5, 0.0]  # right shoulder
        landmarks[11, :3] = [-0.5, 0.5, 0.0]  # left shoulder
        landmarks[24, :3] = [0.5, 0.5, 1.0]  # right hip (forward)
        landmarks[23, :3] = [-0.5, 0.5, 1.0]  # left hip (forward)

        angle = calculator.calculate_torso_lean(landmarks)

        # Should be close to 90 degrees (horizontal)
        assert np.isclose(angle, 90.0, atol=5.0)

    def test_elbow_flexion_extended(self, calculator):
        """Test elbow flexion when arm is fully extended."""
        landmarks = np.zeros((33, 4))
        landmarks[:, 3] = 1.0

        # Create extended arm (straight line)
        landmarks[12, :3] = [0.0, 1.0, 0.0]  # shoulder
        landmarks[14, :3] = [0.0, 0.5, 0.0]  # elbow
        landmarks[16, :3] = [0.0, 0.0, 0.0]  # wrist

        angle = calculator.calculate_elbow_flexion(landmarks)

        # Should be close to 180 degrees (straight)
        assert np.isclose(angle, 180.0, atol=5.0)

    def test_elbow_flexion_90_degrees(self, calculator):
        """Test elbow flexion at 90 degrees."""
        landmarks = np.zeros((33, 4))
        landmarks[:, 3] = 1.0

        # Create 90 degree bend
        landmarks[12, :3] = [0.0, 1.0, 0.0]  # shoulder
        landmarks[14, :3] = [0.0, 0.5, 0.0]  # elbow
        landmarks[16, :3] = [0.5, 0.5, 0.0]  # wrist (perpendicular)

        angle = calculator.calculate_elbow_flexion(landmarks)

        # Should be close to 90 degrees
        assert np.isclose(angle, 90.0, atol=5.0)
