"""
Unit tests for velocity calculator module.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.velocity_calculator import VelocityCalculator


class TestVelocityCalculator:
    """Test cases for VelocityCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a VelocityCalculator instance."""
        return VelocityCalculator(smooth_window=5, polyorder=2)

    @pytest.fixture
    def sample_skeleton_df(self):
        """Create sample skeleton DataFrame for testing."""
        num_frames = 50
        fps = 30.0

        data = []
        for i in range(num_frames):
            landmarks = np.zeros((33, 4))
            landmarks[:, 3] = 1.0  # Set visibility

            # Simulate motion: wrist moving in arc
            t = i / num_frames
            landmarks[16, 0] = 0.5 + 0.2 * np.sin(t * 2 * np.pi)  # x
            landmarks[16, 1] = 1.0 + 0.3 * t  # y (rising)
            landmarks[16, 2] = 0.5 + 0.2 * t  # z

            # Elbow
            landmarks[14, 0] = 0.4 + 0.1 * np.sin(t * 2 * np.pi)
            landmarks[14, 1] = 0.8 + 0.2 * t
            landmarks[14, 2] = 0.4 + 0.1 * t

            # Shoulder
            landmarks[12, 0] = 0.3
            landmarks[12, 1] = 0.6
            landmarks[12, 2] = 0.3

            data.append({
                'frame': i,
                'time': i / fps,
                'landmarks_3d': landmarks
            })

        return pd.DataFrame(data)

    def test_calculator_initialization(self):
        """Test VelocityCalculator initialization."""
        calc = VelocityCalculator(smooth_window=7, polyorder=3)
        assert calc.smooth_window == 7
        assert calc.polyorder == 3

    def test_calculator_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            VelocityCalculator(smooth_window=4)  # Must be odd

        with pytest.raises(ValueError):
            VelocityCalculator(smooth_window=5, polyorder=5)  # polyorder >= smooth_window

        with pytest.raises(ValueError):
            VelocityCalculator(polyorder=-1)  # Must be non-negative

    def test_smooth_signal(self, calculator):
        """Test signal smoothing."""
        signal = np.array([1.0, 5.0, 2.0, 4.0, 3.0, 6.0, 3.0, 5.0, 4.0, 7.0])
        smoothed = calculator._smooth_signal(signal)
        assert smoothed.shape == signal.shape

    def test_smooth_signal_short(self, calculator):
        """Test signal smoothing with short signal."""
        signal = np.array([1.0, 2.0])
        smoothed = calculator._smooth_signal(signal)
        assert np.array_equal(smoothed, signal)

    def test_calculate_linear_velocity_constant(self, calculator):
        """Test linear velocity calculation with constant motion."""
        # Create constant velocity motion (1 m/s)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0]
        ])
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        velocity = calculator.calculate_linear_velocity(positions, times)
        assert velocity.shape == (5,)
        # Velocity should be approximately 1 m/s
        assert np.abs(velocity.mean() - 1.0) < 0.5

    def test_calculate_linear_velocity_stationary(self, calculator):
        """Test linear velocity calculation with stationary position."""
        positions = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        times = np.array([0.0, 1.0, 2.0])

        velocity = calculator.calculate_linear_velocity(positions, times)
        assert velocity.shape == (3,)
        # Velocity should be close to 0
        assert velocity.mean() < 0.1

    def test_calculate_linear_velocity_invalid_shape(self, calculator):
        """Test linear velocity calculation with invalid shapes."""
        positions = np.array([[1.0, 2.0]])  # Wrong shape (n, 2)
        times = np.array([0.0])

        with pytest.raises(ValueError):
            calculator.calculate_linear_velocity(positions, times)

    def test_calculate_linear_velocity_mismatched_length(self, calculator):
        """Test linear velocity calculation with mismatched lengths."""
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        times = np.array([0.0, 1.0, 2.0])  # Different length

        with pytest.raises(ValueError):
            calculator.calculate_linear_velocity(positions, times)

    def test_calculate_linear_acceleration(self, calculator):
        """Test linear acceleration calculation."""
        velocities = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        acceleration = calculator.calculate_linear_acceleration(velocities, times)
        assert acceleration.shape == (5,)
        # Acceleration should be approximately 1 m/s²
        assert np.abs(acceleration.mean() - 1.0) < 0.5

    def test_calculate_linear_acceleration_invalid_shape(self, calculator):
        """Test acceleration calculation with invalid shapes."""
        velocities = np.array([[1.0, 2.0]])  # Wrong shape
        times = np.array([0.0, 1.0])

        with pytest.raises(ValueError):
            calculator.calculate_linear_acceleration(velocities, times)

    def test_calculate_angular_velocity(self, calculator):
        """Test angular velocity calculation."""
        # Create linearly increasing angles
        angles = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        angular_velocity = calculator.calculate_angular_velocity(angles, times)
        assert angular_velocity.shape == (5,)
        # Angular velocity should be approximately 10 deg/s
        assert np.abs(angular_velocity.mean() - 10.0) < 5.0

    def test_calculate_angular_velocity_invalid_shape(self, calculator):
        """Test angular velocity calculation with invalid shapes."""
        angles = np.array([[1.0, 2.0]])  # Wrong shape
        times = np.array([0.0, 1.0])

        with pytest.raises(ValueError):
            calculator.calculate_angular_velocity(angles, times)

    def test_extract_landmark_positions(self, calculator, sample_skeleton_df):
        """Test landmark position extraction."""
        positions = calculator._extract_landmark_positions(
            sample_skeleton_df, 'right_wrist'
        )
        assert positions.shape == (len(sample_skeleton_df), 3)

    def test_extract_landmark_positions_invalid_name(self, calculator, sample_skeleton_df):
        """Test landmark position extraction with invalid name."""
        with pytest.raises(KeyError):
            calculator._extract_landmark_positions(
                sample_skeleton_df, 'invalid_landmark'
            )

    def test_analyze_velocity_profile(self, calculator, sample_skeleton_df):
        """Test velocity profile analysis."""
        result = calculator.analyze_velocity_profile(sample_skeleton_df)

        assert isinstance(result, dict)
        assert 'wrist_velocity' in result
        assert 'elbow_velocity' in result
        assert 'shoulder_velocity' in result

        # Check velocity data structure
        wrist_vel = result['wrist_velocity']
        assert 'values' in wrist_vel
        assert 'max' in wrist_vel
        assert 'mean' in wrist_vel
        assert 'std' in wrist_vel

    def test_analyze_velocity_profile_with_angles(self, calculator, sample_skeleton_df):
        """Test velocity profile analysis with angle data."""
        # Create sample angles DataFrame
        angles_df = pd.DataFrame({
            'frame': sample_skeleton_df['frame'],
            'time': sample_skeleton_df['time'],
            'shoulder_abduction': np.linspace(0, 180, len(sample_skeleton_df)),
            'elbow_flexion': np.linspace(90, 180, len(sample_skeleton_df))
        })

        result = calculator.analyze_velocity_profile(
            sample_skeleton_df,
            angles_df=angles_df
        )

        assert 'shoulder_angular_velocity' in result
        assert 'elbow_angular_velocity' in result

    def test_analyze_velocity_profile_with_phase_info(self, calculator, sample_skeleton_df):
        """Test velocity profile analysis with phase information."""
        phase_info = {
            'contact': {'start': 25, 'end': 30}
        }

        result = calculator.analyze_velocity_profile(
            sample_skeleton_df,
            phase_info=phase_info
        )

        # Should have 'at_contact' values
        if 'wrist_velocity' in result:
            assert 'at_contact' in result['wrist_velocity']

    def test_analyze_velocity_profile_invalid_dataframe(self, calculator):
        """Test velocity profile analysis with invalid DataFrame."""
        invalid_df = pd.DataFrame({'frame': [0, 1, 2]})

        with pytest.raises(ValueError):
            calculator.analyze_velocity_profile(invalid_df)

    def test_get_peak_velocity_frame(self, calculator, sample_skeleton_df):
        """Test peak velocity frame detection."""
        peak_frame = calculator.get_peak_velocity_frame(
            sample_skeleton_df,
            landmark_name='right_wrist'
        )

        assert isinstance(peak_frame, int)
        assert 0 <= peak_frame < len(sample_skeleton_df)

    def test_get_peak_velocity_frame_invalid_landmark(self, calculator, sample_skeleton_df):
        """Test peak velocity frame with invalid landmark."""
        with pytest.raises(KeyError):
            calculator.get_peak_velocity_frame(
                sample_skeleton_df,
                landmark_name='invalid_landmark'
            )

    def test_calculate_velocity_timeseries(self, calculator, sample_skeleton_df):
        """Test velocity timeseries calculation."""
        velocity_df = calculator.calculate_velocity_timeseries(sample_skeleton_df)

        assert isinstance(velocity_df, pd.DataFrame)
        assert len(velocity_df) == len(sample_skeleton_df)

        # Check columns
        expected_columns = [
            'frame', 'time', 'wrist_velocity', 'elbow_velocity',
            'shoulder_velocity', 'wrist_acceleration', 'elbow_acceleration'
        ]
        for col in expected_columns:
            assert col in velocity_df.columns

    def test_calculate_velocity_timeseries_invalid_dataframe(self, calculator):
        """Test velocity timeseries calculation with invalid DataFrame."""
        invalid_df = pd.DataFrame({'invalid': [0, 1, 2]})

        with pytest.raises(ValueError):
            calculator.calculate_velocity_timeseries(invalid_df)
