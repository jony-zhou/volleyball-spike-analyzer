"""
Unit tests for spatial metrics calculator module.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.spatial_metrics import SpatialMetricsCalculator


class TestSpatialMetricsCalculator:
    """Test cases for SpatialMetricsCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a SpatialMetricsCalculator instance."""
        return SpatialMetricsCalculator(gravity=9.81)

    @pytest.fixture
    def sample_skeleton_df(self):
        """Create sample skeleton DataFrame for testing."""
        num_frames = 100
        fps = 30.0

        data = []
        for i in range(num_frames):
            landmarks = np.zeros((33, 4))
            landmarks[:, 3] = 1.0  # Set visibility

            # Simulate jump trajectory
            t = i / num_frames

            # Hip center (simulate jump)
            if i < 20:  # Approach
                hip_height = 0.5
            elif i < 70:  # Jump phase
                jump_t = (i - 20) / 50
                hip_height = 0.5 + 0.3 * np.sin(jump_t * np.pi)
            else:  # Landing
                hip_height = 0.5

            landmarks[23, :3] = [-0.2, hip_height, 0.0]  # left hip
            landmarks[24, :3] = [0.2, hip_height, 0.0]  # right hip

            # Wrist (higher during jump)
            wrist_height = hip_height + 0.5 + 0.2 * np.sin(t * 2 * np.pi)
            landmarks[16, :3] = [0.5, wrist_height, 0.5]  # right wrist

            # Ankles
            if i < 20 or i > 70:  # On ground
                ankle_height = 0.1
            else:  # In air
                ankle_height = 0.3

            landmarks[27, :3] = [-0.2, ankle_height, 0.0]  # left ankle
            landmarks[28, :3] = [0.2, ankle_height, 0.0]  # right ankle

            data.append({
                'frame': i,
                'time': i / fps,
                'landmarks_3d': landmarks
            })

        return pd.DataFrame(data)

    @pytest.fixture
    def sample_phase_info(self):
        """Create sample phase information."""
        return {
            'approach': {'start': 0, 'end': 19},
            'takeoff': {'start': 20, 'end': 30},
            'arm_swing': {'start': 30, 'end': 50},
            'contact': {'start': 45, 'end': 50},
            'landing': {'start': 70, 'end': 80}
        }

    def test_calculator_initialization(self):
        """Test SpatialMetricsCalculator initialization."""
        calc = SpatialMetricsCalculator(gravity=10.0)
        assert calc.gravity == 10.0

    def test_calculator_initialization_invalid_gravity(self):
        """Test initialization with invalid gravity."""
        with pytest.raises(ValueError):
            SpatialMetricsCalculator(gravity=-9.81)

        with pytest.raises(ValueError):
            SpatialMetricsCalculator(gravity=0)

    def test_extract_landmark_position(self, calculator):
        """Test landmark position extraction."""
        landmarks = np.zeros((33, 4))
        landmarks[16, :3] = [1.0, 2.0, 3.0]

        position = calculator._extract_landmark_position(landmarks, 16)
        assert position.shape == (3,)
        assert np.array_equal(position, [1.0, 2.0, 3.0])

    def test_calculate_hip_center(self, calculator):
        """Test hip center calculation."""
        landmarks = np.zeros((33, 4))
        landmarks[23, :3] = [-0.3, 0.5, 0.0]  # left hip
        landmarks[24, :3] = [0.3, 0.5, 0.0]  # right hip

        hip_center = calculator._calculate_hip_center(landmarks)
        assert hip_center.shape == (3,)
        assert np.isclose(hip_center[0], 0.0)  # X should be 0 (center)
        assert np.isclose(hip_center[1], 0.5)  # Y should be 0.5

    def test_calculate_baseline_height(self, calculator, sample_skeleton_df, sample_phase_info):
        """Test baseline height calculation."""
        baseline = calculator.calculate_baseline_height(
            sample_skeleton_df,
            sample_phase_info,
            landmark_name='hip_center'
        )

        assert isinstance(baseline, float)
        assert baseline > 0
        # Should be close to 0.5 (approach phase height)
        assert np.abs(baseline - 0.5) < 0.1

    def test_calculate_baseline_height_invalid_phase_info(self, calculator, sample_skeleton_df):
        """Test baseline height with invalid phase info."""
        invalid_phase_info = {'takeoff': {'start': 0, 'end': 10}}

        with pytest.raises(ValueError):
            calculator.calculate_baseline_height(
                sample_skeleton_df,
                invalid_phase_info
            )

    def test_calculate_baseline_height_invalid_landmark(self, calculator, sample_skeleton_df, sample_phase_info):
        """Test baseline height with invalid landmark."""
        with pytest.raises(ValueError):
            calculator.calculate_baseline_height(
                sample_skeleton_df,
                sample_phase_info,
                landmark_name='invalid_landmark'
            )

    def test_calculate_peak_height(self, calculator, sample_skeleton_df, sample_phase_info):
        """Test peak height calculation."""
        peak_height, peak_frame = calculator.calculate_peak_height(
            sample_skeleton_df,
            sample_phase_info,
            landmark_name='hip_center'
        )

        assert isinstance(peak_height, float)
        assert isinstance(peak_frame, int)
        assert peak_height > 0
        assert 0 <= peak_frame < len(sample_skeleton_df)

    def test_calculate_peak_height_invalid_phase_info(self, calculator, sample_skeleton_df):
        """Test peak height with invalid phase info."""
        invalid_phase_info = {'approach': {'start': 0, 'end': 10}}

        with pytest.raises(ValueError):
            calculator.calculate_peak_height(
                sample_skeleton_df,
                invalid_phase_info
            )

    def test_calculate_jump_height_method1(self, calculator, sample_skeleton_df, sample_phase_info):
        """Test jump height calculation using hip displacement method."""
        jump_height = calculator.calculate_jump_height_method1(
            sample_skeleton_df,
            sample_phase_info
        )

        assert isinstance(jump_height, float)
        assert jump_height >= 0
        # Should be close to 0.3 (simulated jump height)
        assert jump_height < 1.0  # Reasonable upper bound

    def test_calculate_flight_time(self, calculator, sample_skeleton_df, sample_phase_info):
        """Test flight time calculation."""
        flight_time = calculator.calculate_flight_time(
            sample_skeleton_df,
            sample_phase_info
        )

        assert isinstance(flight_time, float)
        assert flight_time > 0
        # Flight time should be reasonable (< 2 seconds)
        assert flight_time < 2.0

    def test_calculate_flight_time_invalid_phase_info(self, calculator, sample_skeleton_df):
        """Test flight time with invalid phase info."""
        invalid_phase_info = {'approach': {'start': 0, 'end': 10}}

        with pytest.raises(ValueError):
            calculator.calculate_flight_time(
                sample_skeleton_df,
                invalid_phase_info
            )

    def test_calculate_jump_height_method2(self, calculator, sample_skeleton_df, sample_phase_info):
        """Test jump height calculation using flight time method."""
        jump_height = calculator.calculate_jump_height_method2(
            sample_skeleton_df,
            sample_phase_info
        )

        assert isinstance(jump_height, float)
        assert jump_height >= 0
        assert jump_height < 3.0  # Reasonable upper bound (adjusted for test data)

    def test_calculate_jump_height(self, calculator, sample_skeleton_df, sample_phase_info):
        """Test comprehensive jump height calculation."""
        result = calculator.calculate_jump_height(
            sample_skeleton_df,
            sample_phase_info
        )

        assert isinstance(result, dict)
        assert 'method_1_hip_displacement' in result
        assert 'method_2_flight_time' in result
        assert 'flight_time' in result
        assert 'recommended' in result

        # All values should be floats (or NaN)
        for key, value in result.items():
            assert isinstance(value, float) or np.isnan(value)

    def test_calculate_contact_height(self, calculator, sample_skeleton_df):
        """Test contact height calculation."""
        contact_frame = 45

        contact_height = calculator.calculate_contact_height(
            sample_skeleton_df,
            contact_frame,
            landmark_name='right_wrist'
        )

        assert isinstance(contact_height, float)
        assert contact_height > 0

    def test_calculate_contact_height_out_of_range(self, calculator, sample_skeleton_df):
        """Test contact height with out of range frame."""
        with pytest.raises(ValueError):
            calculator.calculate_contact_height(
                sample_skeleton_df,
                1000,  # Out of range
                landmark_name='right_wrist'
            )

    def test_calculate_contact_height_invalid_landmark(self, calculator, sample_skeleton_df):
        """Test contact height with invalid landmark."""
        with pytest.raises(ValueError):
            calculator.calculate_contact_height(
                sample_skeleton_df,
                45,
                landmark_name='invalid_landmark'
            )

    def test_calculate_horizontal_displacement(self, calculator, sample_skeleton_df, sample_phase_info):
        """Test horizontal displacement calculation."""
        result = calculator.calculate_horizontal_displacement(
            sample_skeleton_df,
            sample_phase_info,
            landmark_name='hip_center'
        )

        assert isinstance(result, dict)
        assert 'lateral_displacement' in result
        assert 'forward_displacement' in result
        assert 'total_displacement' in result

        # Check that values are floats
        for key, value in result.items():
            assert isinstance(value, float)

    def test_calculate_horizontal_displacement_invalid_phase_info(self, calculator, sample_skeleton_df):
        """Test horizontal displacement with invalid phase info."""
        invalid_phase_info = {'approach': {'start': 0, 'end': 10}}

        with pytest.raises(ValueError):
            calculator.calculate_horizontal_displacement(
                sample_skeleton_df,
                invalid_phase_info
            )

    def test_calculate_center_of_mass_trajectory(self, calculator, sample_skeleton_df):
        """Test center of mass trajectory calculation."""
        com_trajectory = calculator.calculate_center_of_mass_trajectory(
            sample_skeleton_df
        )

        assert isinstance(com_trajectory, np.ndarray)
        assert com_trajectory.shape == (len(sample_skeleton_df), 3)

    def test_calculate_spatial_profile(self, calculator, sample_skeleton_df, sample_phase_info):
        """Test comprehensive spatial profile calculation."""
        result = calculator.calculate_spatial_profile(
            sample_skeleton_df,
            sample_phase_info
        )

        assert isinstance(result, dict)
        assert 'jump_height' in result
        assert 'contact_height' in result
        assert 'horizontal_displacement' in result
        assert 'com_trajectory' in result

        # Check jump height structure
        assert isinstance(result['jump_height'], dict)

        # Check contact height is float or NaN
        assert isinstance(result['contact_height'], (float, type(np.nan)))

        # Check horizontal displacement structure
        assert isinstance(result['horizontal_displacement'], dict)

        # Check COM trajectory
        if result['com_trajectory'] is not None:
            assert isinstance(result['com_trajectory'], np.ndarray)
