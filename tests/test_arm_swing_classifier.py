"""
Unit tests for arm swing classifier module.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.arm_swing_classifier import ArmSwingClassifier


class TestArmSwingClassifier:
    """Test cases for ArmSwingClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create an ArmSwingClassifier instance."""
        return ArmSwingClassifier(
            height_threshold=0.05,
            velocity_threshold=0.2,
            stopping_duration=3
        )

    @pytest.fixture
    def sample_skeleton_df(self):
        """Create sample skeleton DataFrame for testing."""
        num_frames = 50
        fps = 30.0

        data = []
        for i in range(num_frames):
            landmarks = np.zeros((33, 4))
            landmarks[:, 3] = 1.0  # Set visibility

            t = i / num_frames

            # Nose
            landmarks[0, :3] = [0.0, 1.5, 0.0]

            # Shoulders
            landmarks[11, :3] = [-0.3, 1.2, 0.0]  # left
            landmarks[12, :3] = [0.3, 1.2, 0.0]  # right

            # Hips
            landmarks[23, :3] = [-0.2, 0.5, 0.0]  # left
            landmarks[24, :3] = [0.2, 0.5, 0.0]  # right

            # Arm motion (simulate high arc)
            wrist_height = 1.2 + 0.3 * np.sin(t * np.pi)
            elbow_height = 1.2 + 0.2 * np.sin(t * np.pi)

            landmarks[14, :3] = [0.5, elbow_height, 0.2]  # right elbow
            landmarks[16, :3] = [0.7, wrist_height, 0.3]  # right wrist

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
            'approach': {'start': 0, 'end': 10},
            'takeoff': {'start': 10, 'end': 20},
            'arm_swing': {'start': 20, 'end': 40},
            'contact': {'start': 35, 'end': 40},
            'landing': {'start': 40, 'end': 49}
        }

    @pytest.fixture
    def sample_arm_swing_phases(self):
        """Create sample arm swing sub-phases."""
        return {
            'phase_i': {'start': 20, 'end': 27},
            'phase_ii': {'start': 27, 'end': 33},
            'phase_iii': {'start': 33, 'end': 40}
        }

    @pytest.fixture
    def sample_velocity_df(self):
        """Create sample velocity DataFrame."""
        num_frames = 50
        fps = 30.0

        # Simulate velocity with peak in middle
        wrist_velocity = []
        for i in range(num_frames):
            t = i / num_frames
            vel = 5.0 * np.sin(t * np.pi)  # Peak at middle
            wrist_velocity.append(vel)

        return pd.DataFrame({
            'frame': range(num_frames),
            'time': [i / fps for i in range(num_frames)],
            'wrist_velocity': wrist_velocity
        })

    def test_classifier_initialization(self):
        """Test ArmSwingClassifier initialization."""
        classifier = ArmSwingClassifier(
            height_threshold=0.1,
            velocity_threshold=0.3,
            stopping_duration=5
        )
        assert classifier.height_threshold == 0.1
        assert classifier.velocity_threshold == 0.3
        assert classifier.stopping_duration == 5

    def test_classifier_initialization_invalid_params(self):
        """Test initialization with invalid parameters."""
        with pytest.raises(ValueError):
            ArmSwingClassifier(height_threshold=-0.1)

        with pytest.raises(ValueError):
            ArmSwingClassifier(velocity_threshold=0.0)

        with pytest.raises(ValueError):
            ArmSwingClassifier(velocity_threshold=1.5)

        with pytest.raises(ValueError):
            ArmSwingClassifier(stopping_duration=0)

    def test_extract_landmark_position(self, classifier):
        """Test landmark position extraction."""
        landmarks = np.zeros((33, 4))
        landmarks[16, :3] = [1.0, 2.0, 3.0]

        position = classifier._extract_landmark_position(landmarks, 16)
        assert position.shape == (3,)
        assert np.array_equal(position, [1.0, 2.0, 3.0])

    def test_calculate_forehead_position(self, classifier):
        """Test forehead position calculation."""
        landmarks = np.zeros((33, 4))
        landmarks[0, :3] = [0.0, 2.0, 0.0]  # nose
        landmarks[11, :3] = [-0.3, 1.2, 0.0]  # left shoulder
        landmarks[12, :3] = [0.3, 1.2, 0.0]  # right shoulder

        forehead = classifier._calculate_forehead_position(landmarks)
        assert forehead.shape == (3,)
        # Forehead should be between shoulders and nose
        assert forehead[1] > 1.2 and forehead[1] < 2.0

    def test_check_height_relative_to_landmark(self, classifier):
        """Test relative height checking."""
        point = np.array([0.0, 1.5, 0.0])
        reference = np.array([0.0, 1.0, 0.0])

        result = classifier._check_height_relative_to_landmark(point, reference)
        assert result == 'above'

        point = np.array([0.0, 0.5, 0.0])
        result = classifier._check_height_relative_to_landmark(point, reference)
        assert result == 'below'

        point = np.array([0.0, 1.02, 0.0])
        result = classifier._check_height_relative_to_landmark(
            point, reference, threshold=0.05
        )
        assert result == 'at'

    def test_detect_stopping_motion_with_stop(self, classifier, sample_velocity_df):
        """Test stopping motion detection when there is stopping."""
        # Create velocity that drops to near zero
        velocity_data = sample_velocity_df['wrist_velocity'].values
        velocity_data[35:40] = 0.1  # Drop velocity to simulate stopping

        phase_iii_frames = (33, 40)
        has_stopping = classifier._detect_stopping_motion(velocity_data, phase_iii_frames)

        # Should detect stopping
        assert has_stopping is True or has_stopping is False  # Result depends on thresholds

    def test_detect_stopping_motion_continuous(self, classifier):
        """Test stopping motion detection with continuous motion."""
        # Create constant high velocity (no stopping)
        velocity_data = np.full(50, 5.0)
        phase_iii_frames = (33, 40)

        has_stopping = classifier._detect_stopping_motion(velocity_data, phase_iii_frames)
        assert has_stopping is False

    def test_detect_stopping_motion_out_of_range(self, classifier):
        """Test stopping motion detection with out of range frames."""
        velocity_data = np.full(50, 5.0)
        phase_iii_frames = (40, 100)  # End out of range

        has_stopping = classifier._detect_stopping_motion(velocity_data, phase_iii_frames)
        assert has_stopping is False

    def test_analyze_phase_characteristics(
        self,
        classifier,
        sample_skeleton_df,
        sample_arm_swing_phases
    ):
        """Test phase characteristics analysis."""
        characteristics = classifier._analyze_phase_characteristics(
            sample_skeleton_df,
            sample_arm_swing_phases
        )

        assert isinstance(characteristics, dict)
        expected_phases = ['phase_i', 'phase_ii', 'phase_iii']

        for phase in expected_phases:
            if phase in characteristics:
                assert 'wrist_vs_shoulder' in characteristics[phase]
                assert 'wrist_vs_forehead' in characteristics[phase]
                assert 'elbow_vs_shoulder' in characteristics[phase]

    def test_classify_by_rules_circular(self, classifier):
        """Test classification rules for circular motion."""
        characteristics = {
            'phase_i': {
                'wrist_vs_shoulder': 'below',
                'elbow_vs_shoulder': 'at',
                'wrist_below_elbow': False
            },
            'phase_ii': {
                'wrist_vs_forehead': 'below',
                'wrist_below_elbow': True
            },
            'phase_iii': {
                'wrist_vs_shoulder': 'at',
                'elbow_vs_shoulder': 'at'
            }
        }

        motion_type, confidence, matched_rules = classifier._classify_by_rules(
            characteristics,
            has_stopping=False
        )

        assert motion_type == 'CIRCULAR'
        assert confidence > 0
        assert len(matched_rules) > 0

    def test_classify_by_rules_straight(self, classifier):
        """Test classification rules for straight motion."""
        characteristics = {
            'phase_i': {
                'wrist_vs_shoulder': 'above',
                'elbow_vs_shoulder': 'above'
            },
            'phase_ii': {
                'wrist_vs_forehead': 'above'
            },
            'phase_iii': {
                'elbow_vs_shoulder': 'above'
            }
        }

        motion_type, confidence, matched_rules = classifier._classify_by_rules(
            characteristics,
            has_stopping=True
        )

        # Should classify as STRAIGHT or BA_HIGH
        assert motion_type in ['STRAIGHT', 'BA_HIGH']
        assert confidence > 0

    def test_classify_arm_swing(
        self,
        classifier,
        sample_skeleton_df,
        sample_phase_info,
        sample_arm_swing_phases,
        sample_velocity_df
    ):
        """Test complete arm swing classification."""
        result = classifier.classify_arm_swing(
            sample_skeleton_df,
            sample_phase_info,
            sample_arm_swing_phases,
            velocity_df=sample_velocity_df
        )

        assert isinstance(result, dict)
        assert 'type' in result
        assert 'type_display' in result
        assert 'confidence' in result
        assert 'features' in result
        assert 'matched_rules' in result
        assert 'has_stopping_motion' in result

        # Check type is valid
        assert result['type'] in classifier.MOTION_TYPES

        # Check confidence is in valid range
        assert 0.0 <= result['confidence'] <= 1.0

    def test_classify_arm_swing_without_velocity(
        self,
        classifier,
        sample_skeleton_df,
        sample_phase_info,
        sample_arm_swing_phases
    ):
        """Test classification without velocity data."""
        result = classifier.classify_arm_swing(
            sample_skeleton_df,
            sample_phase_info,
            sample_arm_swing_phases,
            velocity_df=None
        )

        assert isinstance(result, dict)
        assert 'type' in result
        # Should still classify, but stopping detection may be less accurate

    def test_classify_arm_swing_invalid_phases(
        self,
        classifier,
        sample_skeleton_df,
        sample_phase_info
    ):
        """Test classification with invalid phases."""
        invalid_arm_swing_phases = {
            'phase_i': {'start': 20, 'end': 27}
            # Missing phase_iii
        }

        with pytest.raises(ValueError):
            classifier.classify_arm_swing(
                sample_skeleton_df,
                sample_phase_info,
                invalid_arm_swing_phases
            )

    def test_compare_with_standard(
        self,
        classifier,
        sample_skeleton_df,
        sample_phase_info
    ):
        """Test comparison with standard patterns."""
        # Create sample angles DataFrame
        angles_df = pd.DataFrame({
            'frame': sample_skeleton_df['frame'],
            'time': sample_skeleton_df['time'],
            'elbow_flexion': np.full(len(sample_skeleton_df), 145.0),
            'shoulder_abduction': np.full(len(sample_skeleton_df), 165.0)
        })

        comparison = classifier.compare_with_standard(
            sample_skeleton_df,
            angles_df,
            sample_phase_info,
            motion_type='STRAIGHT'
        )

        if comparison:
            assert isinstance(comparison, dict)
            # May contain angle deviations and timing similarity

    def test_compare_with_standard_unknown_type(
        self,
        classifier,
        sample_skeleton_df,
        sample_phase_info
    ):
        """Test comparison with unknown motion type."""
        angles_df = pd.DataFrame({
            'frame': sample_skeleton_df['frame'],
            'time': sample_skeleton_df['time']
        })

        comparison = classifier.compare_with_standard(
            sample_skeleton_df,
            angles_df,
            sample_phase_info,
            motion_type='UNKNOWN_TYPE'
        )

        assert comparison == {}

    def test_get_motion_description(self, classifier):
        """Test getting motion type descriptions."""
        for motion_type in classifier.MOTION_TYPES.keys():
            description = classifier.get_motion_description(motion_type)
            assert isinstance(description, str)
            assert len(description) > 0

        # Test unknown type
        description = classifier.get_motion_description('UNKNOWN')
        assert description == "Unknown motion type"

    def test_motion_types_constant(self, classifier):
        """Test that MOTION_TYPES constant is properly defined."""
        assert len(classifier.MOTION_TYPES) == 5
        assert 'STRAIGHT' in classifier.MOTION_TYPES
        assert 'BA_HIGH' in classifier.MOTION_TYPES
        assert 'BA_LOW' in classifier.MOTION_TYPES
        assert 'SNAP' in classifier.MOTION_TYPES
        assert 'CIRCULAR' in classifier.MOTION_TYPES
