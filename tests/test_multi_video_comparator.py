"""
Unit tests for multi-video comparator module.
"""

import numpy as np
import pandas as pd
import pytest

from src.comparison.multi_video_comparator import MultiVideoComparator


class TestMultiVideoComparator:
    """Test cases for MultiVideoComparator class."""

    @pytest.fixture
    def comparator(self):
        """Create a MultiVideoComparator instance."""
        return MultiVideoComparator()

    @pytest.fixture
    def sample_analysis_results(self):
        """Create sample analysis results."""
        num_frames = 50
        fps = 30.0

        # Create skeleton DataFrame
        skeleton_data = []
        for i in range(num_frames):
            landmarks = np.zeros((33, 4))
            landmarks[:, 3] = 1.0
            skeleton_data.append({
                'frame': i,
                'time': i / fps,
                'landmarks_3d': landmarks
            })
        skeleton_df = pd.DataFrame(skeleton_data)

        # Create results dictionary
        results = {
            'skeleton_df': skeleton_df,
            'phases': {
                'approach': {'start': 0, 'end': 10, 'duration': 0.33},
                'takeoff': {'start': 10, 'end': 15, 'duration': 0.17},
                'arm_swing': {'start': 15, 'end': 30, 'duration': 0.50},
                'contact': {'start': 25, 'end': 30, 'duration': 0.17},
                'landing': {'start': 30, 'end': 40, 'duration': 0.33}
            },
            'classification': {
                'type': 'STRAIGHT',
                'type_display': 'Straight',
                'confidence': 0.85
            },
            'spatial_data': {
                'jump_height': {
                    'recommended': 0.45,
                    'method_1_hip_displacement': 0.43,
                    'method_2_flight_time': 0.47,
                    'flight_time': 0.31
                },
                'contact_height': 2.85,
                'horizontal_displacement': {
                    'total_displacement': 0.85,
                    'forward_displacement': 0.80,
                    'lateral_displacement': 0.25
                }
            },
            'velocity_data': {
                'wrist_velocity': {
                    'max': 8.5,
                    'mean': 3.2,
                    'at_contact': 7.8
                },
                'shoulder_angular_velocity': {
                    'max': 1500.0,
                    'mean': 650.0
                }
            },
            'angles_df': pd.DataFrame({
                'frame': range(num_frames),
                'time': [i / fps for i in range(num_frames)]
            }),
            'velocity_df': pd.DataFrame({
                'frame': range(num_frames),
                'time': [i / fps for i in range(num_frames)],
                'wrist_velocity': np.random.rand(num_frames) * 5
            })
        }

        return results

    def test_comparator_initialization(self):
        """Test MultiVideoComparator initialization."""
        comparator = MultiVideoComparator()
        assert comparator.videos_data == []

    def test_add_video_analysis(self, comparator, sample_analysis_results):
        """Test adding video analysis."""
        comparator.add_video_analysis('video1', sample_analysis_results)
        assert len(comparator.videos_data) == 1
        assert comparator.videos_data[0]['name'] == 'video1'

        comparator.add_video_analysis('video2', sample_analysis_results)
        assert len(comparator.videos_data) == 2

    def test_clear_videos(self, comparator, sample_analysis_results):
        """Test clearing videos."""
        comparator.add_video_analysis('video1', sample_analysis_results)
        comparator.add_video_analysis('video2', sample_analysis_results)
        assert len(comparator.videos_data) == 2

        comparator.clear_videos()
        assert len(comparator.videos_data) == 0

    def test_align_by_contact_frame(self, comparator, sample_analysis_results):
        """Test alignment by contact frame."""
        comparator.add_video_analysis('video1', sample_analysis_results)
        comparator.add_video_analysis('video2', sample_analysis_results)

        aligned_data = comparator.align_by_contact_frame()

        assert len(aligned_data) == 2
        for video_data in aligned_data:
            assert 'results' in video_data
            assert 'contact_time_offset' in video_data['results']

    def test_align_by_contact_frame_no_contact(self, comparator):
        """Test alignment when contact phase is missing."""
        results = {
            'skeleton_df': pd.DataFrame({
                'frame': [0, 1],
                'time': [0.0, 0.033],
                'landmarks_3d': [np.zeros((33, 4))] * 2
            }),
            'phases': {
                'approach': {'start': 0, 'end': 1}
                # Missing contact phase
            }
        }

        comparator.add_video_analysis('video1', results)
        aligned_data = comparator.align_by_contact_frame()

        # Should still return data, but not aligned
        assert len(aligned_data) == 1

    def test_generate_comparison_table(self, comparator, sample_analysis_results):
        """Test comparison table generation."""
        comparator.add_video_analysis('video1', sample_analysis_results)
        comparator.add_video_analysis('video2', sample_analysis_results)

        comparison_df = comparator.generate_comparison_table()

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'Video' in comparison_df.columns

    def test_generate_comparison_table_empty(self, comparator):
        """Test comparison table generation with no videos."""
        comparison_df = comparator.generate_comparison_table()

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 0

    def test_plot_comparison_radar(self, comparator, sample_analysis_results):
        """Test radar chart plotting."""
        comparator.add_video_analysis('video1', sample_analysis_results)
        comparator.add_video_analysis('video2', sample_analysis_results)

        fig = comparator.plot_comparison_radar()

        assert fig is not None
        # Check that figure has data
        assert len(fig.data) > 0

    def test_plot_comparison_radar_empty(self, comparator):
        """Test radar chart plotting with no videos."""
        fig = comparator.plot_comparison_radar()

        assert fig is not None
        # Should return empty figure

    def test_plot_comparison_radar_custom_metrics(self, comparator, sample_analysis_results):
        """Test radar chart with custom metrics."""
        comparator.add_video_analysis('video1', sample_analysis_results)

        custom_metrics = ['Jump Height', 'Max Wrist Velocity']
        fig = comparator.plot_comparison_radar(metrics_to_compare=custom_metrics)

        assert fig is not None

    def test_plot_velocity_comparison(self, comparator, sample_analysis_results):
        """Test velocity comparison plotting."""
        comparator.add_video_analysis('video1', sample_analysis_results)
        comparator.add_video_analysis('video2', sample_analysis_results)

        fig = comparator.plot_velocity_comparison()

        assert fig is not None
        # Check that figure has data
        assert len(fig.data) > 0

    def test_calculate_similarity_score(self, comparator, sample_analysis_results):
        """Test similarity score calculation."""
        comparator.add_video_analysis('video1', sample_analysis_results)
        comparator.add_video_analysis('video2', sample_analysis_results)

        similarities = comparator.calculate_similarity_score(0, 1)

        assert isinstance(similarities, dict)
        assert 'overall' in similarities
        assert 0.0 <= similarities['overall'] <= 1.0

    def test_calculate_similarity_score_out_of_range(self, comparator, sample_analysis_results):
        """Test similarity score with out of range indices."""
        comparator.add_video_analysis('video1', sample_analysis_results)

        with pytest.raises(IndexError):
            comparator.calculate_similarity_score(0, 5)

    def test_calculate_similarity_score_same_motion_type(self, comparator):
        """Test similarity with same motion type."""
        results1 = {
            'classification': {'type': 'STRAIGHT'},
            'spatial_data': {},
            'velocity_data': {}
        }
        results2 = {
            'classification': {'type': 'STRAIGHT'},
            'spatial_data': {},
            'velocity_data': {}
        }

        comparator.add_video_analysis('video1', results1)
        comparator.add_video_analysis('video2', results2)

        similarities = comparator.calculate_similarity_score(0, 1)

        assert similarities['same_motion_type'] is True

    def test_calculate_similarity_score_different_motion_type(self, comparator):
        """Test similarity with different motion types."""
        results1 = {
            'classification': {'type': 'STRAIGHT'},
            'spatial_data': {},
            'velocity_data': {}
        }
        results2 = {
            'classification': {'type': 'CIRCULAR'},
            'spatial_data': {},
            'velocity_data': {}
        }

        comparator.add_video_analysis('video1', results1)
        comparator.add_video_analysis('video2', results2)

        similarities = comparator.calculate_similarity_score(0, 1)

        assert similarities['same_motion_type'] is False

    def test_export_comparison_report_csv(self, comparator, sample_analysis_results, tmp_path):
        """Test exporting comparison report as CSV."""
        comparator.add_video_analysis('video1', sample_analysis_results)
        comparator.add_video_analysis('video2', sample_analysis_results)

        output_path = tmp_path / "comparison.csv"
        result_path = comparator.export_comparison_report(str(output_path), format='csv')

        assert output_path.exists()
        assert result_path == str(output_path.absolute())

    def test_export_comparison_report_json(self, comparator, sample_analysis_results, tmp_path):
        """Test exporting comparison report as JSON."""
        comparator.add_video_analysis('video1', sample_analysis_results)

        output_path = tmp_path / "comparison.json"
        result_path = comparator.export_comparison_report(str(output_path), format='json')

        assert output_path.exists()
        assert result_path == str(output_path.absolute())

    def test_export_comparison_report_invalid_format(self, comparator, sample_analysis_results, tmp_path):
        """Test exporting with invalid format."""
        comparator.add_video_analysis('video1', sample_analysis_results)

        output_path = tmp_path / "comparison.txt"

        with pytest.raises(ValueError):
            comparator.export_comparison_report(str(output_path), format='invalid')
