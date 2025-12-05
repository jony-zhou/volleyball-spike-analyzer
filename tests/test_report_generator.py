"""
Unit tests for report generator module.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.reporting.report_generator import ReportGenerator


class TestReportGenerator:
    """Test cases for ReportGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create a ReportGenerator instance."""
        return ReportGenerator()

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing."""
        return {
            'metadata': {
                'num_frames': 150,
                'fps': 30.0,
                'duration': 5.0
            },
            'classification': {
                'type': 'STRAIGHT',
                'type_display': 'Straight',
                'confidence': 0.85,
                'matched_rules': [
                    'High arc motion with stopping',
                    'Phase I: Wrist and elbow above shoulder',
                    'Phase II: Wrist above forehead'
                ],
                'has_stopping_motion': True
            },
            'phases': {
                'approach': {
                    'start_frame': 0,
                    'end_frame': 30,
                    'start_time': 0.0,
                    'end_time': 1.0,
                    'duration': 1.0
                },
                'takeoff': {
                    'start_frame': 30,
                    'end_frame': 45,
                    'start_time': 1.0,
                    'end_time': 1.5,
                    'duration': 0.5
                },
                'arm_swing': {
                    'start_frame': 45,
                    'end_frame': 90,
                    'start_time': 1.5,
                    'end_time': 3.0,
                    'duration': 1.5
                },
                'contact': {
                    'start_frame': 80,
                    'end_frame': 90,
                    'start_time': 2.67,
                    'end_time': 3.0,
                    'duration': 0.33
                },
                'landing': {
                    'start_frame': 90,
                    'end_frame': 120,
                    'start_time': 3.0,
                    'end_time': 4.0,
                    'duration': 1.0
                }
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
                'elbow_velocity': {
                    'max': 6.2,
                    'mean': 2.5
                },
                'shoulder_angular_velocity': {
                    'max': 1500.0,
                    'mean': 650.0
                },
                'elbow_angular_velocity': {
                    'max': 1200.0,
                    'mean': 500.0
                }
            },
            'angles_summary': {
                'shoulder_abduction': {
                    'mean': 145.0,
                    'max': 170.0,
                    'min': 90.0
                }
            }
        }

    def test_generator_initialization(self):
        """Test ReportGenerator initialization."""
        generator = ReportGenerator()
        assert generator is not None

    def test_format_metric_float(self, generator):
        """Test formatting float metrics."""
        result = generator._format_metric(3.14159, 'm', decimals=2)
        assert result == '3.14 m'

    def test_format_metric_int(self, generator):
        """Test formatting integer metrics."""
        result = generator._format_metric(42, 's', decimals=1)
        assert result == '42.0 s'

    def test_format_metric_none(self, generator):
        """Test formatting None values."""
        result = generator._format_metric(None, 'm')
        assert result == 'N/A'

    def test_format_metric_nan(self, generator):
        """Test formatting NaN values."""
        result = generator._format_metric(np.nan, 'm')
        assert result == 'N/A'

    def test_format_metric_no_unit(self, generator):
        """Test formatting without unit."""
        result = generator._format_metric(3.14, decimals=1)
        assert result == '3.1'

    def test_generate_html_header(self, generator):
        """Test HTML header generation."""
        header = generator._generate_html_header('test_video.mp4')

        assert isinstance(header, str)
        assert '<!DOCTYPE html>' in header
        assert 'test_video.mp4' in header
        assert '<style>' in header

    def test_generate_video_info_section(self, generator, sample_metrics):
        """Test video information section generation."""
        section = generator._generate_video_info_section(sample_metrics)

        assert isinstance(section, str)
        assert 'Video Information' in section
        assert '150' in section  # num_frames
        assert '30' in section  # fps

    def test_generate_classification_section(self, generator, sample_metrics):
        """Test classification section generation."""
        section = generator._generate_classification_section(
            sample_metrics['classification']
        )

        assert isinstance(section, str)
        assert 'Motion Classification' in section
        assert 'Straight' in section
        assert '85.0%' in section  # confidence

    def test_generate_phase_section(self, generator, sample_metrics):
        """Test phase analysis section generation."""
        section = generator._generate_phase_section(sample_metrics['phases'])

        assert isinstance(section, str)
        assert 'Phase Analysis' in section
        assert 'Approach' in section
        assert 'Takeoff' in section

    def test_generate_spatial_section(self, generator, sample_metrics):
        """Test spatial metrics section generation."""
        section = generator._generate_spatial_section(sample_metrics['spatial_data'])

        assert isinstance(section, str)
        assert 'Spatial Metrics' in section
        assert 'Jump Height' in section
        assert '0.45' in section  # recommended jump height

    def test_generate_velocity_section(self, generator, sample_metrics):
        """Test velocity metrics section generation."""
        section = generator._generate_velocity_section(sample_metrics['velocity_data'])

        assert isinstance(section, str)
        assert 'Velocity Metrics' in section
        assert 'Wrist Velocity' in section
        assert '8.5' in section  # max wrist velocity

    def test_generate_angles_section(self, generator, sample_metrics):
        """Test joint angles section generation."""
        section = generator._generate_angles_section(sample_metrics['angles_summary'])

        assert isinstance(section, str)
        assert 'Joint Angles' in section

    def test_generate_html_footer(self, generator):
        """Test HTML footer generation."""
        footer = generator._generate_html_footer()

        assert isinstance(footer, str)
        assert '</body>' in footer
        assert '</html>' in footer

    def test_generate_html_report(self, generator, sample_metrics, tmp_path):
        """Test complete HTML report generation."""
        output_path = tmp_path / "report.html"

        result_path = generator.generate_html_report(
            'test_video.mp4',
            sample_metrics,
            str(output_path)
        )

        # Check file was created
        assert output_path.exists()
        assert result_path == str(output_path.absolute())

        # Check file content
        content = output_path.read_text(encoding='utf-8')
        assert '<!DOCTYPE html>' in content
        assert 'test_video.mp4' in content
        assert 'Straight' in content

    def test_generate_html_report_creates_directory(self, generator, sample_metrics, tmp_path):
        """Test that report generation creates output directory."""
        output_path = tmp_path / "reports" / "subdir" / "report.html"

        generator.generate_html_report(
            'test_video.mp4',
            sample_metrics,
            str(output_path)
        )

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_generate_text_report(self, generator, sample_metrics, tmp_path):
        """Test complete text report generation."""
        output_path = tmp_path / "report.txt"

        result_path = generator.generate_text_report(
            'test_video.mp4',
            sample_metrics,
            str(output_path)
        )

        # Check file was created
        assert output_path.exists()
        assert result_path == str(output_path.absolute())

        # Check file content
        content = output_path.read_text(encoding='utf-8')
        assert 'VOLLEYBALL SPIKE ANALYSIS REPORT' in content
        assert 'test_video.mp4' in content
        assert 'Straight' in content

    def test_generate_text_report_creates_directory(self, generator, sample_metrics, tmp_path):
        """Test that text report generation creates output directory."""
        output_path = tmp_path / "reports" / "text" / "report.txt"

        generator.generate_text_report(
            'test_video.mp4',
            sample_metrics,
            str(output_path)
        )

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_generate_html_report_minimal_metrics(self, generator, tmp_path):
        """Test HTML report with minimal metrics."""
        minimal_metrics = {
            'metadata': {'num_frames': 100, 'fps': 30, 'duration': 3.33}
        }

        output_path = tmp_path / "minimal_report.html"

        result_path = generator.generate_html_report(
            'test.mp4',
            minimal_metrics,
            str(output_path)
        )

        assert output_path.exists()

        # Should still generate valid HTML
        content = output_path.read_text(encoding='utf-8')
        assert '<!DOCTYPE html>' in content
        assert '</html>' in content

    def test_generate_text_report_minimal_metrics(self, generator, tmp_path):
        """Test text report with minimal metrics."""
        minimal_metrics = {
            'metadata': {'num_frames': 100}
        }

        output_path = tmp_path / "minimal_report.txt"

        result_path = generator.generate_text_report(
            'test.mp4',
            minimal_metrics,
            str(output_path)
        )

        assert output_path.exists()

        # Should still generate valid report
        content = output_path.read_text(encoding='utf-8')
        assert 'VOLLEYBALL SPIKE ANALYSIS REPORT' in content
