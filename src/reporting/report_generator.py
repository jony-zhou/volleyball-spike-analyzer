"""
Report generation module for volleyball spike analysis.

This module provides functionality to generate analysis reports
in various formats including HTML and text.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate analysis reports in various formats.

    This class provides methods to create comprehensive reports
    including video information, phase analysis, joint angles,
    velocities, spatial metrics, and motion classification.

    Note: PDF generation requires additional libraries (reportlab or weasyprint)
    which are optional dependencies.
    """

    def __init__(self) -> None:
        """Initialize the ReportGenerator."""
        logger.info("ReportGenerator initialized")

    def _format_metric(self, value, unit: str = '', decimals: int = 2) -> str:
        """
        Format metric value for display.

        Args:
            value: Metric value.
            unit: Unit string (e.g., 'm', 's', '°').
            decimals: Number of decimal places.

        Returns:
            Formatted string.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 'N/A'

        if isinstance(value, (int, float)):
            return f"{value:.{decimals}f} {unit}".strip()

        return str(value)

    def generate_html_report(
        self,
        video_name: str,
        all_metrics: dict,
        output_path: str
    ) -> str:
        """
        Generate interactive HTML report.

        Args:
            video_name: Name of the analyzed video.
            all_metrics: Dictionary containing all analysis metrics.
            output_path: Path to output HTML file.

        Returns:
            Absolute path to created file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Build HTML content
        html_parts = []

        # Header
        html_parts.append(self._generate_html_header(video_name))

        # Video Information
        html_parts.append(self._generate_video_info_section(all_metrics))

        # Motion Classification
        if 'classification' in all_metrics:
            html_parts.append(self._generate_classification_section(all_metrics['classification']))

        # Phase Analysis
        if 'phases' in all_metrics:
            html_parts.append(self._generate_phase_section(all_metrics['phases']))

        # Spatial Metrics
        if 'spatial_data' in all_metrics:
            html_parts.append(self._generate_spatial_section(all_metrics['spatial_data']))

        # Velocity Metrics
        if 'velocity_data' in all_metrics:
            html_parts.append(self._generate_velocity_section(all_metrics['velocity_data']))

        # Joint Angles
        if 'angles_summary' in all_metrics:
            html_parts.append(self._generate_angles_section(all_metrics['angles_summary']))

        # Footer
        html_parts.append(self._generate_html_footer())

        # Write to file
        html_content = '\n'.join(html_parts)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Generated HTML report: {output_file}")
        return str(output_file.absolute())

    def _generate_html_header(self, video_name: str) -> str:
        """Generate HTML header."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Volleyball Spike Analysis Report - {video_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section-title {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px 15px;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            border-radius: 4px;
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            display: block;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }}
        .classification-badge {{
            display: inline-block;
            padding: 10px 20px;
            background: #28a745;
            color: white;
            border-radius: 25px;
            font-size: 1.2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .confidence {{
            color: #666;
            font-size: 0.9em;
        }}
        .feature-list {{
            list-style-type: none;
            padding-left: 0;
        }}
        .feature-list li {{
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        .feature-list li:before {{
            content: "✓ ";
            color: #28a745;
            font-weight: bold;
            margin-right: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th {{
            background-color: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            color: #666;
            padding: 20px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏐 Volleyball Spike Analysis Report</h1>
        <p>Video: <strong>{video_name}</strong></p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""

    def _generate_video_info_section(self, metrics: dict) -> str:
        """Generate video information section."""
        metadata = metrics.get('metadata', {})

        num_frames = metadata.get('num_frames', 'N/A')
        fps = metadata.get('fps', 'N/A')
        duration = metadata.get('duration', 'N/A')

        return f"""
    <div class="section">
        <h2 class="section-title">📹 Video Information</h2>
        <div class="metric">
            <span class="metric-label">Total Frames</span>
            <span class="metric-value">{num_frames}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Frame Rate</span>
            <span class="metric-value">{self._format_metric(fps, 'fps', 1)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Duration</span>
            <span class="metric-value">{self._format_metric(duration, 's', 2)}</span>
        </div>
    </div>
"""

    def _generate_classification_section(self, classification: dict) -> str:
        """Generate motion classification section."""
        motion_type = classification.get('type_display', 'Unknown')
        confidence = classification.get('confidence', 0) * 100
        matched_rules = classification.get('matched_rules', [])
        has_stopping = classification.get('has_stopping_motion', False)

        features_html = '\n'.join([
            f"<li>{rule}</li>"
            for rule in matched_rules
        ])

        return f"""
    <div class="section">
        <h2 class="section-title">🏷️ Motion Classification</h2>
        <div class="classification-badge">{motion_type}</div>
        <span class="confidence">Confidence: {confidence:.1f}%</span>

        <h3>Matched Features:</h3>
        <ul class="feature-list">
            {features_html}
        </ul>

        <p><strong>Stopping Motion Detected:</strong> {'Yes' if has_stopping else 'No'}</p>
    </div>
"""

    def _generate_phase_section(self, phases: dict) -> str:
        """Generate phase analysis section."""
        table_rows = []

        phase_order = ['approach', 'takeoff', 'arm_swing', 'contact', 'landing']

        for phase_name in phase_order:
            if phase_name in phases:
                phase_data = phases[phase_name]
                table_rows.append(f"""
            <tr>
                <td>{phase_name.replace('_', ' ').title()}</td>
                <td>{phase_data.get('start_frame', 'N/A')}</td>
                <td>{phase_data.get('end_frame', 'N/A')}</td>
                <td>{self._format_metric(phase_data.get('start_time'), 's')}</td>
                <td>{self._format_metric(phase_data.get('end_time'), 's')}</td>
                <td>{self._format_metric(phase_data.get('duration'), 's')}</td>
            </tr>
        """)

        return f"""
    <div class="section">
        <h2 class="section-title">📊 Phase Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Phase</th>
                    <th>Start Frame</th>
                    <th>End Frame</th>
                    <th>Start Time</th>
                    <th>End Time</th>
                    <th>Duration</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
    </div>
"""

    def _generate_spatial_section(self, spatial_data: dict) -> str:
        """Generate spatial metrics section."""
        jump_height = spatial_data.get('jump_height', {})
        recommended_jh = jump_height.get('recommended', np.nan)
        method1 = jump_height.get('method_1_hip_displacement', np.nan)
        method2 = jump_height.get('method_2_flight_time', np.nan)
        flight_time = jump_height.get('flight_time', np.nan)

        contact_height = spatial_data.get('contact_height', np.nan)

        h_disp = spatial_data.get('horizontal_displacement', {})
        total_disp = h_disp.get('total_displacement', np.nan)
        forward_disp = h_disp.get('forward_displacement', np.nan)
        lateral_disp = h_disp.get('lateral_displacement', np.nan)

        return f"""
    <div class="section">
        <h2 class="section-title">📏 Spatial Metrics</h2>

        <h3>Jump Performance</h3>
        <div class="metric">
            <span class="metric-label">Jump Height (Recommended)</span>
            <span class="metric-value">{self._format_metric(recommended_jh, 'm')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Method 1 (Hip Displacement)</span>
            <span class="metric-value">{self._format_metric(method1, 'm')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Method 2 (Flight Time)</span>
            <span class="metric-value">{self._format_metric(method2, 'm')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Flight Time</span>
            <span class="metric-value">{self._format_metric(flight_time, 's')}</span>
        </div>

        <h3>Contact Point</h3>
        <div class="metric">
            <span class="metric-label">Contact Height</span>
            <span class="metric-value">{self._format_metric(contact_height, 'm')}</span>
        </div>

        <h3>Horizontal Displacement</h3>
        <div class="metric">
            <span class="metric-label">Total Displacement</span>
            <span class="metric-value">{self._format_metric(total_disp, 'm')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Forward (Z-axis)</span>
            <span class="metric-value">{self._format_metric(forward_disp, 'm')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Lateral (X-axis)</span>
            <span class="metric-value">{self._format_metric(lateral_disp, 'm')}</span>
        </div>
    </div>
"""

    def _generate_velocity_section(self, velocity_data: dict) -> str:
        """Generate velocity metrics section."""
        wrist_vel = velocity_data.get('wrist_velocity', {})
        elbow_vel = velocity_data.get('elbow_velocity', {})
        shoulder_ang_vel = velocity_data.get('shoulder_angular_velocity', {})
        elbow_ang_vel = velocity_data.get('elbow_angular_velocity', {})

        return f"""
    <div class="section">
        <h2 class="section-title">🚀 Velocity Metrics</h2>

        <h3>Linear Velocities</h3>
        <div class="metric">
            <span class="metric-label">Max Wrist Velocity</span>
            <span class="metric-value">{self._format_metric(wrist_vel.get('max'), 'm/s')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Avg Wrist Velocity</span>
            <span class="metric-value">{self._format_metric(wrist_vel.get('mean'), 'm/s')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Velocity at Contact</span>
            <span class="metric-value">{self._format_metric(wrist_vel.get('at_contact'), 'm/s')}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Max Elbow Velocity</span>
            <span class="metric-value">{self._format_metric(elbow_vel.get('max'), 'm/s')}</span>
        </div>

        <h3>Angular Velocities</h3>
        <div class="metric">
            <span class="metric-label">Max Shoulder Angular Velocity</span>
            <span class="metric-value">{self._format_metric(shoulder_ang_vel.get('max'), '°/s', 0)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Max Elbow Angular Velocity</span>
            <span class="metric-value">{self._format_metric(elbow_ang_vel.get('max'), '°/s', 0)}</span>
        </div>
    </div>
"""

    def _generate_angles_section(self, angles_summary: dict) -> str:
        """Generate joint angles section."""
        # This would typically show average angles by phase
        return f"""
    <div class="section">
        <h2 class="section-title">📐 Joint Angles</h2>
        <p>Detailed joint angle analysis available in exported CSV files.</p>
    </div>
"""

    def _generate_html_footer(self) -> str:
        """Generate HTML footer."""
        return """
    <div class="footer">
        <p>Generated by Volleyball Spike Analyzer</p>
        <p>🏐 Powered by MediaPipe and Claude Code</p>
    </div>
</body>
</html>
"""

    def generate_text_report(
        self,
        video_name: str,
        all_metrics: dict,
        output_path: str
    ) -> str:
        """
        Generate text report.

        Args:
            video_name: Name of the analyzed video.
            all_metrics: Dictionary containing all analysis metrics.
            output_path: Path to output text file.

        Returns:
            Absolute path to created file.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "=" * 70,
            "VOLLEYBALL SPIKE ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Video: {video_name}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]

        # Video Information
        if 'metadata' in all_metrics:
            metadata = all_metrics['metadata']
            lines.extend([
                "-" * 70,
                "VIDEO INFORMATION",
                "-" * 70,
                "",
                f"Total Frames: {metadata.get('num_frames', 'N/A')}",
                f"Frame Rate: {self._format_metric(metadata.get('fps'), 'fps', 1)}",
                f"Duration: {self._format_metric(metadata.get('duration'), 's')}",
                ""
            ])

        # Motion Classification
        if 'classification' in all_metrics:
            classification = all_metrics['classification']
            lines.extend([
                "-" * 70,
                "MOTION CLASSIFICATION",
                "-" * 70,
                "",
                f"Type: {classification.get('type_display', 'Unknown')}",
                f"Confidence: {classification.get('confidence', 0) * 100:.1f}%",
                f"Stopping Motion: {'Yes' if classification.get('has_stopping_motion') else 'No'}",
                "",
                "Matched Features:",
            ])

            for rule in classification.get('matched_rules', []):
                lines.append(f"  - {rule}")

            lines.append("")

        # Add other sections as needed...

        lines.append("=" * 70)

        report_content = "\n".join(lines)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"Generated text report: {output_file}")
        return str(output_file.absolute())
