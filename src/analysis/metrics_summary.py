"""
Metrics summary module for volleyball spike analysis.

This module provides functionality to aggregate all analysis results
and export them in various formats.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetricsSummary:
    """
    Aggregate and export all analysis metrics.

    This class consolidates results from:
    - Phase detection
    - Joint angle calculation
    - Velocity analysis
    - Spatial metrics

    Attributes:
        skeleton_df: DataFrame containing skeleton data.
        phase_info: Dictionary of phase boundaries.
        angle_data: DataFrame or dict of angle data.
        velocity_data: Dictionary of velocity metrics.
        spatial_data: Dictionary of spatial metrics.
    """

    def __init__(
        self,
        skeleton_df: pd.DataFrame,
        phase_info: Optional[dict] = None,
        arm_swing_phases: Optional[dict] = None,
        angle_data: Optional[pd.DataFrame] = None,
        velocity_data: Optional[dict] = None,
        spatial_data: Optional[dict] = None
    ) -> None:
        """
        Initialize the MetricsSummary.

        Args:
            skeleton_df: DataFrame with skeleton data.
            phase_info: Dictionary of main phase boundaries.
            arm_swing_phases: Dictionary of arm swing sub-phases.
            angle_data: DataFrame with joint angles.
            velocity_data: Dictionary with velocity metrics.
            spatial_data: Dictionary with spatial metrics.
        """
        self.skeleton_df = skeleton_df
        self.phase_info = phase_info
        self.arm_swing_phases = arm_swing_phases
        self.angle_data = angle_data
        self.velocity_data = velocity_data
        self.spatial_data = spatial_data

        logger.info("MetricsSummary initialized")

    def _serialize_numpy(self, obj):
        """
        Convert numpy types to Python native types for JSON serialization.

        Args:
            obj: Object to convert.

        Returns:
            Converted object.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._serialize_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_numpy(item) for item in obj]
        else:
            return obj

    def generate_phase_summary(self) -> dict:
        """
        Generate summary of phase detection results.

        Returns:
            Dictionary with phase timing information.
        """
        if self.phase_info is None:
            return {}

        fps = 1.0 / self.skeleton_df['time'].diff().mean()

        summary = {}
        for phase_name, bounds in self.phase_info.items():
            start_frame = bounds['start']
            end_frame = bounds['end']
            start_time = self.skeleton_df.iloc[start_frame]['time']
            end_time = self.skeleton_df.iloc[end_frame]['time']

            summary[phase_name] = {
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'start_time': float(start_time),
                'end_time': float(end_time),
                'duration': float(end_time - start_time),
                'num_frames': int(end_frame - start_frame + 1)
            }

        # Add arm swing sub-phases if available
        if self.arm_swing_phases:
            summary['arm_swing_sub_phases'] = {}
            for phase_name, bounds in self.arm_swing_phases.items():
                start_frame = bounds['start']
                end_frame = bounds['end']
                start_time = self.skeleton_df.iloc[start_frame]['time']
                end_time = self.skeleton_df.iloc[end_frame]['time']

                summary['arm_swing_sub_phases'][phase_name] = {
                    'start_frame': int(start_frame),
                    'end_frame': int(end_frame),
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'duration': float(end_time - start_time)
                }

        return summary

    def generate_angle_summary(self) -> dict:
        """
        Generate summary of joint angle analysis.

        Returns:
            Dictionary with angle statistics by phase.
        """
        if self.angle_data is None or self.phase_info is None:
            return {}

        summary = {}
        angle_columns = ['shoulder_abduction', 'shoulder_horizontal_abduction',
                        'elbow_flexion', 'torso_rotation', 'torso_lean']

        for phase_name, bounds in self.phase_info.items():
            start_frame = bounds['start']
            end_frame = bounds['end']

            phase_angles = self.angle_data.iloc[start_frame:end_frame + 1]
            phase_summary = {}

            for angle_col in angle_columns:
                if angle_col in phase_angles.columns:
                    values = phase_angles[angle_col].dropna()
                    if len(values) > 0:
                        phase_summary[angle_col] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max())
                        }

            if phase_summary:
                summary[phase_name] = phase_summary

        return summary

    def generate_velocity_summary(self) -> dict:
        """
        Generate summary of velocity analysis.

        Returns:
            Dictionary with velocity metrics.
        """
        if self.velocity_data is None:
            return {}

        summary = {}

        # Extract key metrics (excluding full timeseries 'values')
        for metric_name, metric_data in self.velocity_data.items():
            if isinstance(metric_data, dict):
                summary[metric_name] = {
                    k: v for k, v in metric_data.items()
                    if k != 'values'  # Exclude timeseries data
                }

        return summary

    def generate_spatial_summary(self) -> dict:
        """
        Generate summary of spatial metrics.

        Returns:
            Dictionary with spatial metrics.
        """
        if self.spatial_data is None:
            return {}

        summary = {}

        # Jump height
        if 'jump_height' in self.spatial_data:
            summary['jump_height'] = {
                k: v for k, v in self.spatial_data['jump_height'].items()
                if k != 'com_trajectory'  # Exclude trajectory data
            }

        # Contact height
        if 'contact_height' in self.spatial_data:
            summary['contact_height'] = float(self.spatial_data['contact_height'])

        # Horizontal displacement
        if 'horizontal_displacement' in self.spatial_data:
            summary['horizontal_displacement'] = self.spatial_data['horizontal_displacement']

        return summary

    def generate_summary_dict(self) -> dict:
        """
        Generate complete summary dictionary.

        Returns:
            Dictionary containing all metrics:
            {
                'metadata': {...},
                'phases': {...},
                'angles': {...},
                'velocities': {...},
                'spatial': {...}
            }
        """
        fps = 1.0 / self.skeleton_df['time'].diff().mean() if len(self.skeleton_df) > 1 else 30.0

        summary = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_frames': int(len(self.skeleton_df)),
                'fps': float(fps),
                'duration': float(self.skeleton_df['time'].iloc[-1] - self.skeleton_df['time'].iloc[0])
            },
            'phases': self.generate_phase_summary(),
            'angles': self.generate_angle_summary(),
            'velocities': self.generate_velocity_summary(),
            'spatial': self.generate_spatial_summary()
        }

        # Serialize numpy types
        summary = self._serialize_numpy(summary)

        logger.info("Generated complete metrics summary")
        return summary

    def export_to_json(self, output_path: str) -> str:
        """
        Export summary to JSON file.

        Args:
            output_path: Path to output JSON file.

        Returns:
            Absolute path to created file.
        """
        summary = self.generate_summary_dict()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported summary to JSON: {output_file}")
        return str(output_file.absolute())

    def export_phases_to_csv(self, output_path: str) -> str:
        """
        Export phase information to CSV file.

        Args:
            output_path: Path to output CSV file.

        Returns:
            Absolute path to created file.
        """
        if self.phase_info is None:
            logger.warning("No phase information available for export")
            return ""

        phase_summary = self.generate_phase_summary()

        # Convert to DataFrame
        rows = []
        for phase_name, data in phase_summary.items():
            if phase_name != 'arm_swing_sub_phases':
                row = {'phase': phase_name}
                row.update(data)
                rows.append(row)

        # Add sub-phases if available
        if 'arm_swing_sub_phases' in phase_summary:
            for sub_phase_name, data in phase_summary['arm_swing_sub_phases'].items():
                row = {'phase': f"arm_swing_{sub_phase_name}"}
                row.update(data)
                rows.append(row)

        df = pd.DataFrame(rows)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False)

        logger.info(f"Exported phases to CSV: {output_file}")
        return str(output_file.absolute())

    def export_velocity_timeseries_to_csv(self, output_path: str) -> str:
        """
        Export velocity timeseries to CSV file.

        Args:
            output_path: Path to output CSV file.

        Returns:
            Absolute path to created file.
        """
        if self.velocity_data is None:
            logger.warning("No velocity data available for export")
            return ""

        # Build DataFrame from timeseries data
        data = {
            'frame': self.skeleton_df['frame'].values,
            'time': self.skeleton_df['time'].values
        }

        for metric_name, metric_data in self.velocity_data.items():
            if isinstance(metric_data, dict) and 'values' in metric_data:
                data[metric_name] = metric_data['values']

        df = pd.DataFrame(data)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False)

        logger.info(f"Exported velocity timeseries to CSV: {output_file}")
        return str(output_file.absolute())

    def export_summary_metrics_to_csv(self, output_path: str) -> str:
        """
        Export summary metrics to CSV file (one row per metric).

        Args:
            output_path: Path to output CSV file.

        Returns:
            Absolute path to created file.
        """
        summary = self.generate_summary_dict()

        rows = []

        # Flatten the nested dictionary structure
        def flatten_dict(d: dict, prefix: str = '') -> None:
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key

                if isinstance(value, dict):
                    flatten_dict(value, full_key)
                elif not isinstance(value, (list, np.ndarray)):
                    rows.append({
                        'metric': full_key,
                        'value': value
                    })

        # Flatten all sections except metadata
        for section in ['phases', 'angles', 'velocities', 'spatial']:
            if section in summary:
                flatten_dict(summary[section], section)

        df = pd.DataFrame(rows)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_file, index=False)

        logger.info(f"Exported summary metrics to CSV: {output_file}")
        return str(output_file.absolute())

    def generate_text_report(self) -> str:
        """
        Generate human-readable text report.

        Returns:
            Formatted text report as string.
        """
        summary = self.generate_summary_dict()

        lines = [
            "=" * 60,
            "VOLLEYBALL SPIKE ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Generated: {summary['metadata']['timestamp']}",
            f"Duration: {summary['metadata']['duration']:.2f} seconds",
            f"Frames: {summary['metadata']['num_frames']}",
            f"FPS: {summary['metadata']['fps']:.1f}",
            ""
        ]

        # Phase information
        if summary['phases']:
            lines.extend([
                "-" * 60,
                "MOTION PHASES",
                "-" * 60,
                ""
            ])

            for phase_name, data in summary['phases'].items():
                if phase_name != 'arm_swing_sub_phases':
                    lines.append(
                        f"{phase_name.upper()}: "
                        f"{data['start_time']:.2f}s - {data['end_time']:.2f}s "
                        f"(duration: {data['duration']:.2f}s)"
                    )

            lines.append("")

        # Velocity metrics
        if summary['velocities']:
            lines.extend([
                "-" * 60,
                "VELOCITY METRICS",
                "-" * 60,
                ""
            ])

            for metric_name, data in summary['velocities'].items():
                lines.append(f"{metric_name.upper().replace('_', ' ')}:")
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"  {key}: {value:.2f}")
                lines.append("")

        # Spatial metrics
        if summary['spatial']:
            lines.extend([
                "-" * 60,
                "SPATIAL METRICS",
                "-" * 60,
                ""
            ])

            if 'jump_height' in summary['spatial']:
                lines.append("JUMP HEIGHT:")
                for key, value in summary['spatial']['jump_height'].items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        lines.append(f"  {key}: {value:.3f} m")
                lines.append("")

            if 'contact_height' in summary['spatial']:
                lines.append(f"CONTACT HEIGHT: {summary['spatial']['contact_height']:.3f} m")
                lines.append("")

            if 'horizontal_displacement' in summary['spatial']:
                lines.append("HORIZONTAL DISPLACEMENT:")
                for key, value in summary['spatial']['horizontal_displacement'].items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        lines.append(f"  {key}: {value:.3f} m")
                lines.append("")

        lines.append("=" * 60)

        report = "\n".join(lines)
        logger.info("Generated text report")

        return report

    def export_text_report(self, output_path: str) -> str:
        """
        Export text report to file.

        Args:
            output_path: Path to output text file.

        Returns:
            Absolute path to created file.
        """
        report = self.generate_text_report()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Exported text report: {output_file}")
        return str(output_file.absolute())
