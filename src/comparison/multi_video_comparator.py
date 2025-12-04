"""
Multi-video comparison module for volleyball spike analysis.

This module provides functionality to compare analyses from multiple videos,
including alignment, comparison tables, and visualization.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class MultiVideoComparator:
    """
    Compare multiple video analyses.

    This class provides methods to:
    - Align multiple videos by contact frame
    - Generate comparison tables
    - Create comparison visualizations
    - Calculate similarity metrics

    Attributes:
        videos_data: List of video analysis results.
    """

    def __init__(self) -> None:
        """Initialize the MultiVideoComparator."""
        self.videos_data = []
        logger.info("MultiVideoComparator initialized")

    def add_video_analysis(
        self,
        video_name: str,
        analysis_results: dict
    ) -> None:
        """
        Add a video analysis to the comparison.

        Args:
            video_name: Name or identifier for the video.
            analysis_results: Dictionary containing all analysis results.
        """
        self.videos_data.append({
            'name': video_name,
            'results': analysis_results
        })

        logger.info(f"Added video '{video_name}' to comparison ({len(self.videos_data)} total)")

    def clear_videos(self) -> None:
        """Clear all videos from comparison."""
        self.videos_data = []
        logger.info("Cleared all videos from comparison")

    def align_by_contact_frame(self) -> List[dict]:
        """
        Align all videos by their contact frame.

        This creates a common timeline where t=0 is the contact moment
        for all videos.

        Returns:
            List of aligned video data with adjusted time values.
        """
        aligned_data = []

        for video_data in self.videos_data:
            results = video_data['results']

            # Get contact frame
            if 'phases' not in results or 'contact' not in results['phases']:
                logger.warning(f"Video '{video_data['name']}' has no contact phase, skipping alignment")
                aligned_data.append(video_data)
                continue

            contact_frame = results['phases']['contact']['start']
            skeleton_df = results.get('skeleton_df')

            if skeleton_df is None or contact_frame >= len(skeleton_df):
                logger.warning(f"Invalid data for video '{video_data['name']}'")
                aligned_data.append(video_data)
                continue

            # Get contact time
            contact_time = skeleton_df.iloc[contact_frame]['time']

            # Create aligned skeleton_df with adjusted times
            aligned_skeleton_df = skeleton_df.copy()
            aligned_skeleton_df['time'] = aligned_skeleton_df['time'] - contact_time

            # Adjust angle_df if present
            if 'angles_df' in results and results['angles_df'] is not None:
                aligned_angles_df = results['angles_df'].copy()
                aligned_angles_df['time'] = aligned_angles_df['time'] - contact_time
            else:
                aligned_angles_df = None

            # Adjust velocity_df if present
            if 'velocity_df' in results and results['velocity_df'] is not None:
                aligned_velocity_df = results['velocity_df'].copy()
                aligned_velocity_df['time'] = aligned_velocity_df['time'] - contact_time
            else:
                aligned_velocity_df = None

            # Create aligned results
            aligned_results = results.copy()
            aligned_results['skeleton_df'] = aligned_skeleton_df
            aligned_results['angles_df'] = aligned_angles_df
            aligned_results['velocity_df'] = aligned_velocity_df
            aligned_results['contact_time_offset'] = contact_time

            aligned_data.append({
                'name': video_data['name'],
                'results': aligned_results
            })

        logger.info(f"Aligned {len(aligned_data)} videos by contact frame")
        return aligned_data

    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table for all videos.

        Returns:
            DataFrame with comparison metrics for each video.
        """
        rows = []

        for video_data in self.videos_data:
            name = video_data['name']
            results = video_data['results']

            row = {'Video': name}

            # Motion classification
            if 'classification' in results:
                row['Motion Type'] = results['classification'].get('type_display', 'N/A')
                row['Confidence'] = f"{results['classification'].get('confidence', 0) * 100:.1f}%"

            # Spatial metrics
            if 'spatial_data' in results and results['spatial_data']:
                spatial = results['spatial_data']

                if 'jump_height' in spatial:
                    jh = spatial['jump_height'].get('recommended', np.nan)
                    row['Jump Height (m)'] = f"{jh:.2f}" if not np.isnan(jh) else 'N/A'

                if 'contact_height' in spatial:
                    ch = spatial['contact_height']
                    row['Contact Height (m)'] = f"{ch:.2f}" if not np.isnan(ch) else 'N/A'

                if 'jump_height' in spatial and 'flight_time' in spatial['jump_height']:
                    ft = spatial['jump_height']['flight_time']
                    row['Flight Time (s)'] = f"{ft:.2f}" if not np.isnan(ft) else 'N/A'

                if 'horizontal_displacement' in spatial:
                    hd = spatial['horizontal_displacement'].get('total_displacement', np.nan)
                    row['Horizontal Reach (m)'] = f"{hd:.2f}" if not np.isnan(hd) else 'N/A'

            # Velocity metrics
            if 'velocity_data' in results and results['velocity_data']:
                velocity = results['velocity_data']

                if 'wrist_velocity' in velocity:
                    max_vel = velocity['wrist_velocity'].get('max', np.nan)
                    row['Max Wrist Vel (m/s)'] = f"{max_vel:.2f}" if not np.isnan(max_vel) else 'N/A'

                    at_contact = velocity['wrist_velocity'].get('at_contact', np.nan)
                    row['Vel at Contact (m/s)'] = f"{at_contact:.2f}" if not np.isnan(at_contact) else 'N/A'

                if 'shoulder_angular_velocity' in velocity:
                    ang_vel = velocity['shoulder_angular_velocity'].get('max', np.nan)
                    row['Max Shoulder Ang Vel (°/s)'] = f"{ang_vel:.0f}" if not np.isnan(ang_vel) else 'N/A'

            # Phase timing
            if 'phases' in results and results['phases']:
                phases = results['phases']

                for phase_name in ['approach', 'takeoff', 'arm_swing', 'contact', 'landing']:
                    if phase_name in phases and 'duration' in phases[phase_name]:
                        duration = phases[phase_name]['duration']
                        col_name = f"{phase_name.replace('_', ' ').title()} Duration (s)"
                        row[col_name] = f"{duration:.2f}"

            rows.append(row)

        comparison_df = pd.DataFrame(rows)

        logger.info(f"Generated comparison table with {len(comparison_df)} videos")
        return comparison_df

    def plot_comparison_radar(
        self,
        metrics_to_compare: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create radar chart comparing multiple videos.

        Args:
            metrics_to_compare: List of metric names to include.
                If None, uses default set.

        Returns:
            Plotly Figure object with radar chart.
        """
        if not self.videos_data:
            logger.warning("No videos to compare")
            return go.Figure()

        # Default metrics
        if metrics_to_compare is None:
            metrics_to_compare = [
                'Jump Height',
                'Max Wrist Velocity',
                'Contact Height',
                'Shoulder Angular Velocity',
                'Flight Time'
            ]

        # Normalize metrics to 0-100 scale
        normalization_factors = {
            'Jump Height': 0.8,  # max 0.8m
            'Max Wrist Velocity': 12.0,  # max 12 m/s
            'Contact Height': 3.5,  # max 3.5m
            'Shoulder Angular Velocity': 2000.0,  # max 2000 deg/s
            'Flight Time': 0.5,  # max 0.5s
            'Horizontal Reach': 1.5  # max 1.5m
        }

        fig = go.Figure()

        for video_data in self.videos_data:
            name = video_data['name']
            results = video_data['results']

            values = []
            labels = []

            # Extract metrics
            for metric in metrics_to_compare:
                value = None

                if metric == 'Jump Height' and 'spatial_data' in results:
                    if 'jump_height' in results['spatial_data']:
                        value = results['spatial_data']['jump_height'].get('recommended')

                elif metric == 'Max Wrist Velocity' and 'velocity_data' in results:
                    if 'wrist_velocity' in results['velocity_data']:
                        value = results['velocity_data']['wrist_velocity'].get('max')

                elif metric == 'Contact Height' and 'spatial_data' in results:
                    value = results['spatial_data'].get('contact_height')

                elif metric == 'Shoulder Angular Velocity' and 'velocity_data' in results:
                    if 'shoulder_angular_velocity' in results['velocity_data']:
                        value = results['velocity_data']['shoulder_angular_velocity'].get('max')

                elif metric == 'Flight Time' and 'spatial_data' in results:
                    if 'jump_height' in results['spatial_data']:
                        value = results['spatial_data']['jump_height'].get('flight_time')

                elif metric == 'Horizontal Reach' and 'spatial_data' in results:
                    if 'horizontal_displacement' in results['spatial_data']:
                        value = results['spatial_data']['horizontal_displacement'].get('total_displacement')

                # Normalize value
                if value is not None and not np.isnan(value):
                    norm_factor = normalization_factors.get(metric, 1.0)
                    normalized = min(100, (value / norm_factor) * 100)
                    values.append(normalized)
                    labels.append(metric)

            if values:
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill='toself',
                    name=name
                ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Multi-Video Performance Comparison",
            height=600
        )

        logger.info(f"Created radar chart comparing {len(self.videos_data)} videos")
        return fig

    def plot_velocity_comparison(self) -> go.Figure:
        """
        Create velocity comparison plot for all videos.

        Returns:
            Plotly Figure object with velocity comparison.
        """
        fig = go.Figure()

        aligned_data = self.align_by_contact_frame()

        for video_data in aligned_data:
            name = video_data['name']
            results = video_data['results']

            if 'velocity_df' in results and results['velocity_df'] is not None:
                velocity_df = results['velocity_df']

                if 'wrist_velocity' in velocity_df.columns:
                    fig.add_trace(go.Scatter(
                        x=velocity_df['time'],
                        y=velocity_df['wrist_velocity'],
                        mode='lines',
                        name=f"{name} - Wrist",
                        line=dict(width=2)
                    ))

        # Add vertical line at t=0 (contact moment)
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            annotation_text="Contact",
            annotation_position="top"
        )

        fig.update_layout(
            title="Wrist Velocity Comparison (Aligned by Contact)",
            xaxis_title="Time from Contact (seconds)",
            yaxis_title="Velocity (m/s)",
            height=500,
            hovermode='x unified'
        )

        logger.info(f"Created velocity comparison plot for {len(aligned_data)} videos")
        return fig

    def calculate_similarity_score(
        self,
        video1_idx: int,
        video2_idx: int
    ) -> dict:
        """
        Calculate similarity score between two videos.

        Args:
            video1_idx: Index of first video.
            video2_idx: Index of second video.

        Returns:
            Dictionary with similarity metrics.

        Raises:
            IndexError: If indices are out of range.
        """
        if video1_idx >= len(self.videos_data) or video2_idx >= len(self.videos_data):
            raise IndexError("Video index out of range")

        video1 = self.videos_data[video1_idx]['results']
        video2 = self.videos_data[video2_idx]['results']

        similarities = {}

        # Motion type similarity
        if 'classification' in video1 and 'classification' in video2:
            type1 = video1['classification'].get('type')
            type2 = video2['classification'].get('type')
            similarities['same_motion_type'] = (type1 == type2)

        # Spatial metrics similarity
        if 'spatial_data' in video1 and 'spatial_data' in video2:
            # Jump height similarity
            jh1 = video1['spatial_data'].get('jump_height', {}).get('recommended', np.nan)
            jh2 = video2['spatial_data'].get('jump_height', {}).get('recommended', np.nan)

            if not np.isnan(jh1) and not np.isnan(jh2):
                diff = abs(jh1 - jh2)
                similarity = max(0, 1.0 - diff / max(jh1, jh2))
                similarities['jump_height_similarity'] = float(similarity)

        # Velocity similarity
        if 'velocity_data' in video1 and 'velocity_data' in video2:
            vel1 = video1['velocity_data'].get('wrist_velocity', {}).get('max', np.nan)
            vel2 = video2['velocity_data'].get('wrist_velocity', {}).get('max', np.nan)

            if not np.isnan(vel1) and not np.isnan(vel2):
                diff = abs(vel1 - vel2)
                similarity = max(0, 1.0 - diff / max(vel1, vel2))
                similarities['velocity_similarity'] = float(similarity)

        # Overall similarity (average of available metrics)
        numeric_similarities = [
            v for v in similarities.values()
            if isinstance(v, float)
        ]

        if numeric_similarities:
            similarities['overall'] = float(np.mean(numeric_similarities))
        else:
            similarities['overall'] = 0.0

        logger.info(
            f"Similarity between video {video1_idx} and {video2_idx}: "
            f"{similarities.get('overall', 0):.2%}"
        )

        return similarities

    def export_comparison_report(
        self,
        output_path: str,
        format: str = 'csv'
    ) -> str:
        """
        Export comparison report to file.

        Args:
            output_path: Path to output file.
            format: Output format ('csv' or 'json').

        Returns:
            Absolute path to created file.

        Raises:
            ValueError: If format is not supported.
        """
        if format not in ['csv', 'json']:
            raise ValueError(f"Unsupported format: {format}")

        comparison_df = self.generate_comparison_table()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            comparison_df.to_csv(output_file, index=False)
        elif format == 'json':
            comparison_df.to_json(output_file, orient='records', indent=2)

        logger.info(f"Exported comparison report to: {output_file}")
        return str(output_file.absolute())
