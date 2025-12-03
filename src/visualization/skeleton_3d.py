"""
3D skeleton visualization module using Plotly.

This module provides functionality to create interactive 3D visualizations
of pose data using Plotly.
"""

import logging
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class Skeleton3D:
    """
    Create interactive 3D skeleton visualizations.

    This class provides methods for creating animated 3D plots of skeleton
    data using Plotly with MediaPipe's connection definitions.

    Attributes:
        figure_width: Width of the figure in pixels.
        figure_height: Height of the figure in pixels.
        background_color: Background color for the plot.
        show_grid: Whether to show grid lines.
    """

    def __init__(
        self,
        figure_width: int = 1200,
        figure_height: int = 800,
        background_color: str = "white",
        show_grid: bool = True
    ) -> None:
        """
        Initialize the Skeleton3D visualizer.

        Args:
            figure_width: Width of the figure in pixels.
            figure_height: Height of the figure in pixels.
            background_color: Background color for the plot.
            show_grid: Whether to show grid lines.

        Raises:
            ValueError: If dimensions are invalid.
        """
        if figure_width <= 0 or figure_height <= 0:
            raise ValueError("Figure dimensions must be positive")

        self.figure_width = figure_width
        self.figure_height = figure_height
        self.background_color = background_color
        self.show_grid = show_grid

        # Get MediaPipe pose connections
        self.mp_pose = mp.solutions.pose
        self.connections = self.mp_pose.POSE_CONNECTIONS

        logger.info("Skeleton3D initialized")

    def create_frame(
        self,
        landmarks_3d: np.ndarray,
        title: str = "3D Skeleton"
    ) -> go.Figure:
        """
        Create a 3D skeleton visualization for a single frame.

        Args:
            landmarks_3d: 3D landmarks array of shape (33, 3 or 4).
            title: Title for the plot.

        Returns:
            Plotly Figure object.

        Raises:
            ValueError: If landmarks shape is invalid.
        """
        if landmarks_3d.shape[0] != 33:
            raise ValueError(f"Expected 33 landmarks, got {landmarks_3d.shape[0]}")

        # Extract coordinates
        x = landmarks_3d[:, 0]
        y = landmarks_3d[:, 1]
        z = landmarks_3d[:, 2]

        # Create figure
        fig = go.Figure()

        # Add connections (bones)
        for connection in self.connections:
            start_idx = connection[0]
            end_idx = connection[1]

            fig.add_trace(go.Scatter3d(
                x=[x[start_idx], x[end_idx]],
                y=[y[start_idx], y[end_idx]],
                z=[z[start_idx], z[end_idx]],
                mode='lines',
                line=dict(color='red', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add landmarks (joints)
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=6,
                color='blue',
                symbol='circle'
            ),
            name='Landmarks',
            hovertemplate='<b>Landmark %{pointNumber}</b><br>' +
                          'X: %{x:.3f}<br>' +
                          'Y: %{y:.3f}<br>' +
                          'Z: %{z:.3f}<br>' +
                          '<extra></extra>'
        ))

        # Update layout
        fig.update_layout(
            title=title,
            width=self.figure_width,
            height=self.figure_height,
            scene=dict(
                xaxis=dict(
                    title='X',
                    showgrid=self.show_grid,
                    backgroundcolor=self.background_color
                ),
                yaxis=dict(
                    title='Y',
                    showgrid=self.show_grid,
                    backgroundcolor=self.background_color
                ),
                zaxis=dict(
                    title='Z',
                    showgrid=self.show_grid,
                    backgroundcolor=self.background_color
                ),
                aspectmode='data'
            ),
            paper_bgcolor=self.background_color
        )

        return fig

    def create_animation(
        self,
        landmarks_sequence: np.ndarray,
        fps: float = 30.0,
        title: str = "3D Skeleton Animation"
    ) -> go.Figure:
        """
        Create an animated 3D skeleton visualization.

        Args:
            landmarks_sequence: Array of shape (num_frames, 33, 3 or 4).
            fps: Frames per second for animation speed.
            title: Title for the plot.

        Returns:
            Plotly Figure object with animation.

        Raises:
            ValueError: If landmarks_sequence shape is invalid.
        """
        if landmarks_sequence.ndim != 3:
            raise ValueError(
                f"Expected 3D array (frames, landmarks, coords), got shape {landmarks_sequence.shape}"
            )

        if landmarks_sequence.shape[1] != 33:
            raise ValueError(f"Expected 33 landmarks, got {landmarks_sequence.shape[1]}")

        num_frames = landmarks_sequence.shape[0]

        # Create frames for animation
        frames = []
        for frame_idx in range(num_frames):
            landmarks = landmarks_sequence[frame_idx]

            x = landmarks[:, 0]
            y = landmarks[:, 1]
            z = landmarks[:, 2]

            # Create traces for this frame
            frame_data = []

            # Add connections
            for connection in self.connections:
                start_idx = connection[0]
                end_idx = connection[1]

                frame_data.append(go.Scatter3d(
                    x=[x[start_idx], x[end_idx]],
                    y=[y[start_idx], y[end_idx]],
                    z=[z[start_idx], z[end_idx]],
                    mode='lines',
                    line=dict(color='red', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Add landmarks
            frame_data.append(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(size=6, color='blue'),
                name='Landmarks',
                hovertemplate='<b>Landmark %{pointNumber}</b><br>' +
                              'X: %{x:.3f}<br>' +
                              'Y: %{y:.3f}<br>' +
                              'Z: %{z:.3f}<br>' +
                              '<extra></extra>'
            ))

            frames.append(go.Frame(
                data=frame_data,
                name=str(frame_idx)
            ))

        # Create initial figure with first frame
        initial_landmarks = landmarks_sequence[0]
        fig = self.create_frame(initial_landmarks, title)

        # Add frames to figure
        fig.frames = frames

        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [
                            None,
                            {
                                'frame': {'duration': int(1000 / fps), 'redraw': True},
                                'fromcurrent': True,
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }
                        ]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [
                            [None],
                            {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }
                        ]
                    }
                ],
                'x': 0.1,
                'y': 0,
                'xanchor': 'left',
                'yanchor': 'top'
            }],
            sliders=[{
                'active': 0,
                'steps': [
                    {
                        'args': [
                            [str(i)],
                            {
                                'frame': {'duration': 0, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }
                        ],
                        'label': str(i),
                        'method': 'animate'
                    }
                    for i in range(num_frames)
                ],
                'x': 0.1,
                'len': 0.85,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        )

        logger.info(f"Created animation with {num_frames} frames")

        return fig

    def create_comparison(
        self,
        landmarks_list: List[np.ndarray],
        labels: List[str],
        title: str = "3D Skeleton Comparison"
    ) -> go.Figure:
        """
        Create a side-by-side comparison of multiple skeletons.

        Args:
            landmarks_list: List of 3D landmarks arrays, each of shape (33, 3 or 4).
            labels: List of labels for each skeleton.
            title: Title for the plot.

        Returns:
            Plotly Figure object with subplots.

        Raises:
            ValueError: If inputs are invalid or lengths don't match.
        """
        if len(landmarks_list) != len(labels):
            raise ValueError("Number of landmark sets must match number of labels")

        if len(landmarks_list) == 0:
            raise ValueError("landmarks_list is empty")

        num_skeletons = len(landmarks_list)

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=num_skeletons,
            subplot_titles=labels,
            specs=[[{'type': 'scatter3d'} for _ in range(num_skeletons)]]
        )

        # Add each skeleton to its subplot
        for col_idx, (landmarks, label) in enumerate(zip(landmarks_list, labels), start=1):
            if landmarks.shape[0] != 33:
                raise ValueError(f"Expected 33 landmarks for {label}, got {landmarks.shape[0]}")

            x = landmarks[:, 0]
            y = landmarks[:, 1]
            z = landmarks[:, 2]

            # Add connections
            for connection in self.connections:
                start_idx = connection[0]
                end_idx = connection[1]

                fig.add_trace(go.Scatter3d(
                    x=[x[start_idx], x[end_idx]],
                    y=[y[start_idx], y[end_idx]],
                    z=[z[start_idx], z[end_idx]],
                    mode='lines',
                    line=dict(color='red', width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=1, col=col_idx)

            # Add landmarks
            fig.add_trace(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(size=6, color='blue'),
                name=label,
                hovertemplate='<b>Landmark %{pointNumber}</b><br>' +
                              'X: %{x:.3f}<br>' +
                              'Y: %{y:.3f}<br>' +
                              'Z: %{z:.3f}<br>' +
                              '<extra></extra>'
            ), row=1, col=col_idx)

        # Update layout
        fig.update_layout(
            title=title,
            width=self.figure_width * num_skeletons // 2,
            height=self.figure_height,
            paper_bgcolor=self.background_color
        )

        logger.info(f"Created comparison with {num_skeletons} skeletons")

        return fig

    def save_figure(
        self,
        fig: go.Figure,
        output_path: str,
        format: str = 'html'
    ) -> None:
        """
        Save a Plotly figure to file.

        Args:
            fig: Plotly Figure object.
            output_path: Path to save the figure.
            format: Output format ('html', 'png', 'jpg', 'svg', 'pdf').

        Raises:
            ValueError: If format is not supported.
        """
        valid_formats = ['html', 'png', 'jpg', 'jpeg', 'svg', 'pdf']

        if format.lower() not in valid_formats:
            raise ValueError(
                f"Format '{format}' not supported. Valid formats: {valid_formats}"
            )

        if format.lower() == 'html':
            fig.write_html(output_path)
        else:
            fig.write_image(output_path, format=format)

        logger.info(f"Saved figure to: {output_path}")
