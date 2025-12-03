"""
Data export utilities module.

This module provides functionality for exporting pose data to various
formats including CSV, JSON, and Parquet.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataExporter:
    """
    Export pose data to various file formats.

    This class provides methods for exporting skeleton data, joint angles,
    and velocities to CSV, JSON, and Parquet formats.

    Attributes:
        output_dir: Directory for output files.
        include_timestamps: Whether to include frame timestamps.
    """

    def __init__(
        self,
        output_dir: str,
        include_timestamps: bool = True
    ) -> None:
        """
        Initialize the DataExporter.

        Args:
            output_dir: Path to output directory.
            include_timestamps: Whether to include timestamps in exports.
        """
        self.output_dir = Path(output_dir)
        self.include_timestamps = include_timestamps

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"DataExporter initialized with output_dir: {output_dir}")

    def export_landmarks_to_csv(
        self,
        landmarks_sequence: np.ndarray,
        output_filename: str,
        fps: float = 30.0,
        landmark_names: Optional[List[str]] = None
    ) -> Path:
        """
        Export landmark sequence to CSV file.

        Args:
            landmarks_sequence: Array of shape (num_frames, num_landmarks, num_coords).
            output_filename: Output filename (without path).
            fps: Video frame rate for timestamp calculation.
            landmark_names: Optional list of landmark names.

        Returns:
            Path to the created CSV file.

        Raises:
            ValueError: If landmarks_sequence shape is invalid.
        """
        if landmarks_sequence.ndim != 3:
            raise ValueError(
                f"Expected 3D array (frames, landmarks, coords), got shape {landmarks_sequence.shape}"
            )

        num_frames, num_landmarks, num_coords = landmarks_sequence.shape

        # Prepare data
        data = []
        for frame_idx in range(num_frames):
            row = {}

            # Add timestamp if requested
            if self.include_timestamps:
                row['frame'] = frame_idx
                row['timestamp'] = frame_idx / fps

            # Add landmark coordinates
            for landmark_idx in range(num_landmarks):
                if landmark_names and landmark_idx < len(landmark_names):
                    prefix = landmark_names[landmark_idx]
                else:
                    prefix = f"landmark_{landmark_idx}"

                coords = landmarks_sequence[frame_idx, landmark_idx]

                if num_coords >= 3:
                    row[f"{prefix}_x"] = coords[0]
                    row[f"{prefix}_y"] = coords[1]
                    row[f"{prefix}_z"] = coords[2]
                if num_coords >= 4:
                    row[f"{prefix}_visibility"] = coords[3]

            data.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(data)
        output_path = self.output_dir / output_filename

        df.to_csv(output_path, index=False)
        logger.info(f"Exported landmarks to CSV: {output_path}")

        return output_path

    def export_angles_to_csv(
        self,
        angles_sequence: List[Dict[str, float]],
        output_filename: str,
        fps: float = 30.0
    ) -> Path:
        """
        Export joint angles sequence to CSV file.

        Args:
            angles_sequence: List of angle dictionaries for each frame.
            output_filename: Output filename (without path).
            fps: Video frame rate for timestamp calculation.

        Returns:
            Path to the created CSV file.

        Raises:
            ValueError: If angles_sequence is empty.
        """
        if not angles_sequence:
            raise ValueError("angles_sequence is empty")

        # Prepare data
        data = []
        for frame_idx, angles in enumerate(angles_sequence):
            row = {}

            # Add timestamp if requested
            if self.include_timestamps:
                row['frame'] = frame_idx
                row['timestamp'] = frame_idx / fps

            # Add angles
            row.update(angles)
            data.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(data)
        output_path = self.output_dir / output_filename

        df.to_csv(output_path, index=False)
        logger.info(f"Exported angles to CSV: {output_path}")

        return output_path

    def export_velocities_to_csv(
        self,
        velocities: np.ndarray,
        output_filename: str,
        fps: float = 30.0,
        landmark_names: Optional[List[str]] = None
    ) -> Path:
        """
        Export velocities to CSV file.

        Args:
            velocities: Array of shape (num_frames, num_landmarks).
            output_filename: Output filename (without path).
            fps: Video frame rate for timestamp calculation.
            landmark_names: Optional list of landmark names.

        Returns:
            Path to the created CSV file.

        Raises:
            ValueError: If velocities shape is invalid.
        """
        if velocities.ndim != 2:
            raise ValueError(
                f"Expected 2D array (frames, landmarks), got shape {velocities.shape}"
            )

        num_frames, num_landmarks = velocities.shape

        # Prepare data
        data = []
        for frame_idx in range(num_frames):
            row = {}

            # Add timestamp if requested
            if self.include_timestamps:
                row['frame'] = frame_idx
                row['timestamp'] = frame_idx / fps

            # Add velocities
            for landmark_idx in range(num_landmarks):
                if landmark_names and landmark_idx < len(landmark_names):
                    col_name = f"{landmark_names[landmark_idx]}_velocity"
                else:
                    col_name = f"landmark_{landmark_idx}_velocity"

                row[col_name] = velocities[frame_idx, landmark_idx]

            data.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(data)
        output_path = self.output_dir / output_filename

        df.to_csv(output_path, index=False)
        logger.info(f"Exported velocities to CSV: {output_path}")

        return output_path

    def export_to_json(
        self,
        data: Dict[str, Any],
        output_filename: str,
        indent: int = 2
    ) -> Path:
        """
        Export data to JSON file.

        Args:
            data: Dictionary to export.
            output_filename: Output filename (without path).
            indent: JSON indentation level.

        Returns:
            Path to the created JSON file.
        """
        output_path = self.output_dir / output_filename

        # Convert numpy arrays to lists for JSON serialization
        data_serializable = self._make_json_serializable(data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_serializable, f, indent=indent)

        logger.info(f"Exported data to JSON: {output_path}")

        return output_path

    def export_to_parquet(
        self,
        df: pd.DataFrame,
        output_filename: str,
        compression: str = 'snappy'
    ) -> Path:
        """
        Export DataFrame to Parquet file.

        Args:
            df: DataFrame to export.
            output_filename: Output filename (without path).
            compression: Compression algorithm ('snappy', 'gzip', 'brotli', or None).

        Returns:
            Path to the created Parquet file.
        """
        output_path = self.output_dir / output_filename

        df.to_parquet(output_path, compression=compression, index=False)
        logger.info(f"Exported data to Parquet: {output_path}")

        return output_path

    def export_summary(
        self,
        summary_data: Dict[str, Any],
        output_filename: str = "summary.json"
    ) -> Path:
        """
        Export summary statistics to JSON.

        Args:
            summary_data: Dictionary containing summary information.
            output_filename: Output filename (without path).

        Returns:
            Path to the created JSON file.
        """
        return self.export_to_json(summary_data, output_filename)

    def _make_json_serializable(self, data: Any) -> Any:
        """
        Convert data to JSON-serializable format.

        Args:
            data: Input data (can contain numpy arrays, etc.).

        Returns:
            JSON-serializable version of the data.
        """
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_json_serializable(item) for item in data]
        else:
            return data

    def create_summary_report(
        self,
        landmarks_sequence: np.ndarray,
        angles_sequence: Optional[List[Dict[str, float]]] = None,
        velocities: Optional[np.ndarray] = None,
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Create a summary report of the analysis.

        Args:
            landmarks_sequence: Array of landmarks.
            angles_sequence: Optional list of angles.
            velocities: Optional velocities array.
            fps: Video frame rate.

        Returns:
            Dictionary containing summary statistics.
        """
        num_frames = len(landmarks_sequence)
        duration = num_frames / fps

        summary = {
            'video_info': {
                'num_frames': num_frames,
                'fps': fps,
                'duration_seconds': duration
            },
            'landmarks_info': {
                'num_landmarks': landmarks_sequence.shape[1],
                'num_coords': landmarks_sequence.shape[2]
            }
        }

        # Add angle statistics if available
        if angles_sequence:
            angle_stats = {}
            all_angles = {}

            # Collect all angles
            for angles in angles_sequence:
                for joint, angle in angles.items():
                    if joint not in all_angles:
                        all_angles[joint] = []
                    all_angles[joint].append(angle)

            # Calculate statistics
            for joint, values in all_angles.items():
                angle_stats[joint] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

            summary['angle_statistics'] = angle_stats

        # Add velocity statistics if available
        if velocities is not None:
            summary['velocity_statistics'] = {
                'mean': float(np.mean(velocities)),
                'std': float(np.std(velocities)),
                'max': float(np.max(velocities))
            }

        return summary
