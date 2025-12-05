# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Volleyball Spike Analyzer is a computer vision-based system for analyzing volleyball spike techniques using MediaPipe Pose estimation. The application provides 2D skeleton overlay on videos and 3D interactive visualizations using Streamlit as the web interface.

## Common Commands

### Development
```bash
# Run the Streamlit application
streamlit run app.py

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Virtual Environment (Windows)
```bash
python -m venv venv
venv\Scripts\activate
```

## Architecture

### Core Pipeline
The system follows a pipeline architecture:
1. **Video Input** → 2. **Pose Extraction (MediaPipe)** → 3. **Skeleton Processing** → 4. **Visualization (2D/3D)** → 5. **Data Export**

### Module Structure

- **src/core/**: MediaPipe pose extraction and skeleton data processing
  - `pose_extractor.py`: Extracts 33 body landmarks in both 2D (x, y, visibility) and 3D (x, y, z, visibility)
  - `skeleton_processor.py`: Post-processes skeleton data, calculates joint angles, velocities

- **src/visualization/**: Rendering components
  - `video_overlay.py`: Draws 2D skeleton overlay on video frames using OpenCV
  - `skeleton_3d.py`: Creates interactive 3D skeleton animations using Plotly

- **src/utils/**: Support utilities
  - `video_io.py`: Video reading/writing operations with OpenCV
  - `data_export.py`: Export pose data to CSV/JSON formats

- **app.py**: Main Streamlit application (not yet implemented)

### Key Dependencies

- **MediaPipe (>=0.10.21)**: Pose estimation with 33 landmarks
- **OpenCV (>=4.10.0)**: Video I/O and 2D overlay rendering
- **NumPy (<2.0.0)**: Locked to 1.x to avoid MediaPipe conflicts
- **Streamlit (>=1.42.0)**: Web UI framework
- **Plotly (>=5.24.0)**: 3D interactive visualizations
- **Pandas (>=2.2.0)**: Data handling
- **PyArrow (>=17.0.0)**: Parquet format support for better performance

## Configuration

All settings are centralized in `config/config.yaml`:

- **mediapipe**: Model complexity (0-2), confidence thresholds, segmentation settings
- **video**: Supported formats, max file size (500MB), resize dimensions (1280x720)
- **visualization**: Colors, line thickness, figure dimensions
- **export**: Output formats (CSV/JSON), coordinate systems (2D/3D), frame sampling
- **analysis**: Key joints for spike analysis, angle/velocity calculation flags, smoothing window
- **performance**: GPU usage, batch processing, worker threads
- **paths**: Input/output/cache/temp directories under `data/`

### Key Analysis Joints
The system focuses on 12 key joints for spike analysis: shoulders, elbows, wrists, hips, knees, ankles (left/right pairs).

## Development Status

Currently in Phase 1 (MVP):
- Project structure is set up
- Core modules are defined but not yet implemented
- Main application (`app.py`) needs to be created
- Tests need to be written

## Important Notes

- NumPy is locked to 1.x versions to prevent conflicts with MediaPipe
- The system processes video frame-by-frame; performance settings in config control GPU usage and batch processing
- MediaPipe provides 33 landmarks total, but spike analysis focuses on 12 key joints
- 3D coordinates from MediaPipe are relative to hip center, not absolute world coordinates
- Supported video formats: MP4, AVI, MOV, MKV (max 500MB)
