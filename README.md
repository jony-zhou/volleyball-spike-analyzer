# Volleyball Spike Analyzer

A computer vision-based system for analyzing volleyball spike techniques using pose estimation and 3D visualization.

## Features

- **Video Upload**: Support for multiple video formats (MP4, AVI, MOV, MKV)
- **Pose Extraction**: Real-time skeleton tracking using MediaPipe Pose
- **2D & 3D Coordinates**: Extract both 2D and 3D landmark positions
- **3D Visualization**: Interactive 3D skeleton animation using Plotly
- **Data Export**: Export pose data to CSV/JSON for further analysis
- **Web Interface**: User-friendly interface built with Streamlit

## Project Structure

```
volleyball-spike-analyzer/
├── .gitignore
├── requirements.txt
├── README.md
├── config/
│   └── config.yaml          # Configuration settings
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── pose_extractor.py      # MediaPipe pose extraction
│   │   └── skeleton_processor.py  # Skeleton data processing
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── video_overlay.py       # 2D skeleton overlay
│   │   └── skeleton_3d.py         # 3D visualization
│   └── utils/
│       ├── __init__.py
│       ├── video_io.py            # Video I/O operations
│       └── data_export.py         # Data export utilities
├── tests/
│   └── test_pose_extractor.py
├── data/
│   ├── input/                     # Input videos
│   ├── output/                    # Processed results
│   └── cache/                     # Temporary cache
└── app.py                         # Streamlit application

```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd volleyball-spike-analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit web interface:
```bash
streamlit run app.py
```

The application will open in your default web browser (typically at `http://localhost:8501`).

### Basic Workflow

1. **Upload Video**: Upload a volleyball spike video (MP4, AVI, MOV, or MKV)
2. **Extract Pose**: The system automatically extracts skeleton data
3. **View Results**:
   - Watch the original video with 2D skeleton overlay
   - Explore the interactive 3D skeleton animation
   - Download pose data for further analysis

### Configuration

Edit `config/config.yaml` to customize:
- MediaPipe model settings (accuracy vs. speed)
- Video processing parameters
- Visualization styles
- Export formats and options

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

Run linting and type checking:
```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

## Technical Details

### MediaPipe Pose

This project uses MediaPipe Pose for human pose estimation:
- **33 body landmarks** in 2D (x, y, visibility)
- **33 body landmarks** in 3D (x, y, z, visibility)
- Real-time performance on CPU
- High accuracy for athletic movements

### Key Landmarks for Spike Analysis

- Shoulders (left/right)
- Elbows (left/right)
- Wrists (left/right)
- Hips (left/right)
- Knees (left/right)
- Ankles (left/right)

## Roadmap

### Phase 1: MVP (Current)
- [x] Project structure setup
- [ ] Basic pose extraction
- [ ] 2D video overlay
- [ ] 3D skeleton visualization
- [ ] Simple Streamlit interface

### Phase 2: Analysis
- [ ] Joint angle calculation
- [ ] Velocity and acceleration metrics
- [ ] Spike phase detection (approach, jump, hit, landing)
- [ ] Performance metrics dashboard

### Phase 3: Advanced Features
- [ ] Multi-player comparison
- [ ] Technique scoring system
- [ ] Movement recommendations
- [ ] Historical trend analysis

## Contributing

This is a personal side project, but suggestions and improvements are welcome!

## License

MIT License - Feel free to use and modify for your own projects.

## Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for pose estimation
- [Streamlit](https://streamlit.io/) for the web interface
- [Plotly](https://plotly.com/) for 3D visualization

## Contact

For questions or feedback, please open an issue on GitHub.
