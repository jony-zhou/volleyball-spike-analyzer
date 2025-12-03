"""
Volleyball Spike Analyzer - Streamlit Application.

This is the main web application for analyzing volleyball spike techniques
using pose estimation and 3D visualization.
"""

import logging
from pathlib import Path
from typing import Optional

import streamlit as st
import yaml
import numpy as np

from src.core.pose_extractor import PoseExtractor
from src.core.skeleton_processor import SkeletonProcessor
from src.utils.video_io import VideoReader, validate_video_file
from src.utils.data_export import DataExporter
from src.visualization.video_overlay import VideoOverlay
from src.visualization.skeleton_3d import Skeleton3D

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Volleyball Spike Analyzer",
    page_icon="🏐",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("config/config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main application function."""
    # Load configuration
    config = load_config()

    # Title and description
    st.title("🏐 Volleyball Spike Analyzer")
    st.markdown("""
    Analyze volleyball spike techniques using computer vision and pose estimation.
    Upload a video to extract 2D/3D skeleton data and visualize movement patterns.
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Settings")

    # MediaPipe settings
    st.sidebar.subheader("Pose Detection")
    model_complexity = st.sidebar.select_slider(
        "Model Complexity",
        options=[0, 1, 2],
        value=config['mediapipe']['model_complexity'],
        help="Higher values are more accurate but slower"
    )

    min_detection_confidence = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=config['mediapipe']['min_detection_confidence'],
        step=0.05
    )

    # Visualization settings
    st.sidebar.subheader("Visualization")
    show_2d_overlay = st.sidebar.checkbox("Show 2D Overlay", value=True)
    show_3d_animation = st.sidebar.checkbox("Show 3D Animation", value=True)

    # Analysis settings
    st.sidebar.subheader("Analysis")
    calculate_angles = st.sidebar.checkbox("Calculate Joint Angles", value=True)
    calculate_velocities = st.sidebar.checkbox("Calculate Velocities", value=True)

    # Main content area
    st.header("📹 Upload Video")

    uploaded_file = st.file_uploader(
        "Choose a volleyball spike video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help=f"Maximum file size: {config['video']['max_file_size_mb']}MB"
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_input_path = Path(config['paths']['temp_dir']) / uploaded_file.name
        temp_input_path.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_input_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ Uploaded: {uploaded_file.name}")

        # Display video info
        with VideoReader(str(temp_input_path)) as video_reader:
            props = video_reader.get_properties()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Resolution", f"{props['width']}x{props['height']}")
        col2.metric("FPS", f"{props['fps']:.1f}")
        col3.metric("Frames", props['frame_count'])
        col4.metric("Duration", f"{props['duration']:.1f}s")

        # Process button
        if st.button("🚀 Start Analysis", type="primary"):
            process_video(
                str(temp_input_path),
                config,
                model_complexity,
                min_detection_confidence,
                show_2d_overlay,
                show_3d_animation,
                calculate_angles,
                calculate_velocities
            )


def process_video(
    video_path: str,
    config: dict,
    model_complexity: int,
    min_detection_confidence: float,
    show_2d_overlay: bool,
    show_3d_animation: bool,
    calculate_angles: bool,
    calculate_velocities: bool
):
    """Process uploaded video and display results."""
    try:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Extract poses
        status_text.text("⏳ Extracting poses from video...")
        progress_bar.progress(10)

        with PoseExtractor(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        ) as extractor:
            landmarks_sequence = extractor.extract_from_video(video_path)

        progress_bar.progress(40)

        # Check if any poses were detected
        valid_frames = [lm for lm in landmarks_sequence if lm is not None]
        if not valid_frames:
            st.error("❌ No poses detected in the video. Please try a different video.")
            return

        st.info(f"✓ Detected poses in {len(valid_frames)}/{len(landmarks_sequence)} frames")

        # Step 2: Process skeleton data
        status_text.text("⏳ Processing skeleton data...")
        progress_bar.progress(50)

        with VideoReader(video_path) as video_reader:
            fps = video_reader.get_properties()['fps']

        processor = SkeletonProcessor(
            calculate_angles=calculate_angles,
            calculate_velocities=calculate_velocities
        )

        processed_data = processor.process_sequence(landmarks_sequence, fps=fps)
        progress_bar.progress(70)

        # Step 3: Create visualizations
        status_text.text("⏳ Creating visualizations...")

        # 3D Animation
        if show_3d_animation:
            st.header("🎬 3D Skeleton Animation")

            visualizer_3d = Skeleton3D(
                figure_width=config['visualization']['figure_width'],
                figure_height=config['visualization']['figure_height']
            )

            fig_3d = visualizer_3d.create_animation(
                processed_data['landmarks_3d_smooth'],
                fps=fps
            )

            st.plotly_chart(fig_3d, use_container_width=True)

        progress_bar.progress(85)

        # Display analysis results
        if calculate_angles and 'angles' in processed_data:
            st.header("📊 Joint Angle Analysis")

            # Show angles for a sample frame (middle of video)
            mid_frame = len(processed_data['angles']) // 2
            angles_mid = processed_data['angles'][mid_frame]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Right Side")
                for joint in ['right_shoulder', 'right_elbow', 'right_hip', 'right_knee']:
                    if joint in angles_mid:
                        st.metric(joint.replace('_', ' ').title(), f"{angles_mid[joint]:.1f}°")

            with col2:
                st.subheader("Left Side")
                for joint in ['left_shoulder', 'left_elbow', 'left_hip', 'left_knee']:
                    if joint in angles_mid:
                        st.metric(joint.replace('_', ' ').title(), f"{angles_mid[joint]:.1f}°")

        # Step 4: Export data
        status_text.text("⏳ Exporting data...")
        progress_bar.progress(95)

        output_dir = Path(config['paths']['output_dir'])
        exporter = DataExporter(str(output_dir))

        # Export landmarks
        landmarks_csv = exporter.export_landmarks_to_csv(
            processed_data['landmarks_3d_smooth'],
            "landmarks_3d.csv",
            fps=fps
        )

        # Export angles if available
        if calculate_angles and 'angles' in processed_data:
            angles_csv = exporter.export_angles_to_csv(
                processed_data['angles'],
                "joint_angles.csv",
                fps=fps
            )

        # Create and export summary
        summary = exporter.create_summary_report(
            processed_data['landmarks_3d_smooth'],
            processed_data.get('angles'),
            processed_data.get('velocities'),
            fps=fps
        )

        summary_json = exporter.export_summary(summary, "analysis_summary.json")

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        # Download section
        st.header("💾 Download Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            with open(landmarks_csv, 'rb') as f:
                st.download_button(
                    "📄 Download 3D Landmarks (CSV)",
                    f,
                    file_name="landmarks_3d.csv",
                    mime="text/csv"
                )

        if calculate_angles and 'angles' in processed_data:
            with col2:
                with open(angles_csv, 'rb') as f:
                    st.download_button(
                        "📄 Download Joint Angles (CSV)",
                        f,
                        file_name="joint_angles.csv",
                        mime="text/csv"
                    )

        with col3:
            with open(summary_json, 'rb') as f:
                st.download_button(
                    "📄 Download Summary (JSON)",
                    f,
                    file_name="analysis_summary.json",
                    mime="application/json"
                )

        st.success("🎉 All results are ready for download!")

    except Exception as e:
        logger.error(f"Error processing video: {e}", exc_info=True)
        st.error(f"❌ Error processing video: {str(e)}")


if __name__ == "__main__":
    main()
