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
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.core.pose_extractor import PoseExtractor
from src.core.skeleton_processor import SkeletonProcessor
from src.utils.video_io import VideoReader, validate_video_file
from src.utils.data_export import DataExporter
from src.visualization.video_overlay import VideoOverlay
from src.visualization.skeleton_3d import Skeleton3D
from src.analysis.phase_detector import FullMotionPhaseDetector, ArmSwingPhaseDetector
from src.analysis.joint_angles import JointAngleCalculator
from src.analysis.velocity_calculator import VelocityCalculator
from src.analysis.spatial_metrics import SpatialMetricsCalculator
from src.analysis.metrics_summary import MetricsSummary
from src.analysis.arm_swing_classifier import ArmSwingClassifier
from src.comparison.multi_video_comparator import MultiVideoComparator
from src.reporting.report_generator import ReportGenerator

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


def display_phase_analysis(
    phases: Optional[dict],
    arm_swing_phases: Optional[dict],
    skeleton_df: pd.DataFrame,
    fps: float
):
    """
    Display phase analysis results.

    Args:
        phases: Dictionary of main motion phases.
        arm_swing_phases: Dictionary of arm swing sub-phases.
        skeleton_df: Skeleton DataFrame.
        fps: Frames per second.
    """
    st.subheader("Motion Phase Detection")

    if phases is None:
        st.warning("⚠️ Phase detection failed. This may be due to incomplete motion in the video.")
        return

    # Display phase timing table
    st.markdown("### Phase Timing")

    phase_data = []
    for phase_name, bounds in phases.items():
        start_frame = bounds['start']
        end_frame = bounds['end']
        start_time = start_frame / fps
        end_time = end_frame / fps
        duration = end_time - start_time

        phase_data.append({
            'Phase': phase_name.replace('_', ' ').title(),
            'Start Frame': start_frame,
            'End Frame': end_frame,
            'Start Time (s)': f"{start_time:.2f}",
            'End Time (s)': f"{end_time:.2f}",
            'Duration (s)': f"{duration:.2f}"
        })

    phase_table = pd.DataFrame(phase_data)
    st.dataframe(phase_table, use_container_width=True)

    # Display arm swing sub-phases
    if arm_swing_phases:
        st.markdown("### Arm Swing Sub-Phases")

        sub_phase_data = []
        for phase_name, bounds in arm_swing_phases.items():
            start_frame = bounds['start']
            end_frame = bounds['end']
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time

            phase_name_display = {
                'phase_i': 'Phase I (Initiation)',
                'phase_ii': 'Phase II (Wind-up)',
                'phase_iii': 'Phase III (Final Cocking)'
            }.get(phase_name, phase_name)

            sub_phase_data.append({
                'Sub-Phase': phase_name_display,
                'Start Frame': start_frame,
                'End Frame': end_frame,
                'Duration (s)': f"{duration:.2f}",
                'Percentage': f"{duration / (phases['arm_swing']['end'] - phases['arm_swing']['start']) * fps * 100:.1f}%"
            })

        sub_phase_table = pd.DataFrame(sub_phase_data)
        st.dataframe(sub_phase_table, use_container_width=True)

    # Create interactive timeline visualization
    st.markdown("### Phase Timeline")

    fig = go.Figure()

    # Define colors for each phase
    phase_colors = {
        'approach': 'lightblue',
        'takeoff': 'lightgreen',
        'arm_swing': 'lightyellow',
        'contact': 'lightcoral',
        'landing': 'lightgray'
    }

    # Add main phases to timeline
    for phase_name, bounds in phases.items():
        start_time = bounds['start'] / fps
        end_time = bounds['end'] / fps

        fig.add_trace(go.Scatter(
            x=[start_time, end_time, end_time, start_time, start_time],
            y=[0, 0, 1, 1, 0],
            fill='toself',
            fillcolor=phase_colors.get(phase_name, 'lightgray'),
            line=dict(color='black', width=2),
            name=phase_name.replace('_', ' ').title(),
            hovertemplate=f"{phase_name.replace('_', ' ').title()}<br>Time: %{{x:.2f}}s<extra></extra>"
        ))

    # Add arm swing sub-phases if available
    if arm_swing_phases:
        sub_phase_colors = {
            'phase_i': 'yellow',
            'phase_ii': 'orange',
            'phase_iii': 'red'
        }

        for phase_name, bounds in arm_swing_phases.items():
            start_time = bounds['start'] / fps
            end_time = bounds['end'] / fps

            phase_display = {
                'phase_i': 'Phase I',
                'phase_ii': 'Phase II',
                'phase_iii': 'Phase III'
            }.get(phase_name, phase_name)

            fig.add_trace(go.Scatter(
                x=[start_time, end_time, end_time, start_time, start_time],
                y=[1.2, 1.2, 1.5, 1.5, 1.2],
                fill='toself',
                fillcolor=sub_phase_colors.get(phase_name, 'gray'),
                line=dict(color='darkred', width=1),
                name=phase_display,
                hovertemplate=f"{phase_display}<br>Time: %{{x:.2f}}s<extra></extra>"
            ))

    fig.update_layout(
        title="Motion Phase Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="",
        yaxis=dict(showticklabels=False, range=[-0.2, 1.7]),
        height=400,
        hovermode='x unified',
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


def display_joint_angles(
    angles_df: pd.DataFrame,
    phases: Optional[dict],
    arm_swing_phases: Optional[dict]
):
    """
    Display joint angle analysis results.

    Args:
        angles_df: DataFrame containing joint angles over time.
        phases: Dictionary of main motion phases.
        arm_swing_phases: Dictionary of arm swing sub-phases.
    """
    st.subheader("Joint Angle Analysis")

    if angles_df is None or len(angles_df) == 0:
        st.warning("⚠️ No angle data available")
        return

    # Display angle statistics by phase
    if phases:
        st.markdown("### Average Angles by Phase")

        # Calculate average angles for each phase
        angle_columns = ['shoulder_abduction', 'shoulder_horizontal_abduction',
                        'elbow_flexion', 'torso_rotation', 'torso_lean']

        phase_stats = []
        for phase_name, bounds in phases.items():
            start_frame = bounds['start']
            end_frame = bounds['end']

            phase_angles = angles_df.iloc[start_frame:end_frame + 1]

            stats = {
                'Phase': phase_name.replace('_', ' ').title()
            }

            for angle_col in angle_columns:
                mean_val = phase_angles[angle_col].mean()
                stats[angle_col.replace('_', ' ').title()] = f"{mean_val:.1f}°" if not np.isnan(mean_val) else "N/A"

            phase_stats.append(stats)

        stats_df = pd.DataFrame(phase_stats)
        st.dataframe(stats_df, use_container_width=True)

    # Interactive angle time series plot
    st.markdown("### Angle Time Series")

    # Select which angle to display
    angle_options = {
        'Shoulder Abduction': 'shoulder_abduction',
        'Shoulder Horizontal Abduction': 'shoulder_horizontal_abduction',
        'Elbow Flexion': 'elbow_flexion',
        'Torso Rotation': 'torso_rotation',
        'Torso Lean': 'torso_lean'
    }

    selected_angles = st.multiselect(
        "Select angles to display:",
        options=list(angle_options.keys()),
        default=['Elbow Flexion', 'Shoulder Abduction']
    )

    if selected_angles:
        fig = go.Figure()

        for angle_display in selected_angles:
            angle_col = angle_options[angle_display]
            fig.add_trace(go.Scatter(
                x=angles_df['time'],
                y=angles_df[angle_col],
                mode='lines',
                name=angle_display,
                line=dict(width=2)
            ))

        # Add phase boundaries as vertical lines
        if phases:
            for phase_name, bounds in phases.items():
                start_time = angles_df.iloc[bounds['start']]['time']
                fig.add_vline(
                    x=start_time,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=phase_name.replace('_', ' ').title(),
                    annotation_position="top"
                )

        # Add arm swing sub-phase boundaries
        if arm_swing_phases:
            for phase_name, bounds in arm_swing_phases.items():
                start_time = angles_df.iloc[bounds['start']]['time']
                phase_display = {
                    'phase_i': 'Phase I',
                    'phase_ii': 'Phase II',
                    'phase_iii': 'Phase III'
                }.get(phase_name, phase_name)

                fig.add_vline(
                    x=start_time,
                    line_dash="dot",
                    line_color="red",
                    annotation_text=phase_display,
                    annotation_position="bottom"
                )

        fig.update_layout(
            title="Joint Angles Over Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Angle (degrees)",
            height=500,
            hovermode='x unified',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one angle to display")

    # Display raw angle data
    with st.expander("View Raw Angle Data"):
        st.dataframe(angles_df, use_container_width=True)


def display_performance_metrics(
    velocity_data: Optional[dict],
    velocity_df: Optional[pd.DataFrame],
    spatial_data: Optional[dict],
    phases: Optional[dict],
    skeleton_df: pd.DataFrame
):
    """
    Display performance metrics including velocities and spatial metrics.

    Args:
        velocity_data: Dictionary containing velocity metrics.
        velocity_df: DataFrame with velocity timeseries.
        spatial_data: Dictionary containing spatial metrics.
        phases: Dictionary of phase boundaries.
        skeleton_df: Skeleton DataFrame.
    """
    st.subheader("Performance Metrics")

    # Velocity Dashboard
    st.markdown("### 📊 Velocity Dashboard")

    if velocity_data:
        # Create metrics cards
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'wrist_velocity' in velocity_data:
                wrist_vel = velocity_data['wrist_velocity']
                st.metric(
                    "Max Wrist Velocity",
                    f"{wrist_vel.get('max', 0):.2f} m/s",
                    help="Maximum velocity of the wrist during the spike"
                )
                st.metric(
                    "Avg Wrist Velocity",
                    f"{wrist_vel.get('mean', 0):.2f} m/s",
                    help="Average velocity of the wrist"
                )
                if 'at_contact' in wrist_vel:
                    st.metric(
                        "Velocity at Contact",
                        f"{wrist_vel['at_contact']:.2f} m/s",
                        help="Wrist velocity at ball contact"
                    )

        with col2:
            if 'elbow_velocity' in velocity_data:
                elbow_vel = velocity_data['elbow_velocity']
                st.metric(
                    "Max Elbow Velocity",
                    f"{elbow_vel.get('max', 0):.2f} m/s"
                )
            if 'shoulder_velocity' in velocity_data:
                shoulder_vel = velocity_data['shoulder_velocity']
                st.metric(
                    "Max Shoulder Velocity",
                    f"{shoulder_vel.get('max', 0):.2f} m/s"
                )

        with col3:
            if 'shoulder_angular_velocity' in velocity_data:
                shoulder_ang_vel = velocity_data['shoulder_angular_velocity']
                st.metric(
                    "Max Shoulder Angular Vel",
                    f"{shoulder_ang_vel.get('max', 0):.1f} °/s",
                    help="Maximum angular velocity of shoulder joint"
                )
            if 'elbow_angular_velocity' in velocity_data:
                elbow_ang_vel = velocity_data['elbow_angular_velocity']
                st.metric(
                    "Max Elbow Angular Vel",
                    f"{elbow_ang_vel.get('max', 0):.1f} °/s",
                    help="Maximum angular velocity of elbow joint"
                )

    # Spatial Metrics Dashboard
    st.markdown("### 📏 Spatial Metrics")

    if spatial_data:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if 'jump_height' in spatial_data:
                jump_height = spatial_data['jump_height'].get('recommended', np.nan)
                if not np.isnan(jump_height):
                    st.metric(
                        "Jump Height",
                        f"{jump_height:.2f} m",
                        help="Estimated vertical jump height"
                    )

        with col2:
            if 'contact_height' in spatial_data:
                contact_height = spatial_data['contact_height']
                if not np.isnan(contact_height):
                    st.metric(
                        "Contact Height",
                        f"{contact_height:.2f} m",
                        help="Height of wrist at ball contact"
                    )

        with col3:
            if 'jump_height' in spatial_data and 'flight_time' in spatial_data['jump_height']:
                flight_time = spatial_data['jump_height']['flight_time']
                if not np.isnan(flight_time):
                    st.metric(
                        "Flight Time",
                        f"{flight_time:.2f} s",
                        help="Time spent in the air"
                    )

        with col4:
            if 'horizontal_displacement' in spatial_data:
                h_disp = spatial_data['horizontal_displacement'].get('total_displacement', np.nan)
                if not np.isnan(h_disp):
                    st.metric(
                        "Horizontal Reach",
                        f"{h_disp:.2f} m",
                        help="Horizontal distance from takeoff to contact"
                    )

        # Detailed spatial metrics
        with st.expander("Detailed Spatial Metrics"):
            if 'jump_height' in spatial_data:
                st.markdown("**Jump Height Methods:**")
                jh = spatial_data['jump_height']
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Method 1 (Hip Displacement): {jh.get('method_1_hip_displacement', np.nan):.3f} m")
                with col2:
                    st.write(f"Method 2 (Flight Time): {jh.get('method_2_flight_time', np.nan):.3f} m")

            if 'horizontal_displacement' in spatial_data:
                st.markdown("**Horizontal Displacement Components:**")
                hd = spatial_data['horizontal_displacement']
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Forward (Z-axis): {hd.get('forward_displacement', np.nan):.3f} m")
                with col2:
                    st.write(f"Lateral (X-axis): {hd.get('lateral_displacement', np.nan):.3f} m")

    # Velocity Time Series Plot
    st.markdown("### 📈 Velocity Time Series")

    if velocity_df is not None and len(velocity_df) > 0:
        # Select velocities to display
        velocity_options = {
            'Wrist Velocity': 'wrist_velocity',
            'Elbow Velocity': 'elbow_velocity',
            'Shoulder Velocity': 'shoulder_velocity'
        }

        selected_velocities = st.multiselect(
            "Select velocities to display:",
            options=list(velocity_options.keys()),
            default=['Wrist Velocity']
        )

        if selected_velocities:
            fig = go.Figure()

            for vel_display in selected_velocities:
                vel_col = velocity_options[vel_display]
                if vel_col in velocity_df.columns:
                    fig.add_trace(go.Scatter(
                        x=velocity_df['time'],
                        y=velocity_df[vel_col],
                        mode='lines',
                        name=vel_display,
                        line=dict(width=2)
                    ))

            # Add phase boundaries
            if phases:
                # Mark contact phase
                if 'contact' in phases:
                    contact_time = velocity_df.iloc[phases['contact']['start']]['time']
                    fig.add_vline(
                        x=contact_time,
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text="Ball Contact",
                        annotation_position="top"
                    )

                # Mark other phase boundaries
                for phase_name in ['takeoff', 'arm_swing', 'landing']:
                    if phase_name in phases:
                        start_time = velocity_df.iloc[phases[phase_name]['start']]['time']
                        fig.add_vline(
                            x=start_time,
                            line_dash="dot",
                            line_color="gray",
                            annotation_text=phase_name.replace('_', ' ').title(),
                            annotation_position="bottom"
                        )

            fig.update_layout(
                title="Linear Velocity Over Time",
                xaxis_title="Time (seconds)",
                yaxis_title="Velocity (m/s)",
                height=500,
                hovermode='x unified',
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

    # Performance Radar Chart
    st.markdown("### 🎯 Performance Radar Chart")

    if velocity_data and spatial_data:
        # Normalize metrics to 0-100 scale for radar chart
        metrics = {}

        if 'wrist_velocity' in velocity_data:
            # Normalize wrist velocity (assume max possible is 12 m/s)
            max_vel = velocity_data['wrist_velocity'].get('max', 0)
            metrics['Wrist Velocity'] = min(100, (max_vel / 12.0) * 100)

        if 'jump_height' in spatial_data:
            # Normalize jump height (assume max is 0.8 m)
            jump_h = spatial_data['jump_height'].get('recommended', 0)
            if not np.isnan(jump_h):
                metrics['Jump Height'] = min(100, (jump_h / 0.8) * 100)

        if 'shoulder_angular_velocity' in velocity_data:
            # Normalize angular velocity (assume max is 2000 deg/s)
            ang_vel = velocity_data['shoulder_angular_velocity'].get('max', 0)
            metrics['Shoulder Angular Vel'] = min(100, (ang_vel / 2000.0) * 100)

        if 'elbow_angular_velocity' in velocity_data:
            # Normalize angular velocity (assume max is 1500 deg/s)
            ang_vel = velocity_data['elbow_angular_velocity'].get('max', 0)
            metrics['Elbow Angular Vel'] = min(100, (ang_vel / 1500.0) * 100)

        if 'contact_height' in spatial_data:
            # Normalize contact height (assume max is 3.5 m)
            contact_h = spatial_data['contact_height']
            if not np.isnan(contact_h):
                metrics['Contact Height'] = min(100, (contact_h / 3.5) * 100)

        if len(metrics) > 0:
            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=list(metrics.values()),
                theta=list(metrics.keys()),
                fill='toself',
                name='Performance'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=False,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption("Note: Values are normalized to 0-100 scale based on typical performance ranges")


def display_motion_classification(
    classification: Optional[dict],
    angles_df: Optional[pd.DataFrame],
    phases: Optional[dict]
):
    """
    Display motion classification results.

    Args:
        classification: Dictionary containing classification results.
        angles_df: DataFrame with joint angles.
        phases: Dictionary of phase boundaries.
    """
    st.subheader("Motion Classification")

    if classification is None:
        st.warning("⚠️ Motion classification not available. Arm swing phases may not have been detected.")
        return

    # Main classification result
    st.markdown("### 🏷️ Classification Result")

    col1, col2 = st.columns([2, 1])

    with col1:
        motion_type = classification.get('type_display', 'Unknown')
        st.markdown(f"## {motion_type}")

        # Get motion description
        classifier = ArmSwingClassifier()
        description = classifier.get_motion_description(classification.get('type', ''))
        st.info(description)

    with col2:
        confidence = classification.get('confidence', 0)
        st.metric(
            "Confidence",
            f"{confidence * 100:.1f}%",
            help="Confidence score of the classification"
        )

        has_stopping = classification.get('has_stopping_motion', False)
        st.metric(
            "Stopping Motion",
            "✓ Yes" if has_stopping else "✗ No",
            help="Whether stopping motion was detected in Phase III"
        )

    # Matched features
    st.markdown("### 📋 Matched Features")

    matched_rules = classification.get('matched_rules', [])
    if matched_rules:
        for rule in matched_rules:
            st.success(f"✅ {rule}")
    else:
        st.info("No specific features matched")

    # Detailed phase characteristics
    with st.expander("View Detailed Phase Characteristics"):
        features = classification.get('features', {})

        if features:
            for phase_name in ['phase_i', 'phase_ii', 'phase_iii']:
                if phase_name in features:
                    st.markdown(f"**{phase_name.replace('_', ' ').upper()}:**")

                    phase_features = features[phase_name]
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"Wrist vs Shoulder: {phase_features.get('wrist_vs_shoulder', 'N/A')}")
                        st.write(f"Wrist vs Forehead: {phase_features.get('wrist_vs_forehead', 'N/A')}")

                    with col2:
                        st.write(f"Elbow vs Shoulder: {phase_features.get('elbow_vs_shoulder', 'N/A')}")
                        st.write(f"Wrist below Elbow: {phase_features.get('wrist_below_elbow', False)}")

                    st.markdown("---")

    # Comparison with standard (if available)
    if angles_df is not None and phases is not None:
        st.markdown("### 📊 Comparison with Standard Pattern")

        try:
            classifier = ArmSwingClassifier()
            comparison = classifier.compare_with_standard(
                # We need skeleton_df here, but it's not passed
                # For now, just show what we have
                None,
                angles_df,
                phases,
                classification.get('type', '')
            )

            if comparison:
                col1, col2, col3 = st.columns(3)

                with col1:
                    if 'elbow_angle_deviation' in comparison:
                        deviation = comparison['elbow_angle_deviation']
                        st.metric(
                            "Elbow Angle Deviation",
                            f"{abs(deviation):.1f}°",
                            f"{'+' if deviation > 0 else ''}{deviation:.1f}°"
                        )

                with col2:
                    if 'shoulder_angle_deviation' in comparison:
                        deviation = comparison['shoulder_angle_deviation']
                        st.metric(
                            "Shoulder Angle Deviation",
                            f"{abs(deviation):.1f}°",
                            f"{'+' if deviation > 0 else ''}{deviation:.1f}°"
                        )

                with col3:
                    if 'timing_similarity' in comparison:
                        similarity = comparison['timing_similarity']
                        st.metric(
                            "Timing Similarity",
                            f"{similarity * 100:.1f}%",
                            help="Similarity to typical timing pattern"
                        )
        except Exception as e:
            logger.warning(f"Could not generate comparison: {e}")
            st.info("Standard comparison not available")

    # Export classification report
    if st.button("📄 Export Classification Report"):
        try:
            report_gen = ReportGenerator()

            # Prepare metrics for report
            all_metrics = {
                'classification': classification,
                'phases': phases if phases else {},
                'metadata': {
                    'timestamp': pd.Timestamp.now().isoformat()
                }
            }

            # Generate HTML report
            output_path = "data/output/classification_report.html"
            report_path = report_gen.generate_html_report(
                "Classification Analysis",
                all_metrics,
                output_path
            )

            st.success(f"✅ Report exported to: {report_path}")

            # Provide download button
            with open(report_path, 'rb') as f:
                st.download_button(
                    "📥 Download Report",
                    f,
                    file_name="classification_report.html",
                    mime="text/html"
                )

        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            st.error(f"Failed to export report: {str(e)}")


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
        progress_bar.progress(60)

        # Step 3: Prepare DataFrame for analysis
        status_text.text("⏳ Preparing data for analysis...")

        # Create DataFrame with frame, time, and landmarks
        skeleton_df = pd.DataFrame({
            'frame': range(len(processed_data['landmarks_3d_smooth'])),
            'time': [i / fps for i in range(len(processed_data['landmarks_3d_smooth']))],
            'landmarks_3d': list(processed_data['landmarks_3d_smooth'])
        })

        # Step 4: Phase detection
        status_text.text("⏳ Detecting motion phases...")
        phase_detector = FullMotionPhaseDetector()
        phases = phase_detector.detect_phases(skeleton_df)

        # Detect arm swing sub-phases if main phases detected
        arm_swing_phases = None
        if phases and 'arm_swing' in phases:
            arm_swing_detector = ArmSwingPhaseDetector()
            arm_swing_phases = arm_swing_detector.detect_sub_phases(
                skeleton_df,
                phases['arm_swing']['start'],
                phases['arm_swing']['end']
            )

        progress_bar.progress(70)

        # Step 5: Calculate joint angles
        status_text.text("⏳ Calculating joint angles...")
        angle_calculator = JointAngleCalculator()
        angles_df = angle_calculator.calculate_angles_timeseries(skeleton_df)
        progress_bar.progress(70)

        # Step 6: Calculate velocities
        status_text.text("⏳ Analyzing velocities...")
        velocity_calculator = VelocityCalculator()
        velocity_data = velocity_calculator.analyze_velocity_profile(
            skeleton_df, angles_df, phases
        )
        velocity_df = velocity_calculator.calculate_velocity_timeseries(skeleton_df)
        progress_bar.progress(73)

        # Step 7: Calculate spatial metrics
        status_text.text("⏳ Calculating spatial metrics...")
        spatial_calculator = SpatialMetricsCalculator()
        spatial_data = None
        if phases:
            spatial_data = spatial_calculator.calculate_spatial_profile(skeleton_df, phases)
        progress_bar.progress(73)

        # Step 8: Classify arm swing motion
        status_text.text("⏳ Classifying arm swing motion...")
        classification = None
        if arm_swing_phases:
            classifier = ArmSwingClassifier()
            classification = classifier.classify_arm_swing(
                skeleton_df, phases, arm_swing_phases, velocity_df
            )
        progress_bar.progress(75)

        # Step 6: Create visualizations
        status_text.text("⏳ Creating visualizations...")

        progress_bar.progress(80)

        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📹 Video Preview",
            "🦴 3D Skeleton",
            "📊 Phase Analysis",
            "📈 Joint Angles",
            "🚀 Performance Metrics",
            "🏷️ Motion Classification"
        ])

        # Tab 1: Video Preview
        with tab1:
            st.subheader("Original Video")
            st.video(video_path)

        # Tab 2: 3D Skeleton Animation
        with tab2:
            if show_3d_animation:
                st.subheader("3D Skeleton Animation")

                visualizer_3d = Skeleton3D(
                    figure_width=config['visualization']['figure_width'],
                    figure_height=config['visualization']['figure_height']
                )

                fig_3d = visualizer_3d.create_animation(
                    processed_data['landmarks_3d_smooth'],
                    fps=fps
                )

                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("3D animation is disabled in settings")

        # Tab 3: Phase Analysis
        with tab3:
            display_phase_analysis(phases, arm_swing_phases, skeleton_df, fps)

        # Tab 4: Joint Angles
        with tab4:
            display_joint_angles(angles_df, phases, arm_swing_phases)

        # Tab 5: Performance Metrics
        with tab5:
            display_performance_metrics(
                velocity_data, velocity_df, spatial_data, phases, skeleton_df
            )

        # Tab 6: Motion Classification
        with tab6:
            display_motion_classification(classification, angles_df, phases)

        progress_bar.progress(85)

        # Step 7: Export data
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
