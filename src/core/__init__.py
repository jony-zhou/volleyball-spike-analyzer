"""
Core module for pose extraction and skeleton processing.

This module contains the main logic for extracting pose data from videos
and processing skeleton information.
"""

from src.core.pose_extractor import PoseExtractor
from src.core.skeleton_processor import SkeletonProcessor

__all__ = ["PoseExtractor", "SkeletonProcessor"]
