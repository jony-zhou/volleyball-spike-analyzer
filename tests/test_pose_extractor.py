"""
Unit tests for the PoseExtractor module.
"""

import numpy as np
import pytest
import cv2

from src.core.pose_extractor import PoseExtractor


class TestPoseExtractor:
    """Test cases for PoseExtractor class."""

    def test_initialization_with_valid_parameters(self):
        """Test successful initialization with valid parameters."""
        extractor = PoseExtractor(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        assert extractor.model_complexity == 1
        assert extractor.min_detection_confidence == 0.5
        assert extractor.min_tracking_confidence == 0.5

        extractor.close()

    def test_initialization_with_invalid_model_complexity(self):
        """Test initialization fails with invalid model complexity."""
        with pytest.raises(ValueError, match="model_complexity must be 0, 1, or 2"):
            PoseExtractor(model_complexity=3)

    def test_initialization_with_invalid_confidence(self):
        """Test initialization fails with invalid confidence values."""
        with pytest.raises(ValueError, match="min_detection_confidence must be between"):
            PoseExtractor(min_detection_confidence=1.5)

        with pytest.raises(ValueError, match="min_tracking_confidence must be between"):
            PoseExtractor(min_tracking_confidence=-0.1)

    def test_context_manager(self):
        """Test PoseExtractor works as context manager."""
        with PoseExtractor() as extractor:
            assert extractor is not None
            assert hasattr(extractor, 'pose')

    def test_extract_pose_with_invalid_frame(self):
        """Test extract_pose fails with invalid frame."""
        with PoseExtractor() as extractor:
            with pytest.raises(ValueError, match="Input frame is invalid or empty"):
                extractor.extract_pose(None)

            with pytest.raises(ValueError, match="Input frame is invalid or empty"):
                extractor.extract_pose(np.array([]))

    def test_extract_pose_with_valid_frame(self):
        """Test extract_pose with a synthetic frame."""
        with PoseExtractor() as extractor:
            # Create a simple test frame (solid color)
            test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

            # May return None if no person detected (expected for blank frame)
            result = extractor.extract_pose(test_frame)

            # Result should be None or dict with correct structure
            if result is not None:
                assert 'landmarks_2d' in result
                assert 'landmarks_3d' in result
                assert result['landmarks_2d'].shape == (33, 3)
                assert result['landmarks_3d'].shape == (33, 4)

    def test_extract_pose_returns_correct_shape(self):
        """Test that landmarks have correct shape when detected."""
        with PoseExtractor() as extractor:
            # Create a more complex test image
            test_frame = cv2.imread('test_image.jpg') if False else np.random.randint(
                0, 255, (480, 640, 3), dtype=np.uint8
            )

            result = extractor.extract_pose(test_frame)

            if result is not None:
                # Check 2D landmarks shape (33 landmarks, x/y/visibility)
                assert result['landmarks_2d'].shape[0] == 33
                assert result['landmarks_2d'].shape[1] == 3

                # Check 3D landmarks shape (33 landmarks, x/y/z/visibility)
                assert result['landmarks_3d'].shape[0] == 33
                assert result['landmarks_3d'].shape[1] == 4

                # Check coordinate ranges (normalized 0-1)
                assert np.all(result['landmarks_2d'][:, :2] >= 0)
                assert np.all(result['landmarks_2d'][:, :2] <= 1)

                # Check visibility values (0-1)
                assert np.all(result['landmarks_2d'][:, 2] >= 0)
                assert np.all(result['landmarks_2d'][:, 2] <= 1)

    def test_extract_from_video_with_invalid_path(self):
        """Test extract_from_video fails with non-existent file."""
        with PoseExtractor() as extractor:
            with pytest.raises(FileNotFoundError):
                extractor.extract_from_video("nonexistent_video.mp4")

    def test_extract_from_video_with_invalid_frame_skip(self):
        """Test extract_from_video fails with invalid frame_skip."""
        with PoseExtractor() as extractor:
            with pytest.raises(ValueError, match="frame_skip must be at least 1"):
                extractor.extract_from_video("video.mp4", frame_skip=0)

    def test_get_landmark_names(self):
        """Test get_landmark_names returns correct number of landmarks."""
        with PoseExtractor() as extractor:
            landmark_names = extractor.get_landmark_names()

            assert len(landmark_names) == 33
            assert isinstance(landmark_names, list)
            assert all(isinstance(name, str) for name in landmark_names)

    def test_landmark_visibility_range(self):
        """Test that visibility values are in valid range."""
        with PoseExtractor() as extractor:
            # Create random test frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            result = extractor.extract_pose(test_frame)

            if result is not None:
                # Check visibility is in [0, 1]
                visibility_2d = result['landmarks_2d'][:, 2]
                assert np.all((visibility_2d >= 0) & (visibility_2d <= 1))

                visibility_3d = result['landmarks_3d'][:, 3]
                assert np.all((visibility_3d >= 0) & (visibility_3d <= 1))


class TestPoseExtractorIntegration:
    """Integration tests for PoseExtractor."""

    def test_process_multiple_frames(self):
        """Test processing multiple frames in sequence."""
        with PoseExtractor() as extractor:
            results = []

            # Process 5 test frames
            for i in range(5):
                # Create slightly different frames
                test_frame = np.ones((480, 640, 3), dtype=np.uint8) * (50 + i * 10)
                result = extractor.extract_pose(test_frame)
                results.append(result)

            assert len(results) == 5

    def test_different_model_complexities(self):
        """Test that different model complexities work."""
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        for complexity in [0, 1, 2]:
            with PoseExtractor(model_complexity=complexity) as extractor:
                result = extractor.extract_pose(test_frame)

                # Should not raise exception
                assert result is None or isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
