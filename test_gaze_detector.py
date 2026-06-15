
"test_gaze_detector.py — Unit Tests"

import pytest
import numpy as np
import cv2
from gaze_detector import GazeDetector, GazeConfig, GazeResult


class TestGazeConfig:
    """Test configuration defaults and custom values."""

    def test_default_config(self):
        config = GazeConfig()
        assert config.refine_landmarks is True
        assert config.suspicion_threshold_sec == 2.0
        assert config.horizontal_center_min == 0.35

    def test_custom_config(self):
        config = GazeConfig(suspicion_threshold_sec=5.0, max_num_faces=2)
        assert config.suspicion_threshold_sec == 5.0
        assert config.max_num_faces == 2


class TestGazeDetector:
    """Test the GazeDetector class."""

    @pytest.fixture
    def detector(self):
        config = GazeConfig(
            suspicion_threshold_sec=1.0,
            history_length=5,
        )
        d = GazeDetector(config)
        yield d
        d.close()

    @pytest.fixture
    def blank_frame(self):
        """Create a blank frame (no face)."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_no_face_detected(self, detector, blank_frame):
        """A blank frame should return face_detected=False."""
        result = detector.detect(blank_frame)
        assert result.face_detected is False
        assert result.gaze_direction == "No Face"

    def test_result_type(self, detector, blank_frame):
        """Result should be a GazeResult instance."""
        result = detector.detect(blank_frame)
        assert isinstance(result, GazeResult)

    def test_ratio_ranges(self, detector, blank_frame):
        """Even with no face, default ratios should be in [0, 1]."""
        result = detector.detect(blank_frame)
        assert 0.0 <= result.horizontal_ratio <= 1.0
        assert 0.0 <= result.vertical_ratio <= 1.0

    def test_classify_direction(self, detector):
        """Test the direction classification logic directly."""
        cfg = detector.config
        assert detector._classify_direction(0.5, 0.5) == "Looking Forward"
        assert detector._classify_direction(0.1, 0.5) == "Looking Left"
        assert detector._classify_direction(0.9, 0.5) == "Looking Right"
        assert detector._classify_direction(0.5, 0.1) == "Looking Up"
        assert detector._classify_direction(0.5, 0.9) == "Looking Down"

    def test_stats_initial(self, detector):
        """Stats should start at zero."""
        stats = detector.get_stats()
        assert stats["total_suspicious_events"] == 0
        assert stats["total_suspicious_duration_sec"] == 0.0

    def test_stats_reset(self, detector):
        """Reset should clear all stats."""
        detector._total_suspicious_events = 5
        detector.reset_stats()
        stats = detector.get_stats()
        assert stats["total_suspicious_events"] == 0

    def test_draw_returns_frame(self, detector, blank_frame):
        """draw() should return a numpy array of same shape."""
        result = detector.detect(blank_frame)
        annotated = detector.draw(blank_frame, result)
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == blank_frame.shape


class TestGazeRatios:
    """Test gaze ratio computation with synthetic landmark data."""

    @pytest.fixture
    def detector(self):
        d = GazeDetector(GazeConfig())
        yield d
        d.close()

    def test_smooth_ratios_empty(self, detector):
        """Empty history should return center (0.5, 0.5)."""
        h, v = detector._smooth_ratios()
        assert h == 0.5
        assert v == 0.5

    def test_smooth_ratios_values(self, detector):
        """Averaging should work correctly."""
        detector._gaze_history.extend([
            (0.4, 0.6), (0.5, 0.5), (0.6, 0.4)
        ])
        h, v = detector._smooth_ratios()
        assert abs(h - 0.5) < 0.01
        assert abs(v - 0.5) < 0.01