"""
Unit tests for the Multimodal IoT Scene Narrator modules.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.detector import YOLODetector
from core.narrator import SceneNarrator
from core.tts import TTSEngine


class TestYOLODetector:

    def test_format_detections_empty(self):
        """Empty detection list returns correct message."""
        detector = YOLODetector()
        result = detector.format_detections([])
        assert result == "No objects detected by YOLO."

    def test_format_detections_single(self):
        """Single detection formats correctly."""
        detector = YOLODetector()
        detections = [{"class": "person", "confidence": 0.85, "bbox": [0, 0, 100, 100]}]
        result = detector.format_detections(detections)
        assert "person" in result
        assert "0.85" in result

    def test_format_detections_groups_same_class(self):
        """Multiple detections of same class are grouped."""
        detector = YOLODetector()
        detections = [
            {"class": "person", "confidence": 0.85, "bbox": [0, 0, 100, 100]},
            {"class": "person", "confidence": 0.72, "bbox": [200, 0, 300, 100]},
        ]
        result = detector.format_detections(detections)
        assert "2 person(s)" in result

    def test_draw_detections_returns_frame(self):
        """draw_detections returns a frame of the same shape."""
        detector = YOLODetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [{"class": "person", "confidence": 0.85, "bbox": [10, 10, 100, 100]}]
        result = detector.draw_detections(frame, detections)
        assert result.shape == frame.shape


class TestSceneNarrator:

    def test_mock_narrate_no_detections(self):
        """Mock narration with no detections returns clear message."""
        narrator = SceneNarrator()
        result = narrator._mock_narrate("No objects detected by YOLO.")
        assert "clear" in result.lower() or "no" in result.lower()

    def test_mock_narrate_with_detections(self):
        """Mock narration with detections includes detection info."""
        narrator = SceneNarrator()
        result = narrator._mock_narrate("YOLO detections: 1 person(s) [conf: 0.85]")
        assert "person" in result or "detected" in result.lower()

    def test_encode_frame_returns_string(self):
        """Frame encoding returns a non-empty base64 string."""
        narrator = SceneNarrator()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = narrator.encode_frame(frame)
        assert isinstance(result, str)
        assert len(result) > 0


class TestTTSEngine:

    def test_toggle_mute(self):
        """Mute toggles correctly."""
        tts = TTSEngine()
        assert tts.muted is False
        tts.toggle_mute()
        assert tts.muted is True
        tts.toggle_mute()
        assert tts.muted is False

    def test_speak_does_not_crash_when_muted(self):
        """Speak silently does nothing when muted."""
        tts = TTSEngine()
        tts.muted = True
        tts.speak("This should not crash")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])