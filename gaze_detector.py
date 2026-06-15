
"gaze_detector.py — Core Detection Engine"

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GazeDetector")


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────
@dataclass
class GazeResult:
    """Container for a single-frame gaze analysis result."""
    gaze_direction: str        
    is_suspicious: bool         
    horizontal_ratio: float     
    vertical_ratio: float       
    left_eye_landmarks: Optional[np.ndarray] = None
    right_eye_landmarks: Optional[np.ndarray] = None
    face_detected: bool = True
    confidence: float = 1.0


@dataclass
class GazeConfig:
    """Configuration parameters for gaze detection."""
    # MediaPipe parameters
    max_num_faces: int = 1
    refine_landmarks: bool = True       
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Gaze threshold parameters
    horizontal_center_min: float = 0.35  # Below this → looking left
    horizontal_center_max: float = 0.65  # Above this → looking right
    vertical_center_min: float = 0.30   # Below this → looking up
    vertical_center_max: float = 0.70    # Above this → looking down

    # Suspicion parameters
    suspicion_threshold_sec: float = 2.0    # Seconds of sustained off center gaze before flagging
    suspicion_cooldown_sec: float = 1.0     # Cooldown after a suspicion event resets
    history_length: int = 15                 # Number of frames to smooth over

    # Display
    draw_eye_mesh: bool = True
    draw_gaze_arrow: bool = True

# ──────────────────────────────────────────────
# MediaPipe Landmark Indices
# ──────────────────────────────────────────────
# Left eye contour (from Face Mesh topology)
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                    173, 157, 158, 159, 160, 161, 246]

# Right eye contour
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                     466, 388, 387, 386, 385, 384, 398]

# Iris center landmarks (available when refine_landmarks=True)
LEFT_IRIS_CENTER = 468
LEFT_IRIS_CONTOUR = [469, 470, 471, 472]
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_CONTOUR = [474, 475, 476, 477]

# Eye corner indices for simpler ratio calculation
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


# ──────────────────────────────────────────────
# Main Detector Class
# ──────────────────────────────────────────────
class GazeDetector:
    """
    Real-time gaze direction detector using MediaPipe Face Mesh.

    Pipeline:
        Frame → Face Mesh → Extract Eye & Iris Landmarks →
        Compute Gaze Ratios → Smooth over History → Classify Direction →
        Track Duration → Flag Suspicious Gaze
    """

    def __init__(self, config: GazeConfig = None):
        self.config = config or GazeConfig()

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.config.max_num_faces,
            refine_landmarks=self.config.refine_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # State tracking for suspicion logic
        self._gaze_history = deque(maxlen=self.config.history_length)
        self._off_center_start: Optional[float] = None
        self._last_suspicion_time: float = 0
        self._suspicion_active: bool = False
        self._total_suspicious_events: int = 0
        self._total_suspicious_duration: float = 0.0
        self._session_start: float = time.time()

    # ──────────────────────────────────────────
    # Core Detection
    # ──────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> GazeResult:
       
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        img_h, img_w = frame.shape[:2]

        if not results.multi_face_landmarks:
            self._gaze_history.clear()
            self._off_center_start = None
            return GazeResult(
                gaze_direction="No Face",
                is_suspicious=False,
                horizontal_ratio=0.5,
                vertical_ratio=0.5,
                face_detected=False,
                confidence=0.0,
            )

        face_landmarks = results.multi_face_landmarks[0]

        # ── Compute Gaze Ratios ──
        h_ratio, v_ratio = self._compute_gaze_ratios(face_landmarks, img_w, img_h)

        # ── Smooth with History ──
        self._gaze_history.append((h_ratio, v_ratio))
        smooth_h, smooth_v = self._smooth_ratios()

        # ── Classify Direction ──
        direction = self._classify_direction(smooth_h, smooth_v)

        # ── Suspicion Logic ──
        is_suspicious = self._update_suspicion(direction)

        # ── Extract eye landmark arrays for drawing ──
        left_eye_pts = self._get_landmark_points(face_landmarks, LEFT_EYE_CONTOUR, img_w, img_h)
        right_eye_pts = self._get_landmark_points(face_landmarks, RIGHT_EYE_CONTOUR, img_w, img_h)

        return GazeResult(
            gaze_direction=direction,
            is_suspicious=is_suspicious,
            horizontal_ratio=smooth_h,
            vertical_ratio=smooth_v,
            left_eye_landmarks=left_eye_pts,
            right_eye_landmarks=right_eye_pts,
            face_detected=True,
            confidence=1.0,
        )

    # ──────────────────────────────────────────
    # Ratio Computation
    # ──────────────────────────────────────────
    def _compute_gaze_ratios(
        self, face_landmarks, img_w: int, img_h: int
    ) -> Tuple[float, float]:
        lm = face_landmarks.landmark

        # ── LEFT EYE ──
        left_iris_x = lm[LEFT_IRIS_CENTER].x * img_w
        left_iris_y = lm[LEFT_IRIS_CENTER].y * img_h
        left_inner_x = lm[LEFT_EYE_INNER].x * img_w
        left_outer_x = lm[LEFT_EYE_OUTER].x * img_w
        left_top_y = lm[LEFT_EYE_TOP].y * img_h
        left_bottom_y = lm[LEFT_EYE_BOTTOM].y * img_h

        left_eye_width = left_inner_x - left_outer_x
        left_eye_height = left_bottom_y - left_top_y

        if abs(left_eye_width) < 1e-6 or abs(left_eye_height) < 1e-6:
            return 0.5, 0.5  # Fallback

        left_h_ratio = (left_iris_x - left_outer_x) / left_eye_width
        left_v_ratio = (left_iris_y - left_top_y) / left_eye_height

        # ── RIGHT EYE ──
        right_iris_x = lm[RIGHT_IRIS_CENTER].x * img_w
        right_iris_y = lm[RIGHT_IRIS_CENTER].y * img_h
        right_inner_x = lm[RIGHT_EYE_INNER].x * img_w
        right_outer_x = lm[RIGHT_EYE_OUTER].x * img_w
        right_top_y = lm[RIGHT_EYE_TOP].y * img_h
        right_bottom_y = lm[RIGHT_EYE_BOTTOM].y * img_h

        right_eye_width = right_outer_x - right_inner_x
        right_eye_height = right_bottom_y - right_top_y

        if abs(right_eye_width) < 1e-6 or abs(right_eye_height) < 1e-6:
            return 0.5, 0.5

        right_h_ratio = (right_iris_x - right_inner_x) / right_eye_width
        right_v_ratio = (right_iris_y - right_top_y) / right_eye_height

        # Average both eyes
        avg_h = (left_h_ratio + right_h_ratio) / 2.0
        avg_v = (left_v_ratio + right_v_ratio) / 2.0

        # Clamp to [0, 1]
        avg_h = np.clip(avg_h, 0.0, 1.0)
        avg_v = np.clip(avg_v, 0.0, 1.0)

        return float(avg_h), float(avg_v)

    def _smooth_ratios(self) -> Tuple[float, float]:
        """Apply moving-average smoothing over gaze history."""
        if not self._gaze_history:
            return 0.5, 0.5
        h_vals = [g[0] for g in self._gaze_history]
        v_vals = [g[1] for g in self._gaze_history]
        return float(np.mean(h_vals)), float(np.mean(v_vals))

    # ──────────────────────────────────────────
    # Direction Classification
    # ──────────────────────────────────────────
    def _classify_direction(self, h: float, v: float) -> str:
        """Map smooth gaze ratios to a human-readable direction label."""
        cfg = self.config

        # Check vertical first (looking up/down is more distinctive)
        if v < cfg.vertical_center_min:
            return "Looking Up"
        if v > cfg.vertical_center_max:
            return "Looking Down"

        # Check horizontal
        if h < cfg.horizontal_center_min:
            return "Looking Left"
        if h > cfg.horizontal_center_max:
            return "Looking Right"

        return "Looking Forward"

    # ──────────────────────────────────────────
    # Suspicion Tracking
    # ──────────────────────────────────────────
    def _update_suspicion(self, direction: str) -> bool:
        """
        Track how long the gaze has been off-center.
        Returns True if the sustained off-center duration exceeds the threshold.
        """
        now = time.time()

        if direction == "Looking Forward":
            # Reset off-center timer
            self._off_center_start = None
            self._suspicion_active = False
            return False

        # Gaze is off-center
        if self._off_center_start is None:
            self._off_center_start = now

        elapsed = now - self._off_center_start

        if elapsed >= self.config.suspicion_threshold_sec:
            if not self._suspicion_active:
                # New suspicious event just started
                self._total_suspicious_events += 1
                logger.warning(
                    f"Suspicious gaze detected: '{direction}' for {elapsed:.1f}s "
                    f"(event #{self._total_suspicious_events})"
                )
            self._suspicion_active = True
            self._total_suspicious_duration = (
                self._total_suspicious_duration + 1 / 30.0  
            )
            return True

        return False

    # ──────────────────────────────────────────
    # Helper: Extract Landmark Pixel Coordinates
    # ──────────────────────────────────────────
    def _get_landmark_points(
        self, face_landmarks, indices: List[int], img_w: int, img_h: int
    ) -> np.ndarray:
        """Convert normalised landmarks to pixel coordinates."""
        pts = []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            pts.append([int(lm.x * img_w), int(lm.y * img_h)])
        return np.array(pts, dtype=np.int32)

    # ──────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────
    def draw(
        self, frame: np.ndarray, result: GazeResult
    ) -> np.ndarray:
        """
        Draw gaze analysis overlays on the frame.

        Overlays:
            - Eye contour and iris circles
            - Gaze direction arrow from each eye
            - Text label with direction and suspicion status
            - Gaze ratio bar (debug/visual feedback)
        """
        annotated = frame.copy()

        if not result.face_detected:
            cv2.putText(
                annotated, "No Face Detected", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
            )
            return annotated

        # ── Draw Eye Contours and Irises ──
        if self.config.draw_eye_mesh:
            annotated = self._draw_eyes(annotated, result)

        # ── Draw Gaze Direction Arrow ──
        if self.config.draw_gaze_arrow:
            annotated = self._draw_gaze_arrow(annotated, result)

        # ── Draw Direction Label ──
        annotated = self._draw_direction_label(annotated, result)

        # ── Draw Gaze Ratio Bars ──
        annotated = self._draw_ratio_bars(annotated, result)

        return annotated

    def _draw_eyes(self, frame: np.ndarray, result: GazeResult) -> np.ndarray:
        """Draw eye contour polygons and iris circles."""
        # Left eye contour
        if result.left_eye_landmarks is not None and len(result.left_eye_landmarks) > 0:
            cv2.polylines(
                frame, [result.left_eye_landmarks], True, (0, 255, 0), 1, cv2.LINE_AA
            )

        # Right eye contour
        if result.right_eye_landmarks is not None and len(result.right_eye_landmarks) > 0:
            cv2.polylines(
                frame, [result.right_eye_landmarks], True, (0, 255, 0), 1, cv2.LINE_AA
            )

        # Draw iris circles using MediaPipe's drawing utility
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mp = self.face_mesh.process(rgb_frame)
        if results_mp.multi_face_landmarks:
            face_lm = results_mp.multi_face_landmarks[0]
            img_h, img_w = frame.shape[:2]

            # Left iris
            iris_left = face_lm.landmark[LEFT_IRIS_CENTER]
            cx, cy = int(iris_left.x * img_w), int(iris_left.y * img_h)
            # Compute iris radius from contour points
            iris_pts = self._get_landmark_points(face_lm, LEFT_IRIS_CONTOUR, img_w, img_h)
            if len(iris_pts) > 0:
                radius = int(np.linalg.norm(iris_pts[0] - iris_pts[2]) / 2)
                cv2.circle(frame, (cx, cy), radius, (255, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2, (255, 0, 255), -1)

            # Right iris
            iris_right = face_lm.landmark[RIGHT_IRIS_CENTER]
            cx, cy = int(iris_right.x * img_w), int(iris_right.y * img_h)
            iris_pts = self._get_landmark_points(face_lm, RIGHT_IRIS_CONTOUR, img_w, img_h)
            if len(iris_pts) > 0:
                radius = int(np.linalg.norm(iris_pts[0] - iris_pts[2]) / 2)
                cv2.circle(frame, (cx, cy), radius, (255, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2, (255, 0, 255), -1)

        return frame

    def _draw_gaze_arrow(self, frame: np.ndarray, result: GazeResult) -> np.ndarray:
        """Draw an arrow indicating gaze direction from each eye center."""
        if result.left_eye_landmarks is None or result.right_eye_landmarks is None:
            return frame

        img_h, img_w = frame.shape[:2]
        arrow_length = 50

        # Compute eye centers
        left_center = result.left_eye_landmarks.mean(axis=0).astype(int)
        right_center = result.right_eye_landmarks.mean(axis=0).astype(int)

        # Compute arrow direction from ratios
        # h_ratio: 0=left, 1=right → arrow points left when ratio is low
        # v_ratio: 0=top, 1=bottom → arrow points up when ratio is low
        dx = (result.horizontal_ratio - 0.5) * 2 * arrow_length
        dy = (result.vertical_ratio - 0.5) * 2 * arrow_length

        color = (0, 0, 255) if result.is_suspicious else (0, 255, 0)

        for center in [left_center, right_center]:
            end_pt = (int(center[0] + dx), int(center[1] + dy))
            cv2.arrowedLine(
                frame, tuple(center), end_pt, color, 2, tipLength=0.3
            )

        return frame

    def _draw_direction_label(
        self, frame: np.ndarray, result: GazeResult
    ) -> np.ndarray:
        """Draw text overlay showing gaze direction and suspicion status."""
        img_h, img_w = frame.shape[:2]

        # Direction label
        direction_color = (0, 255, 0) if result.gaze_direction == "Looking Forward" else (0, 165, 255)
        cv2.putText(
            frame, f"Gaze: {result.gaze_direction}", (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, direction_color, 2, cv2.LINE_AA,
        )

        # Suspicion warning
        if result.is_suspicious:
            # Flashing red box
            if int(time.time() * 4) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (img_w, 80), (0, 0, 200), -1)
                cv2.putText(
                    frame, "⚠ SUSPICIOUS GAZE DETECTED ⚠", (30, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
                )

        # Ratio values (debug)
        cv2.putText(
            frame,
            f"H: {result.horizontal_ratio:.2f}  V: {result.vertical_ratio:.2f}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        return frame

    def _draw_ratio_bars(
        self, frame: np.ndarray, result: GazeResult
    ) -> np.ndarray:
        """Draw horizontal and vertical gaze ratio bars for visual feedback."""
        bar_x, bar_y = frame.shape[1] - 160, 20
        bar_w, bar_h = 130, 15

        # ── Horizontal bar ──
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        # Center zone (green)
        cx1 = bar_x + int(bar_w * self.config.horizontal_center_min)
        cx2 = bar_x + int(bar_w * self.config.horizontal_center_max)
        cv2.rectangle(frame, (cx1, bar_y), (cx2, bar_y + bar_h), (0, 100, 0), -1)
        # Current position marker
        marker_x = bar_x + int(bar_w * result.horizontal_ratio)
        cv2.line(frame, (marker_x, bar_y - 3), (marker_x, bar_y + bar_h + 3), (0, 255, 255), 2)
        cv2.putText(frame, "H-Gaze", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # ── Vertical bar ──
        vy = bar_y + bar_h + 15
        cv2.rectangle(frame, (bar_x, vy), (bar_x + bar_w, vy + bar_h), (50, 50, 50), -1)
        cy1 = bar_x + int(bar_w * self.config.vertical_center_min)
        cy2 = bar_x + int(bar_w * self.config.vertical_center_max)
        cv2.rectangle(frame, (cy1, vy), (cy2, vy + bar_h), (0, 100, 0), -1)
        marker_y_pos = bar_x + int(bar_w * result.vertical_ratio)
        cv2.line(frame, (marker_y_pos, vy - 3), (marker_y_pos, vy + bar_h + 3), (0, 255, 255), 2)
        cv2.putText(frame, "V-Gaze", (bar_x, vy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    # ──────────────────────────────────────────
    # Statistics & Reporting
    # ──────────────────────────────────────────
    def get_stats(self) -> dict:
        """Return session statistics for the report / dashboard."""
        session_duration = time.time() - self._session_start
        return {
            "total_suspicious_events": self._total_suspicious_events,
            "total_suspicious_duration_sec": round(self._total_suspicious_duration, 2),
            "session_duration_sec": round(session_duration, 2),
            "suspicious_percentage": round(
                (self._total_suspicious_duration / max(session_duration, 1)) * 100, 2
            ),
        }

    def reset_stats(self):
        """Reset all session statistics."""
        self._total_suspicious_events = 0
        self._total_suspicious_duration = 0.0
        self._session_start = time.time()
        self._off_center_start = None
        self._suspicion_active = False
        self._gaze_history.clear()

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()