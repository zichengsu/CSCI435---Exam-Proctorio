"""
motion_detector.py -- motion / camera-block detection (OpenCV MOG2)
===================================================================

Detects suspicious movement and a blocked camera during an exam. The original
prototype processed a whole video at once; this version wraps the same logic
(MOG2 background subtraction + frame differencing + the same thresholds) in a
small stateful class so the app can process ONE frame at a time and keep state
between frames.
"""

import time
import numpy as np
import cv2


class MotionDetector:
    """Stateful per-frame motion / violation engine."""

    def __init__(self, fps=25.0):
        # fps is used to convert frame numbers into seconds, exactly like the
        # notebook did with `current_time = frame_number / fps_input`.
        self.fps = fps if fps and fps > 0 else 25.0

        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.previous_gray = None
        self.frame_number = 0
        self.excessive_start = None
        self.camera_blocked_start = None
        self.violation_count = 0

    def process_frame(self, frame):
        """
        Run motion analysis on a single BGR frame.

        Returns a dict with the logged fields, plus a
        `largest_box` that the caller can draw.
        """
        t0 = time.time()
        self.frame_number += 1
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)

        fg_mask = self.background_subtractor.apply(blur)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        # ---- frame differencing ----
        frame_diff_score = 0.0
        if self.previous_gray is not None:
            diff = cv2.absdiff(self.previous_gray, blur)
            _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            frame_diff_score = np.sum(diff_thresh > 0) / (width * height)
        self.previous_gray = blur.copy()

        # ---- contours / motion area ----
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        total_motion_area = 0
        largest_box = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:
                x, y, w, h = cv2.boundingRect(contour)
                total_motion_area += area
                if largest_box is None or area > largest_box[4]:
                    largest_box = (x, y, w, h, area)

        motion_ratio = total_motion_area / (width * height)
        brightness = float(np.mean(gray))
        current_time = self.frame_number / self.fps

        violation = "Normal"
        risk_score = 0
        risk_level = "Low"

        # ---- excessive movement ----
        if motion_ratio > 0.08 or frame_diff_score > 0.07:
            if self.excessive_start is None:
                self.excessive_start = current_time
            if current_time - self.excessive_start >= 0.5:
                violation = "Excessive Movement Detected"
                if motion_ratio > 0.20:
                    risk_score, risk_level = 40, "High"
                elif motion_ratio > 0.12:
                    risk_score, risk_level = 25, "Medium"
                else:
                    risk_score, risk_level = 15, "Low"
        else:
            self.excessive_start = None

        # ---- camera blocked ----
        if brightness < 35:
            if self.camera_blocked_start is None:
                self.camera_blocked_start = current_time
            if current_time - self.camera_blocked_start >= 1.0:
                violation = "Camera Blocked"
                risk_score, risk_level = 50, "High"
        else:
            self.camera_blocked_start = None

        if violation != "Normal":
            self.violation_count += 1

        latency_ms = (time.time() - t0) * 1000.0

        return {
            "violation": violation,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "motion_ratio": round(motion_ratio, 4),
            "frame_diff_score": round(frame_diff_score, 4),
            "brightness": round(brightness, 2),
            "largest_box": largest_box,
            "latency_ms": round(latency_ms, 2),
        }

    @staticmethod
    def draw(frame, result):
        """Draw the motion box on the frame (in place) and return it."""
        box = result.get("largest_box")
        if box is not None:
            x, y, w, h, _ = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
