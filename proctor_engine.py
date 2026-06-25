"""
proctor_engine.py -- integration layer
======================================================================

This is where the four vision modules become ONE system. For each frame it:

    1. Object detection  -> phones / persons        (YOLOv8)
    2. Gaze tracking     -> gaze direction          (MediaPipe)
    3. Face recognition  -> face identity           (face_recognition)
    4. Motion analysis   -> motion / camera-blocked (OpenCV MOG2)
    5. Combines all signals into a single risk score + overall status,
       combining them with the violation-engine weighting:
           face_alert  -> +30
           phone_alert -> +50
           gaze_alert  -> +20
       on top of the motion/camera risk score.

It draws every module's overlay on the frame, then a summary banner.

The heavy CV libraries (ultralytics / mediapipe) are imported lazily inside
__init__ so this file can be imported for inspection without them installed.
"""

import time
import cv2

from motion_detector import MotionDetector
from face_recognizer import FaceRecognizer


# Object-detection module: detections are classified by their class NAME, so it
# works for any model regardless of class numbering (the pretrained COCO model
# uses 0=person / 67=cell phone, while the custom model uses 0=mobilephone).


class _ObjectDetector:
    """
    Per-frame wrapper around the YOLOv8 detection pipeline.
    Mirrors the detection logic in object_detection_proctor.py but exposes a
    single process_frame() call instead of a full generator loop.
    """

    def __init__(self, model_path="yolov8s.pt", conf=0.5):
        from ultralytics import YOLO  # lazy import
        self.model = YOLO(model_path)
        self.conf = conf
        # Map this model's class ids to "person" / "phone" by NAME, so both the
        # pretrained and the custom fine-tuned model are handled correctly.
        self.person_ids, self.phone_ids = [], []
        for cid, name in self.model.names.items():
            low = str(name).lower()
            if "person" in low:
                self.person_ids.append(int(cid))
            if "phone" in low:  # matches "cell phone" and "mobilephone"
                self.phone_ids.append(int(cid))
        self.keep_ids = self.person_ids + self.phone_ids

    def process_frame(self, frame):
        kwargs = dict(conf=self.conf, verbose=False)
        if self.keep_ids:
            kwargs["classes"] = self.keep_ids
        results = self.model(frame, **kwargs)[0]
        persons = phones = 0
        boxes = []
        for box in results.boxes:
            cid = int(box.cls[0])
            c = float(box.conf[0])
            if c < self.conf:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cid in self.phone_ids:
                phones += 1
                boxes.append((x1, y1, x2, y2, "PHONE", (0, 0, 255)))
            elif cid in self.person_ids:
                persons += 1
                boxes.append((x1, y1, x2, y2, "PERSON", (0, 255, 0)))
        return {"persons": persons, "phones": phones, "boxes": boxes}

    @staticmethod
    def draw(frame, result):
        for x1, y1, x2, y2, label, color in result["boxes"]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame


class ProctorEngine:
    def __init__(self, fps=25.0, object_model="yolov8s.pt", conf=0.5,
                 enable_object=True, enable_gaze=True, enable_face=True,
                 enable_motion=True, enable_enhance=True):
        self.enable_object = enable_object
        self.enable_gaze = enable_gaze
        self.enable_face = enable_face
        self.enable_motion = enable_motion
        self.enable_enhance = enable_enhance

        # Image-enhancement task: CLAHE (Contrast Limited Adaptive Histogram
        # Equalisation) is applied to dark frames to improve detection under
        # poor lighting -- a distinct vision capability that also makes the
        # system robust to varying lighting conditions.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.object_detector = (
            _ObjectDetector(object_model, conf) if enable_object else None
        )

        self.gaze_detector = None
        if enable_gaze:
            from gaze_detector import GazeDetector, GazeConfig  # lazy import
            # Tuned to be responsive for a live demo: flags after ~0.5s of
            # off-centre gaze (instead of 2s) and uses tighter centre bands so a
            # deliberate glance away is detected quickly.
            self.gaze_detector = GazeDetector(GazeConfig(
                suspicion_threshold_sec=0.5,
                horizontal_center_min=0.42,
                horizontal_center_max=0.58,
                vertical_center_min=0.40,
                vertical_center_max=0.60,
                history_length=6,
            ))

        self.face_recognizer = FaceRecognizer() if enable_face else None
        self.motion_detector = MotionDetector(fps=fps) if enable_motion else None

        self._fps_t = time.time()
        self._fps = 0.0

    def process_frame(self, frame):
        """Run all enabled modules on a frame, draw overlays, return a dict."""
        t0 = time.time()
        original = frame

        # ---- image enhancement (low-light) ----
        # Auto-applies CLAHE only when the frame is dark, so well-lit frames are
        # untouched and dim ones are brightened before detection.
        enhanced = False
        work = frame
        if self.enable_enhance:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if float(gray.mean()) < 110:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l = self._clahe.apply(l)
                work = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
                enhanced = True

        annotated = work.copy()
        violations = []

        persons = phones = 0
        gaze_dir, gaze_alert, face_found = "N/A", False, False
        identity, face_alert = "N/A", False
        motion_violation, motion_ratio, brightness = "Normal", 0.0, 0.0
        motion_risk, risk_level = 0, "Low"

        # ---- object detection ----
        if self.object_detector:
            r = self.object_detector.process_frame(work)
            persons, phones = r["persons"], r["phones"]
            self.object_detector.draw(annotated, r)
            if phones > 0:
                violations.append(f"Phone Detected ({phones})")
            if persons > 1:
                violations.append(f"Multiple Persons ({persons})")

        # ---- gaze tracking ----
        if self.gaze_detector:
            g = self.gaze_detector.detect(work)
            gaze_dir = g.gaze_direction
            gaze_alert = g.is_suspicious
            face_found = g.face_detected
            annotated = self.gaze_detector.draw(annotated, g)
            if gaze_alert:
                violations.append(f"Suspicious Gaze ({gaze_dir})")

        # ---- face recognition ----
        if self.face_recognizer:
            f = self.face_recognizer.process_frame(work)
            identity = f["name"]
            face_alert = f["face_alert"]
            self.face_recognizer.draw(annotated, f)
            if face_alert:
                violations.append(f"Unregistered Face ({identity})")

        # ---- motion analysis ----
        if self.motion_detector:
            m = self.motion_detector.process_frame(original)
            motion_violation = m["violation"]
            motion_ratio = m["motion_ratio"]
            brightness = m["brightness"]
            motion_risk = m["risk_score"]
            risk_level = m["risk_level"]
            self.motion_detector.draw(annotated, m)
            if motion_violation != "Normal":
                violations.append(motion_violation)

        # ---- combined scoring (violation-engine weights) ----
        final_score = motion_risk
        phone_alert = phones > 0
        if face_alert:
            final_score += 30
        if phone_alert:
            final_score += 50
        if gaze_alert:
            final_score += 20

        if final_score >= 50:
            overall_status = "Violation"
        elif final_score >= 20 or violations:
            overall_status = "Warning"
        else:
            overall_status = "Normal"

        # Re-derive risk_level from the COMBINED score so it always agrees with
        # overall_status. (Previously risk_level only reflected motion intensity,
        # so a stationary phone could read "Low" next to a "Violation".)
        if final_score >= 50:
            risk_level = "High"
        elif final_score >= 20:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # ---- fps ----
        now = time.time()
        self._fps = 1.0 / max(now - self._fps_t, 1e-6)
        self._fps_t = now
        latency_ms = (now - t0) * 1000.0

        self._draw_banner(annotated, overall_status, final_score, risk_level)

        return {
            "annotated": annotated,
            "persons": persons, "phones": phones,
            "gaze_direction": gaze_dir, "gaze_alert": gaze_alert,
            "face_found": face_found, "identity": identity,
            "face_alert": face_alert,
            "motion_violation": motion_violation,
            "motion_ratio": motion_ratio, "brightness": brightness,
            "risk_score": final_score, "risk_level": risk_level,
            "overall_status": overall_status,
            "violations": violations,
            "enhanced": enhanced,
            "fps": round(self._fps, 2), "latency_ms": round(latency_ms, 2),
        }

    @staticmethod
    def _draw_banner(frame, status, score, level):
        w = frame.shape[1]
        color = {"Violation": (0, 0, 200), "Warning": (0, 140, 220),
                 "Normal": (0, 120, 0)}[status]
        cv2.rectangle(frame, (0, 0), (w, 46), color, -1)
        cv2.putText(frame, f"{status}   Risk: {score} ({level})", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
