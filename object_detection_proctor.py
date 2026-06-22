"""
AI Exam Proctoring System -- Member 2 Module
============================================

Phone & Person detection using YOLOv8.

This module is the Member 2 contribution to the CSCI435 group project.
It exposes a single, reusable detection pipeline that supports three input
modalities (webcam, image file, video file) and produces both a visual
overlay (bounding boxes + status banner) and a structured per-frame record
that the Streamlit frontend (Member 5) can consume.

Responsibilities:
    * Phone (COCO class 67) detection
    * Person (COCO class 0) detection + multi-person violation flag
    * Frame-skipping for FPS optimisation
    * Temporal stability buffer (suppresses flicker between detections)
    * Violation logging to a timestamped text file
    * FPS + per-frame latency measurement for the report

Author: Member 2 -- CSCI435 Group Project, Spring 2026
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable, Optional, Union

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# COCO class IDs we care about for exam proctoring
PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67

# BGR colours for OpenCV drawing
COLOR_PERSON_OK = (0, 255, 0)     # green
COLOR_PERSON_WARN = (0, 165, 255)  # orange
COLOR_PHONE = (0, 0, 255)          # red
COLOR_BANNER_BAD = (0, 0, 200)     # dark red
COLOR_BANNER_OK = (0, 120, 0)      # dark green
COLOR_TEXT = (255, 255, 255)       # white

DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_LOG = "violations.txt"
DEFAULT_METRICS = "metrics.csv"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Structured per-frame output, consumable by the Streamlit frontend."""

    frame_id: int
    persons: int
    phones: int
    violations: list[str] = field(default_factory=list)
    fps: float = 0.0
    latency_ms: float = 0.0
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Core detection pipeline
# ---------------------------------------------------------------------------

def run_detection(
    source: Union[int, str, Path] = 0,
    model_path: Union[str, Path] = DEFAULT_MODEL,
    conf_threshold: float = 0.5,
    skip_frames: int = 3,
    stability_frames: int = 3,
    log_file: Optional[Union[str, Path]] = DEFAULT_LOG,
    metrics_file: Optional[Union[str, Path]] = None,
    display: bool = True,
    save_annotated: Optional[Union[str, Path]] = None,
    max_frames: Optional[int] = None,
) -> Generator[FrameResult, None, None]:
    """
    Run phone + person detection on a webcam, video file, or single image.

    Parameters
    ----------
    source : int | str | Path
        ``0`` for the default webcam, a path string for an image or video file.
    model_path : str | Path
        Path to the YOLOv8 weights file. ``yolov8s.pt`` (pre-trained) by
        default. Point this at ``runs/detect/train/weights/best.pt`` after
        fine-tuning to use the custom model.
    conf_threshold : float
        Minimum confidence for a detection to be kept.
    skip_frames : int
        Run YOLO inference every N-th frame; intermediate frames reuse the
        last result. Trades accuracy for FPS. Set to 1 for image input.
    stability_frames : int
        Size of the rolling max-pool buffer used to suppress flicker.
    log_file : str | Path | None
        Where to append violation events. ``None`` disables file logging.
    metrics_file : str | Path | None
        Optional CSV file to dump per-frame metrics for the report.
    display : bool
        Show the OpenCV preview window. Disable for headless / Streamlit use.
    save_annotated : str | Path | None
        If set, write the annotated frame(s) to this path (image: single file,
        video: a directory of frames or an .mp4 if path ends in .mp4).
    max_frames : int | None
        Stop after this many frames. Useful for evaluation and benchmarks.

    Yields
    ------
    FrameResult
        Per-frame structured result. For image input, exactly one result is
        yielded and the function returns.
    """

    # ---------- 1. Load model ----------
    model = YOLO(str(model_path))

    # ---------- 2. Detect input modality ----------
    is_image = False
    if isinstance(source, (str, Path)) and str(source).lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ):
        is_image = True
        # For a single image we want one clean pass -- no skipping.
        skip_frames = 1
        cap = None
        frame = cv2.imread(str(source))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        frames: Iterable = [frame]
    else:
        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        frames = iter(lambda: cap.read(), (False, None))
        print(f"[proctor] Source opened: {source}")

    # ---------- 3. State ----------
    frame_count = 0
    last_results = None
    person_history: deque[int] = deque(maxlen=stability_frames)
    phone_history: deque[int] = deque(maxlen=stability_frames)
    last_logged_violation = ""
    fps_history: deque[float] = deque(maxlen=10)
    prev_time = time.time()

    metrics_writer = None
    metrics_file_handle = None
    if metrics_file:
        metrics_file_handle = open(metrics_file, "w", newline="", encoding="utf-8")
        metrics_writer = csv.writer(metrics_file_handle)
        metrics_writer.writerow(
            ["frame_id", "persons", "phones", "fps", "latency_ms",
             "violations", "timestamp"]
        )

    annotated_writer = None
    if save_annotated and str(save_annotated).endswith(".mp4"):
        # Will initialise after we know frame size on the first frame
        pass

    try:
        for item in frames:
            if is_image:
                frame = item
                ret = True
            else:
                ret, frame = item
            if not ret:
                break

            frame_count += 1
            t_start = time.time()

            person_count = 0
            phone_count = 0
            violations: list[str] = []

            # ---------- 4. Inference (with frame skipping) ----------
            if frame_count % skip_frames == 0:
                results = model(
                    frame,
                    conf=conf_threshold,
                    classes=[PERSON_CLASS_ID, PHONE_CLASS_ID],
                    verbose=False,
                )
                last_results = results[0]
                for box in last_results.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue
                    if class_id == PERSON_CLASS_ID:
                        person_count += 1
                    elif class_id == PHONE_CLASS_ID:
                        phone_count += 1

            # ---------- 5. Temporal stability ----------
            person_history.append(person_count)
            phone_history.append(phone_count)
            stable_persons = max(person_history) if person_history else 0
            stable_phones = max(phone_history) if phone_history else 0

            # ---------- 6. Draw bounding boxes ----------
            if last_results is not None:
                for box in last_results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    if conf < conf_threshold:
                        continue
                    if class_id == PERSON_CLASS_ID:
                        label = f"PERSON {conf:.2f}"
                        color = COLOR_PERSON_WARN if stable_persons > 1 else COLOR_PERSON_OK
                    elif class_id == PHONE_CLASS_ID:
                        label = f"PHONE {conf:.2f}"
                        color = COLOR_PHONE
                    else:
                        continue
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ---------- 7. Violation rules ----------
            if stable_phones > 0:
                violations.append(f"Phone Detected ({stable_phones})")
            if stable_persons > 1:
                violations.append(f"Multiple Persons ({stable_persons})")

            # ---------- 8. Log to file ----------
            status_text = " | ".join(violations) if violations else ""
            if violations and status_text != last_logged_violation:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                if log_file:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"{timestamp} - {status_text}\n")
                last_logged_violation = status_text
            elif not violations:
                last_logged_violation = ""

            # ---------- 9. FPS + latency ----------
            t_end = time.time()
            latency_ms = (t_end - t_start) * 1000.0
            fps = 1.0 / max(t_end - prev_time, 1e-6)
            prev_time = t_end
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history)

            result = FrameResult(
                frame_id=frame_count,
                persons=stable_persons,
                phones=stable_phones,
                violations=violations,
                fps=round(avg_fps, 2),
                latency_ms=round(latency_ms, 2),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )

            # ---------- 10. Metrics CSV ----------
            if metrics_writer:
                metrics_writer.writerow([
                    result.frame_id, result.persons, result.phones,
                    result.fps, result.latency_ms,
                    "; ".join(result.violations), result.timestamp,
                ])

            # ---------- 11. Display overlay ----------
            if display:
                _draw_overlay(frame, result, status_text)
                cv2.imshow("AI Exam Proctoring System", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # ---------- 12. Save annotated ----------
            if save_annotated:
                if is_image:
                    cv2.imwrite(str(save_annotated), frame)
                elif str(save_annotated).endswith(".mp4"):
                    if annotated_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        annotated_writer = cv2.VideoWriter(
                            str(save_annotated), fourcc, avg_fps or 25,
                            (frame.shape[1], frame.shape[0]),
                        )
                    annotated_writer.write(frame)
                else:
                    out_dir = Path(save_annotated)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_dir / f"frame_{frame_count:06d}.jpg"), frame)

            yield result

            if max_frames and frame_count >= max_frames:
                break

            if is_image:
                # Single image -- one yield, then we're done.
                break
    finally:
        if cap is not None:
            cap.release()
        if metrics_file_handle:
            metrics_file_handle.close()
        if annotated_writer:
            annotated_writer.release()
        if display:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Drawing helper -- kept separate so Streamlit can call it on raw frames
# ---------------------------------------------------------------------------

def _draw_overlay(frame, result: FrameResult, status_text: str) -> None:
    """Draw the status banner + counters + FPS onto the frame in-place."""

    h, w = frame.shape[:2]
    if result.violations:
        cv2.rectangle(frame, (0, 0), (w, 60), COLOR_BANNER_BAD, -1)
        cv2.putText(frame, status_text, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    else:
        cv2.rectangle(frame, (0, 0), (w, 60), COLOR_BANNER_OK, -1)
        cv2.putText(frame, "Exam Status: Normal", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)
    cv2.putText(frame, f"Persons: {result.persons}",
                (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    cv2.putText(frame, f"Phones: {result.phones}",
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    cv2.putText(frame, f"FPS: {result.fps:.1f}",
                (w - 130, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="object_detection_proctor",
        description="AI Exam Proctoring -- phone & person detection (Member 2).",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--webcam", type=int, default=0,
                     help="Webcam device index (default: 0).")
    src.add_argument("--image", type=str,
                     help="Path to a single image file.")
    src.add_argument("--video", type=str,
                     help="Path to a video file.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"Path to YOLO weights (default: {DEFAULT_MODEL}).")
    p.add_argument("--conf", type=float, default=0.5,
                   help="Confidence threshold (default: 0.5).")
    p.add_argument("--skip", type=int, default=3,
                   help="Run inference every N frames (default: 3).")
    p.add_argument("--stability", type=int, default=3,
                   help="Rolling-window size for flicker suppression (default: 3).")
    p.add_argument("--log", type=str, default=DEFAULT_LOG,
                   help=f"Violation log file (default: {DEFAULT_LOG}).")
    p.add_argument("--metrics", type=str, default=None,
                   help="Optional CSV file for per-frame metrics.")
    p.add_argument("--save", type=str, default=None,
                   help="Save annotated frames: image path, .mp4, or directory.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Stop after N frames (for benchmarks / evaluation).")
    p.add_argument("--headless", action="store_true",
                   help="Disable the OpenCV preview window (for Streamlit).")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.image:
        source: Union[int, str] = args.image
    elif args.video:
        source = args.video
    else:
        source = args.webcam

    for _ in run_detection(
        source=source,
        model_path=args.model,
        conf_threshold=args.conf,
        skip_frames=args.skip,
        stability_frames=args.stability,
        log_file=args.log,
        metrics_file=args.metrics,
        display=not args.headless,
        save_annotated=args.save,
        max_frames=args.max_frames,
    ):
        pass


if __name__ == "__main__":
    main()
