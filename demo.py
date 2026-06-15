"gaze_live_demo.py — Webcam Demo"

import argparse
import cv2
import time
from gaze_detector import GazeDetector, GazeConfig


def main():
    parser = argparse.ArgumentParser(description="Gaze Detection Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--suspicion-time", type=float, default=2.0,
                        help="Seconds of off-center gaze before flagging")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    args = parser.parse_args()

    # ── Configuration ──
    config = GazeConfig(
        suspicion_threshold_sec=args.suspicion_time,
        draw_eye_mesh=True,
        draw_gaze_arrow=True,
    )
    detector = GazeDetector(config)

    # ── Camera Setup ──
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    print("=" * 60)
    print("  AI Exam Proctor — Gaze Detection Module")
    print("  CSCI435 Project")
    print("=" * 60)
    print(f"  Camera: {args.camera}")
    print(f"  Suspicion threshold: {args.suspicion_time}s")
    print("  Press 'q' to quit, 'r' to reset stats")
    print("=" * 60)

    prev_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Frame capture failed.")
            break

        # ── Detect Gaze ──
        result = detector.detect(frame)

        # ── Draw Overlays ──
        annotated = detector.draw(frame, result)

        # ── FPS Counter ──
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time
        cv2.putText(
            annotated, f"FPS: {fps:.1f}", (annotated.shape[1] - 120, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        # ── Stats Display ──
        stats = detector.get_stats()
        cv2.putText(
            annotated,
            f"Violations: {stats['total_suspicious_events']}  "
            f"Duration: {stats['total_suspicious_duration_sec']:.1f}s",
            (30, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # ── Display ──
        cv2.imshow("Gaze Detection — AI Exam Proctor", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            detector.reset_stats()
            print("Stats reset.")

    # ── Cleanup ──
    stats = detector.get_stats()
    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Duration: {stats['session_duration_sec']:.1f}s")
    print(f"  Suspicious events: {stats['total_suspicious_events']}")
    print(f"  Suspicious duration: {stats['total_suspicious_duration_sec']:.2f}s")
    print(f"  Suspicious %: {stats['suspicious_percentage']:.1f}%")
    print("=" * 60)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()