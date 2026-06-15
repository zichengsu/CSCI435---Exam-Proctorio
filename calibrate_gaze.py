"calibrate_gaze.py — Calibration Utility"

import cv2
import numpy as np
import time
from gaze_detector import GazeDetector, GazeConfig

DIRECTIONS = [
    ("CENTER", "Look directly at the camera"),
    ("LEFT", "Look to your left"),
    ("RIGHT", "Look to your right"),
    ("UP", "Look up"),
    ("DOWN", "Look down"),
]

SAMPLE_DURATION = 3.0  # seconds per direction


def calibrate():
    print("=" * 60)
    print("  Gaze Calibration Tool")
    print("  You will be asked to look in 5 directions.")
    print(f"  Hold each direction for {SAMPLE_DURATION} seconds.")
    print("=" * 60)

    config = GazeConfig(suspicion_threshold_sec=99)  
    detector = GazeDetector(config)
    cap = cv2.VideoCapture(0)

    calibration_data = {}

    for direction, instruction in DIRECTIONS:
        print(f"\n>> {instruction} ({direction})")
        print("   Starting in 3 seconds...")
        time.sleep(3)

        samples_h = []
        samples_v = []
        start = time.time()

        while time.time() - start < SAMPLE_DURATION:
            ret, frame = cap.read()
            if not ret:
                continue

            result = detector.detect(frame)
            if result.face_detected:
                samples_h.append(result.horizontal_ratio)
                samples_v.append(result.vertical_ratio)

            annotated = detector.draw(frame, result)
            remaining = SAMPLE_DURATION - (time.time() - start)
            cv2.putText(
                annotated, f"{instruction} - {remaining:.1f}s",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
            )
            cv2.imshow("Calibration", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if samples_h:
            avg_h = float(np.mean(samples_h))
            avg_v = float(np.mean(samples_v))
            calibration_data[direction] = {"h": avg_h, "v": avg_v}
            print(f"   → H: {avg_h:.3f}, V: {avg_v:.3f}")
        else:
            print(f"   → No face detected during {direction}!")

    # ── Compute Tuned Thresholds ──
    if "CENTER" in calibration_data:
        center_h = calibration_data["CENTER"]["h"]
        center_v = calibration_data["CENTER"]["v"]

        # Left/Right boundaries
        left_h = calibration_data.get("LEFT", {}).get("h", center_h - 0.2)
        right_h = calibration_data.get("RIGHT", {}).get("h", center_h + 0.2)

        # Up/Down boundaries
        up_v = calibration_data.get("UP", {}).get("v", center_v - 0.2)
        down_v = calibration_data.get("DOWN", {}).get("v", center_v + 0.2)

        # Set thresholds at midpoint between center and each extreme
        h_min = (center_h + left_h) / 2
        h_max = (center_h + right_h) / 2
        v_min = (center_v + up_v) / 2
        v_max = (center_v + down_v) / 2

        print("\n" + "=" * 60)
        print("  CALIBRATION RESULTS")
        print("=" * 60)
        print(f"  horizontal_center_min: {h_min:.3f}")
        print(f"  horizontal_center_max: {h_max:.3f}")
        print(f"  vertical_center_min:   {v_min:.3f}")
        print(f"  vertical_center_max:   {v_max:.3f}")
        print("\n  Use these values in GazeConfig():")
        print(f"""
    GazeConfig(
        horizontal_center_min={h_min:.3f},
        horizontal_center_max={h_max:.3f},
        vertical_center_min={v_min:.3f},
        vertical_center_max={v_max:.3f},
    )
        """)
    else:
        print("\nCalibration failed: no center data collected.")

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrate()