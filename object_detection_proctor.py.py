from ultralytics import YOLO
import cv2
import time
from collections import deque


model = YOLO("yolov8s.pt")


def run_detection(source=0,
                  conf_threshold=0.5,
                  skip_frames=3,
                  stability_frames=3,
                  log_file="violations.txt",
                  display=True):
    """
    AI Exam Proctoring Module - Member 2

    - Phone detection
    - Multiple person detection
    - Streamlit compatible
    """

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return None

    print("AI Exam Proctor Started!")

    frame_count = 0
    last_results = None

    # Stability buffers
    person_history = deque(maxlen=stability_frames)
    phone_history = deque(maxlen=stability_frames)

    last_logged_violation = ""

    # FPS tracking
    fps_history = deque(maxlen=10)
    prev_time = time.time()

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        person_count = 0
        phone_count = 0
        violations = []

        # ==========================
        # Detection (skip frames)
        # ==========================
        if frame_count % skip_frames == 0:

            results = model(
                frame,
                conf=conf_threshold,
                classes=[0, 67],
                verbose=False
            )

            last_results = results[0]

            for box in last_results.boxes:

                class_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < conf_threshold:
                    continue

                if class_id == 0:
                    person_count += 1
                elif class_id == 67:
                    phone_count += 1

        # ==========================
        # Stability logic
        # ==========================
        person_history.append(person_count)
        phone_history.append(phone_count)

        stable_persons = max(person_history)
        stable_phones = max(phone_history)

        # ==========================
        # Draw boxes
        # ==========================
        if last_results is not None:

            for box in last_results.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                if conf < conf_threshold:
                    continue

                if class_id == 0:
                    label = f"PERSON {conf:.2f}"
                    color = (0, 165, 255) if stable_persons > 1 else (0, 255, 0)

                elif class_id == 67:
                    label = f"PHONE {conf:.2f}"
                    color = (0, 0, 255)

                else:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ==========================
        # Violation rules
        # ==========================
        if stable_phones > 0:
            violations.append(f"Phone Detected ({stable_phones})")

        if stable_persons > 1:
            violations.append(f"Multiple Persons ({stable_persons})")

        # ==========================
        # Logging
        # ==========================
        if violations:

            status_text = " | ".join(violations)

            if status_text != last_logged_violation:

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                with open(log_file, "a") as f:
                    f.write(f"{timestamp} - {status_text}\n")

                last_logged_violation = status_text

        else:
            last_logged_violation = ""

        # ==========================
        # FPS
        # ==========================
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)

        # ==========================
        # Return data for Streamlit
        # ==========================
        output_data = {
            "persons": stable_persons,
            "phones": stable_phones,
            "violations": violations,
            "fps": round(avg_fps, 2)
        }

        # ==========================
        # Display mode (optional)
        # ==========================
        if display:

            if violations:
                cv2.rectangle(frame, (0, 0),
                              (frame.shape[1], 60), (0, 0, 200), -1)
                cv2.putText(frame, status_text, (10, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (0, 0),
                              (frame.shape[1], 60), (0, 120, 0), -1)
                cv2.putText(frame, "Exam Status: Normal", (10, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, f"Persons: {stable_persons}",
                        (10, frame.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Phones: {stable_phones}",
                        (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                        (frame.shape[1] - 130, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("AI Exam Proctoring System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # IMPORTANT: for Streamlit integration
        yield output_data

    cap.release()
    cv2.destroyAllWindows()


# ==========================
# Standalone run
# ==========================
if __name__ == "__main__":

    for _ in run_detection(display=True):
        pass