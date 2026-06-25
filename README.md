# ARGUS — Intelligent Exam Integrity Monitoring (CSCI435)

A single Streamlit web application that integrates four computer-vision
techniques into one live exam-proctoring workflow.

## Vision capabilities (4+)

| Capability | Tech | File |
|-----------|------|------|
| Face detection / recognition | OpenCV / face_recognition | `face_recognizer.py` |
| Phone & person detection | YOLOv8 (custom + pretrained) | `object_detection_proctor.py` |
| Eye-gaze tracking (keypoints) | MediaPipe Face Mesh | `gaze_detector.py` |
| Motion / camera-block detection | OpenCV MOG2 + frame differencing | `motion_detector.py` |
| Frontend, database & integration | Streamlit, SQLite, Pandas | `app.py`, `database.py`, `proctor_engine.py` |

## Input modalities
1. Live webcam stream
2. Uploaded image
3. Uploaded video file

## How it fits together
`app.py` (Streamlit UI) calls `proctor_engine.py`, which runs all four modules
on every frame and combines them into a single risk score using the
violation-engine weighting (face +30, phone +50, gaze +20, on top of the motion
score). Results are drawn as bounding boxes and a colour-coded status banner,
and every violation is saved to `proctor.db` (SQLite) for the dashboard and
session-history views (read back with pandas).

## Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate     macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```
Make sure `yolov8s.pt` and `best.pt` (the YOLO models) are in the project root.

## Run
```bash
streamlit run app.py
```
A browser tab opens at `http://localhost:8501`. Pick an input source in the
sidebar, toggle modules, and press Start. Local hosting is fine for the demo.

## Face recognition setup
`face_recognizer.py` uses the `face_recognition` library. To enable recognition,
install it (see the dlib note for your OS) and create a `known_faces/` folder
with one photo per registered student, named after them (e.g. `student_name.jpg`).
If the library or folder is missing, it falls back to OpenCV face detection so
the app still runs.
