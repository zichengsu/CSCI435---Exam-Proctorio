# ARGUS — Intelligent Exam Integrity Monitoring (CSCI435)

A single Streamlit web application that integrates five computer-vision
techniques into one live exam-proctoring workflow.

## Vision capabilities (5)

| Capability | Tech | File |
|---|---|---|
| Face detection / recognition | OpenCV / face_recognition | `face_recognizer.py` |
| Phone & person detection | YOLOv8 (custom + pretrained) | `object_detection_proctor.py` |
| Eye-gaze tracking (keypoints) | MediaPipe Face Mesh | `gaze_detector.py` |
| Motion / camera-block detection | OpenCV MOG2 + frame differencing | `motion_detector.py` |
| Low-light image enhancement | OpenCV CLAHE | `proctor_engine.py` |
| Frontend, database & integration | Streamlit, SQLite, Pandas | `app.py`, `database.py`, `proctor_engine.py` |

## Input modalities
- Live webcam stream
- Uploaded image
- Uploaded video file

## How it fits together
`app.py` (the Streamlit UI) calls `proctor_engine.py`, which runs all the vision
modules on every frame and combines them into a single risk score using a
weighted violation engine (face +30, phone +50, gaze +20, on top of the motion
score). Dark frames are first brightened by the CLAHE enhancement stage for
robustness in poor lighting. Results are drawn as bounding boxes and a
colour-coded status banner (Normal / Warning / Violation), and every violation is
saved to `proctor.db` (SQLite) for the Dashboard and Session History views, which
are read back with pandas.

## Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate     macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```
Make sure `yolov8s.pt` and `best.pt` (the YOLO models) are in the project root.

> **Note:** `requirements.txt` pins `numpy<2`, because the installable PyTorch
> build requires NumPy 1.x. If you ever see `RuntimeError: Numpy is not
> available`, run `pip install "numpy<2" "pandas<3"` and restart.

## Run
```bash
streamlit run app.py
```
A browser tab opens at `http://localhost:8501`. Pick an input source in the
sidebar, toggle modules, and press Start. Local hosting is fine for the demo.

## Face recognition setup
`face_recognizer.py` uses the `face_recognition` library. To enable recognition,
install it (note: `dlib` can be awkward on Windows — use `conda install -c
conda-forge dlib` if `pip` fails) and create a `known_faces/` folder with one
photo per registered student, named after them (e.g. `student_name.jpg`). If the
library or folder is missing, the app falls back to OpenCV face detection so it
still runs.
