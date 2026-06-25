"""
face_recognizer.py -- face detection & recognition
==================================================

Uses the `face_recognition` library: register a student photo, then for each
detected face run compare_faces(tolerance=0.5) + face_distance and label it as
a verified student or an unknown person.

To enable recognition:
    1.  pip install face_recognition   (see requirements.txt / README for the
        dlib install note)
    2.  create a folder `known_faces/` and put one photo per registered
        student in it, named after the student, e.g. `student_name.jpg`.

If the library or folder is missing, it falls back to plain OpenCV face
DETECTION so the rest of the app still runs.

`face_alert` is True when a face is present but NOT a registered student
(possible impersonation), which the violation engine uses in its score.
"""

import os
import numpy as np
import cv2

KNOWN_FACES_DIR = "known_faces"

# Try the optional high-accuracy path (face_recognition library).
# NOTE: when its models are missing, face_recognition calls quit()/SystemExit
# at import time, which is NOT an Exception subclass -- so we catch
# BaseException here to make sure a broken face install can never take down
# the whole app (it just falls back to OpenCV face detection instead).
try:
    import face_recognition  # noqa: F401
    _HAS_FR = True
except BaseException:
    _HAS_FR = False


class FaceRecognizer:
    def __init__(self, known_dir=KNOWN_FACES_DIR):
        self.mode = "recognition" if _HAS_FR else "detection_only"
        self.known_encodings = []
        self.known_names = []

        if self.mode == "recognition" and os.path.isdir(known_dir):
            self._load_known_faces(known_dir)
            if not self.known_names:
                # Library present but no enrolled students -> detection only.
                self.mode = "detection_only"

        if self.mode == "detection_only":
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade)

    def _load_known_faces(self, known_dir):
        for fname in os.listdir(known_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(known_dir, fname)
                image = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    self.known_encodings.append(encs[0])
                    self.known_names.append(os.path.splitext(fname)[0])

    def process_frame(self, frame):
        """Return a dict: name, recognized, face_alert, box (x,y,w,h) or None."""
        if self.mode == "recognition":
            return self._recognise(frame)
        return self._detect_only(frame)

    # ----- high-accuracy path: face_recognition logic -----
    def _recognise(self, frame):
        # Pipeline:
        # face_locations -> face_encodings -> compare_faces(tolerance=0.5),
        # using face_distance to pick the closest registered student.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        if not locations:
            return {"name": "No Face", "recognized": False,
                    "face_alert": False, "box": None, "distance": None}

        encodings = face_recognition.face_encodings(rgb, locations)
        top, right, bottom, left = locations[0]
        name, recognized, distance = "Unknown Person", False, None
        if encodings:
            matches = face_recognition.compare_faces(
                self.known_encodings, encodings[0], tolerance=0.5
            )
            distances = face_recognition.face_distance(
                self.known_encodings, encodings[0]
            )
            if len(distances):
                best = int(np.argmin(distances))
                distance = round(float(distances[best]), 4)
                if matches[best]:
                    name = self.known_names[best]
                    recognized = True
        box = (left, top, right - left, bottom - top)
        return {"name": name, "recognized": recognized,
                "face_alert": not recognized, "box": box, "distance": distance}

    # ----- fallback path (works with zero setup) -----
    def _detect_only(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return {"name": "No Face", "recognized": False,
                    "face_alert": False, "box": None}
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        # In detection-only mode we cannot verify identity, so don't raise an
        # alert -- just show that a face is present.
        return {"name": "Unknown", "recognized": False,
                "face_alert": False, "box": (int(x), int(y), int(w), int(h))}

    @staticmethod
    def draw(frame, result):
        box = result.get("box")
        if box is not None:
            x, y, w, h = box
            ok = result.get("recognized")
            color = (0, 200, 0) if ok else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, result.get("name", ""), (x, max(y - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame
