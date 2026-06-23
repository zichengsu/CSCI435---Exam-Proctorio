# Member 4 - Motion Detection and Violation Engine

This folder contains my Member 4 contribution for the AI Exam Proctoring System project.

## Module Description

This module detects suspicious student movement and camera blocking during an online exam video. It uses OpenCV video processing, background subtraction, frame differencing, and rule-based violation logic.

## Features

* Motion detection using OpenCV
* Background subtraction using MOG2
* Frame differencing
* Excessive movement detection
* Camera blocked detection
* Risk score calculation
* Violation counter
* CSV violation log generation
* Output video with text overlays
* FPS and latency measurement

## Tools and Libraries Used

* Python
* OpenCV
* NumPy
* Pandas
* Google Colab

## Test Results

| Test Video         | Expected Result          | Actual Result  | Status |
| ------------------ | ------------------------ | -------------- | ------ |
| Normal sitting     | No violation             | 0 violations   | Pass   |
| Excessive movement | Movement violation       | 41 violations  | Pass   |
| Camera blocked     | Camera blocked violation | 176 violations | Pass   |

## Files Included

* `Member4_Motion_Detection_Violation_Engine.ipynb`
* CSV log files
* Output videos
* Screenshots of results

## Integration Notes

This module can later receive outputs from other team members:

* Member 1: `face_alert`
* Member 2: `phone_alert`
* Member 3: `gaze_alert`

These alerts can be combined with the motion detection result to produce the final exam violation score.
