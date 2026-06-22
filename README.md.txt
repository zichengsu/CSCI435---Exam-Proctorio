# AI-Based Exam Monitoring System - Phone & Person Detection (YOLOv8)

## Project Overview
This project is part of an AI-based exam monitoring system designed to enhance academic integrity during online/offline examinations. It detects prohibited objects such as mobile phones and monitors human presence using computer vision techniques.

The system uses **YOLOv8 (You Only Look Once)** for real-time object detection with a combination of:
- A **pre-trained YOLOv8 model** for general object and person detection
- A **custom-trained YOLOv8 model** fine-tuned on a Roboflow dataset for mobile phone detection

---

## Member Contribution (Debjani)
**Feature:** Phone & Person Detection

### Responsibilities:
- Developed real-time object detection system
- Fine-tuned YOLOv8 on custom dataset (Roboflow)
- Implemented mobile phone detection in exam environment
- Utilized pre-trained YOLOv8 model for person detection
- Tested model performance and accuracy

---

## Models Used

### Pre-Trained Model (YOLOv8 COCO)
- Used for general object detection
- Includes detection of:
  - Person 
  - Common objects (chairs, laptops, etc.)
- Model: `yolov8s.pt`

---

### Custom-Trained Model
- Trained on Roboflow dataset
- Detects:
  - Mobile Phones 
- Model file: `best.pt`
- Framework: YOLOv8 (Ultralytics)

---

## Model Performance
- Precision: ~0.96
- Recall: ~0.97
- mAP@50: ~0.98
- mAP@50-95: ~0.86

These results show high accuracy in detecting mobile phones in real-time environments.

---

## Tech Stack / Tools Used
- Python 
- YOLOv8 (Ultralytics)
- OpenCV
- Roboflow (Dataset Preparation)
- Google Colab (GPU Training)
- LabelImg (Annotation tool)

---

## Project Structure

Member2_Project/
│
├── src/ # Source code
├── scripts/ # Utility scripts
├── sample_outputs/ # Detection results
│
├── best.pt # Custom-trained model
├── data.yaml # Dataset configuration
├── detect.py # Real-time detection script
├── train.py # Training script
├── requirements.txt # Dependencies
│
└── README.md


---

## How to Run the Project

### Install dependencies
```bash
pip install -r requirements.txt

Run real-time detection (webcam)
python detect.py



The system performs:

Real-time person detection 
Mobile phone detection 
Bounding box visualization
High accuracy detection in exam environment

Key Features
Real-time object detection
Custom-trained YOLOv8 model
Pre-trained model integration
High accuracy detection
Lightweight and fast inference

Learning Outcomes
Understanding of deep learning-based object detection
Experience with YOLOv8 architecture
Dataset preparation using Roboflow
Model training on GPU (Google Colab)
Real-time computer vision implementation

Conclusion
This project successfully demonstrates the use of AI in academic monitoring systems using YOLOv8. The combination of pre-trained and custom-trained models ensures accurate detection of both persons and prohibited objects such as mobile phones.
