"gaze_api.py — FastAPI Integration Endpoint"

import base64
import io
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
from gaze_detector import GazeDetector, GazeConfig, GazeResult

# Router for the gaze module
router = APIRouter(prefix="/gaze", tags=["Gaze Detection"])

# ── Global detector instance (shared across requests) ──
_config = GazeConfig(suspicion_threshold_sec=2.0)
_detector = GazeDetector(_config)


# ── Response Models ──
class GazeResponse(BaseModel):
    gaze_direction: str
    is_suspicious: bool
    horizontal_ratio: float
    vertical_ratio: float
    face_detected: bool
    confidence: float
    annotated_image: Optional[str] = None  # Base64-encoded


class StatsResponse(BaseModel):
    total_suspicious_events: int
    total_suspicious_duration_sec: float
    session_duration_sec: float
    suspicious_percentage: float


# ── Single Frame Analysis ──
@router.post("/frame", response_model=GazeResponse)
async def analyze_frame(
    file: UploadFile = File(...),
    return_annotated: bool = True,
):
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return GazeResponse(
            gaze_direction="Error",
            is_suspicious=False,
            horizontal_ratio=0.5,
            vertical_ratio=0.5,
            face_detected=False,
            confidence=0.0,
        )

    result = _detector.detect(frame)

    annotated_b64 = None
    if return_annotated:
        annotated = _detector.draw(frame, result)
        _, buffer = cv2.imencode(".jpg", annotated)
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return GazeResponse(
        gaze_direction=result.gaze_direction,
        is_suspicious=result.is_suspicious,
        horizontal_ratio=round(result.horizontal_ratio, 4),
        vertical_ratio=round(result.vertical_ratio, 4),
        face_detected=result.face_detected,
        confidence=result.confidence,
        annotated_image=annotated_b64,
    )


# ── Video Clip Processing ──
@router.post("/video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Process an uploaded video file.
    Returns frame-by-frame gaze analysis and summary statistics.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)

    # Write to temp file for OpenCV
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = _detector.detect(frame)
        results.append({
            "frame": frame_count,
            "direction": result.gaze_direction,
            "suspicious": result.is_suspicious,
            "h_ratio": round(result.horizontal_ratio, 4),
            "v_ratio": round(result.vertical_ratio, 4),
        })
        frame_count += 1

    cap.release()
    os.unlink(tmp_path)

    stats = _detector.get_stats()
    return {
        "total_frames": frame_count,
        "frame_results": results[-30:],  # Last 30 frames for brevity
        "stats": stats,
    }


# ── Statistics ──
@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    return _detector.get_stats()


@router.post("/reset")
async def reset_stats():
    _detector.reset_stats()
    return {"message": "Stats reset successfully"}


# ── WebSocket for Real-Time Stream ──
@router.websocket("/ws")
async def gaze_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time gaze detection.
    Client sends base64-encoded frames; server returns JSON results.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            # Decode base64 frame
            try:
                img_data = base64.b64decode(data)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception:
                await websocket.send_json({"error": "Invalid image data"})
                continue

            if frame is None:
                await websocket.send_json({"error": "Could not decode frame"})
                continue

            result = _detector.detect(frame)

            await websocket.send_json({
                "gaze_direction": result.gaze_direction,
                "is_suspicious": result.is_suspicious,
                "horizontal_ratio": round(result.horizontal_ratio, 4),
                "vertical_ratio": round(result.vertical_ratio, 4),
                "face_detected": result.face_detected,
                "confidence": result.confidence,
            })

    except WebSocketDisconnect:
        print("WebSocket client disconnected")