"""
Evaluate a YOLOv8 model on a held-out validation set + benchmark runtime
performance (FPS, latency) on a representative video.

Outputs a JSON file ``evaluation_report.json`` containing everything the
CSCI435 report needs for the "Experiments and results" section:

    * mAP@0.5, mAP@0.5:0.95
    * per-class precision / recall
    * inference FPS (mean + std)
    * per-frame latency (mean / p50 / p95 / max, in ms)
    * detection counts

Usage:
    python evaluate_model.py --weights runs/detect/proctor_train/weights/best.pt \\
        --data-yaml datasets/proctor_dataset/data.yaml \\
        --benchmark-video samples/exam_sample.mp4 \\
        --output evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_model",
        description="Evaluate YOLOv8 mAP + runtime performance for the report.",
    )
    p.add_argument("--weights", type=str, required=True,
                   help="Path to YOLO weights (.pt).")
    p.add_argument("--data-yaml", type=str, required=True,
                   help="Path to data.yaml for the validation set.")
    p.add_argument("--benchmark-video", type=str, default=None,
                   help="Optional video file for FPS/latency benchmark.")
    p.add_argument("--benchmark-frames", type=int, default=200,
                   help="Number of frames to benchmark (default: 200).")
    p.add_argument("--conf", type=float, default=0.5,
                   help="Confidence threshold (default: 0.5).")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Inference image size (default: 640).")
    p.add_argument("--device", type=str, default="0",
                   help="Device: 0 / cpu (default: 0).")
    p.add_argument("--output", type=str, default="evaluation_report.json",
                   help="Output JSON file (default: evaluation_report.json).")
    return p


def _run_map_evaluation(model: YOLO, data_yaml: str, imgsz: int,
                        device: str) -> dict[str, Any]:
    """Run ultralytics' built-in validator and extract headline metrics."""
    print(f"[eval] Running mAP validation on {data_yaml} ...")
    metrics = model.val(data=data_yaml, imgsz=imgsz, device=device, verbose=True)

    # `metrics` exposes per-class results via metrics.box
    names = model.names
    per_class = {}
    try:
        for i, name in names.items():
            if i not in (0, 67):  # we only care about person & phone
                continue
            per_class[name] = {
                "precision": float(metrics.box.p[i]) if i < len(metrics.box.p) else None,
                "recall": float(metrics.box.r[i]) if i < len(metrics.box.r) else None,
                "map50": float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else None,
                "map50_95": float(metrics.box.ap[i]) if i < len(metrics.box.ap) else None,
            }
    except Exception as e:
        print(f"[eval] Warning: could not extract per-class metrics ({e})")

    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
    }


def _run_benchmark(model: YOLO, video_path: str, num_frames: int,
                   conf: float, imgsz: int) -> dict[str, Any]:
    """Measure per-frame inference latency and FPS on a video."""
    print(f"[eval] Benchmarking on {video_path} ({num_frames} frames) ...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    latencies_ms: list[float] = []
    fps_history: list[float] = []
    prev_time = time.time()
    frame_idx = 0

    while cap.isOpened() and frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        model(frame, conf=conf, classes=[0, 67], imgsz=imgsz, verbose=False)
        t1 = time.time()

        latency_ms = (t1 - t0) * 1000.0
        latencies_ms.append(latency_ms)

        fps = 1.0 / max(t1 - prev_time, 1e-6)
        fps_history.append(fps)
        prev_time = t1
        frame_idx += 1

    cap.release()

    def percentile(data: list[float], pct: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        k = int(round((pct / 100.0) * (len(s) - 1)))
        return s[k]

    return {
        "frames_evaluated": len(latencies_ms),
        "fps_mean": round(statistics.mean(fps_history), 2) if fps_history else 0.0,
        "fps_std": round(statistics.pstdev(fps_history), 2) if len(fps_history) > 1 else 0.0,
        "latency_ms_mean": round(statistics.mean(latencies_ms), 2) if latencies_ms else 0.0,
        "latency_ms_p50": round(percentile(latencies_ms, 50), 2),
        "latency_ms_p95": round(percentile(latencies_ms, 95), 2),
        "latency_ms_max": round(max(latencies_ms), 2) if latencies_ms else 0.0,
        "meets_realtime_threshold": (
            statistics.mean(fps_history) >= 10.0 if fps_history else False
        ),
        # The brief: "minimum of ten frames per second for video processing"
    }


def main() -> None:
    args = _build_arg_parser().parse_args()

    if not Path(args.weights).is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    if not Path(args.data_yaml).is_file():
        raise SystemExit(f"data.yaml not found: {args.data_yaml}")

    print(f"[eval] Loading model: {args.weights}")
    model = YOLO(args.weights)

    report: dict[str, Any] = {
        "weights": args.weights,
        "data_yaml": args.data_yaml,
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "device": args.device,
    }

    report["accuracy"] = _run_map_evaluation(
        model, args.data_yaml, args.imgsz, args.device
    )

    if args.benchmark_video:
        if not Path(args.benchmark_video).is_file():
            print(f"[eval] Warning: benchmark video not found ({args.benchmark_video}); skipping FPS benchmark.")
        else:
            report["performance"] = _run_benchmark(
                model, args.benchmark_video, args.benchmark_frames,
                args.conf, args.imgsz,
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[eval] Report written to: {out_path}")
    print("[eval] Summary:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
