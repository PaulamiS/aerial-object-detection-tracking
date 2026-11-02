"""Compare per-frame inference time: PyTorch (.pt) vs ONNX Runtime (.onnx)."""
import sys
import time

import cv2
from ultralytics import YOLO


def avg_ms(weights, frames):
    model = YOLO(weights)
    model(frames[0], verbose=False)                      # warm-up
    t = time.perf_counter()
    for f in frames:
        model(f, verbose=False)
    return (time.perf_counter() - t) / len(frames) * 1000


pt, onnx, video = sys.argv[1], sys.argv[2], sys.argv[3]
cap, frames = cv2.VideoCapture(video), []
while len(frames) < 100:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)

print(f"PyTorch: {avg_ms(pt, frames):.1f} ms/frame")
print(f"ONNX:    {avg_ms(onnx, frames):.1f} ms/frame")
