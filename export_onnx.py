"""Export trained weights to ONNX."""
import sys
from ultralytics import YOLO

weights = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/train/weights/best.pt"
print("Exported to:", YOLO(weights).export(format="onnx"))
