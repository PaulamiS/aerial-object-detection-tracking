"""Fine-tune YOLOv8 on VisDrone (aerial images). Run on Colab/Kaggle GPU."""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="VisDrone.yaml", epochs=50, imgsz=640)   # dataset auto-downloads
metrics = model.val()
print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)
