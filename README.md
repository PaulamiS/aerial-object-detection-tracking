# Aerial Object Detection & Tracking

YOLOv8 object detection on aerial imagery (VisDrone), multi-object tracking (ByteTrack),
a custom Kalman-filter state estimator, ONNX export, and a ROS 2 detection node.

## Pipeline
video -> YOLOv8 detector -> ByteTrack IDs -> Kalman filter (smoothing + prediction through missed detections) -> annotated video / ROS 2 `/detections`

## Results
> Fill these in from your own runs before publishing.

| Metric | Value |
|---|---|
| mAP50 (VisDrone val) | XX |
| mAP50-95 | XX |
| PyTorch inference | XX ms/frame |
| ONNX inference | XX ms/frame |

![demo](results/demo.gif)

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python train.py                                   # fine-tune on VisDrone (use a GPU)
python export_onnx.py runs/detect/train/weights/best.pt
python detect_track.py --source video.mp4 --model best.pt
python benchmark.py best.pt best.onnx video.mp4   # PyTorch vs ONNX speed
```

## How the Kalman filter is used
Each tracked object's centre is modelled with a constant-velocity state `[x, y, vx, vy]`.
Every frame the filter predicts the next position and corrects it with the detector's measurement.
When the detector misses an object, the filter keeps predicting its position for up to `--max-age`
frames (shown as red dots), so trajectories don't break.

## ROS 2
`ros2_node/detector_node.py` subscribes to `/image_raw` and publishes `vision_msgs/Detection2DArray`
on `/detections`. Run with `python3 ros2_node/detector_node.py --ros-args -p model:=best.onnx`.
