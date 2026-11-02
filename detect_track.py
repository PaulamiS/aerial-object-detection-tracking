"""Detection (YOLOv8) + ByteTrack IDs + Kalman-filter state estimation.

The Kalman filter smooths each track's centre and keeps predicting its
position for a few frames when the detector misses the object.
"""
import argparse
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

from kalman import ConstantVelocityKF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="input video path")
    ap.add_argument("--model", default="yolov8n.pt", help=".pt or .onnx weights")
    ap.add_argument("--out", default="results/output.mp4")
    ap.add_argument("--max-age", type=int, default=10,
                    help="frames to keep predicting a track after it is lost")
    args = ap.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    filters, missed = {}, defaultdict(int)
    trails = defaultdict(lambda: deque(maxlen=40))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        seen = set()

        if res.boxes.id is not None:
            ids = res.boxes.id.int().cpu().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            for tid, (x1, y1, x2, y2) in zip(ids, boxes):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                seen.add(tid)
                missed[tid] = 0
                if tid not in filters:
                    filters[tid] = ConstantVelocityKF(cx, cy)
                    est = filters[tid].x[:2]
                else:
                    filters[tid].predict()
                    est = filters[tid].update([cx, cy])
                trails[tid].append((int(est[0]), int(est[1])))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Tracks the detector missed: keep predicting with the Kalman filter
        for tid in list(filters):
            if tid in seen:
                continue
            missed[tid] += 1
            if missed[tid] > args.max_age:
                del filters[tid], missed[tid]
                trails.pop(tid, None)
                continue
            est = filters[tid].predict()
            trails[tid].append((int(est[0]), int(est[1])))
            cv2.circle(frame, (int(est[0]), int(est[1])), 5, (0, 0, 255), -1)  # red = predicted

        for pts in trails.values():
            for a, b in zip(pts, list(pts)[1:]):
                cv2.line(frame, a, b, (255, 200, 0), 2)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
