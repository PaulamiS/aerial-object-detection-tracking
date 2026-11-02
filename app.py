"""Streamlit dashboard for the aerial detection + tracking pipeline.

Run:  streamlit run app.py
"""
import os
import tempfile
import time
from collections import defaultdict, deque

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from kalman import ConstantVelocityKF

st.set_page_config(page_title="Aerial Detection & Tracking", layout="wide")
st.title("Aerial Object Detection & Tracking")
st.caption("YOLOv8 (VisDrone) + ByteTrack + Kalman-filter state estimation")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model weights", "best.pt", help="Use best.pt or best.onnx")
    conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
    max_age = st.slider("Kalman max-age (frames)", 1, 30, 10,
                        help="How long to keep predicting a lost object")
    max_frames = st.slider("Max frames to process", 20, 600, 150, 10,
                           help="Lower = faster on a CPU")
    show_trails = st.checkbox("Show trails and predicted points", value=True)

tab_run, tab_results = st.tabs(["Run on a video", "Model results"])

# ---------------- Tab 1: run pipeline ----------------
with tab_run:
    uploaded = st.file_uploader("Upload an aerial video", type=["mp4", "mov", "avi"])
    run = st.button("Run detection + tracking", type="primary", disabled=uploaded is None)

    if run and uploaded is not None:
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Put best.pt in this folder.")
            st.stop()

        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_in.write(uploaded.read())
        tmp_in.close()

        cap = cv2.VideoCapture(tmp_in.name)
        if not cap.isOpened():
            st.error("Could not open the video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames, max_frames)
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        model = YOLO(model_path)  # fresh model so tracker state starts clean
        names = model.names

        filters, missed = {}, defaultdict(int)
        trails = defaultdict(lambda: deque(maxlen=40))
        id_class = {}                 # track id -> class name
        det_per_frame, frame_ms = [], []

        preview = st.empty()
        bar = st.progress(0.0, text="Processing...")

        n = 0
        while n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              conf=conf, verbose=False)[0]
            frame_ms.append((time.perf_counter() - t0) * 1000)

            seen = set()
            if res.boxes.id is not None:
                ids = res.boxes.id.int().cpu().tolist()
                boxes = res.boxes.xyxy.cpu().numpy()
                clss = res.boxes.cls.int().cpu().tolist()
                for tid, (x1, y1, x2, y2), c in zip(ids, boxes, clss):
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    seen.add(tid)
                    missed[tid] = 0
                    id_class.setdefault(tid, names[c])
                    if tid not in filters:
                        filters[tid] = ConstantVelocityKF(cx, cy)
                        est = filters[tid].x[:2]
                    else:
                        filters[tid].predict()
                        est = filters[tid].update([cx, cy])
                    trails[tid].append((int(est[0]), int(est[1])))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{names[c]} #{tid}", (int(x1), max(int(y1) - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            for tid in list(filters):
                if tid in seen:
                    continue
                missed[tid] += 1
                if missed[tid] > max_age:
                    del filters[tid], missed[tid]
                    trails.pop(tid, None)
                    continue
                est = filters[tid].predict()
                trails[tid].append((int(est[0]), int(est[1])))
                if show_trails:
                    cv2.circle(frame, (int(est[0]), int(est[1])), 5, (0, 0, 255), -1)

            if show_trails:
                for pts in trails.values():
                    for a, b in zip(pts, list(pts)[1:]):
                        cv2.line(frame, a, b, (255, 200, 0), 2)

            det_per_frame.append(len(seen))
            writer.write(frame)
            if n % 2 == 0:
                preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            n += 1
            bar.progress(min(n / max(total, 1), 1.0), text=f"Processing frame {n}/{total}")

        cap.release()
        writer.release()
        bar.empty()

        if n == 0:
            st.error("No frames were read from the video.")
            st.stop()

        st.success(f"Processed {n} frames.")

        # ----- summary metrics -----
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frames processed", n)
        c2.metric("Unique objects tracked", len(id_class))
        c3.metric("Avg objects / frame", f"{sum(det_per_frame) / n:.1f}")
        c4.metric("Avg time / frame", f"{sum(frame_ms) / n:.0f} ms")

        # ----- charts -----
        left, right = st.columns(2)
        with left:
            st.subheader("Objects detected per frame")
            st.line_chart(pd.DataFrame({"objects": det_per_frame}))
        with right:
            st.subheader("Tracked objects by class")
            counts = pd.Series(list(id_class.values())).value_counts()
            st.bar_chart(counts)

        with open(out_path, "rb") as f:
            st.download_button("Download annotated video", f.read(),
                               file_name="output_tracked.mp4", mime="video/mp4")

# ---------------- Tab 2: model results ----------------
with tab_results:
    st.subheader("Detection accuracy (VisDrone validation set)")
    a, b = st.columns(2)
    a.metric("mAP50", "28.9%")
    b.metric("mAP50-95", "16.1%")

    st.subheader("Inference speed (CPU, end-to-end per frame)")
    s1, s2 = st.columns(2)
    s1.metric("PyTorch (.pt)", "319.8 ms")
    s2.metric("ONNX Runtime (.onnx)", "202.9 ms", delta="-36.6%", delta_color="inverse")
    st.bar_chart(pd.DataFrame({"ms / frame": [319.8, 202.9]},
                              index=["PyTorch", "ONNX Runtime"]))

    if os.path.exists("results/demo.gif"):
        st.subheader("Demo")
        st.image("results/demo.gif")

    st.caption("Numbers measured on an Intel Core i3-1125G4 CPU. "
               "VisDrone is hard because objects are small and crowded.")