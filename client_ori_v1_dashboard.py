import argparse
import csv
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pathlib
import torch
from flask import Flask, Response, jsonify, render_template, send_file

try:
    import psutil
except ImportError:
    psutil = None

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.segment.general import process_mask
from utils.torch_utils import select_device


def get_class_color(cls_name: str) -> tuple:
    np.random.seed(abs(hash(cls_name)) % (2 ** 32))
    return tuple(int(c) for c in np.random.randint(40, 255, 3))


def centroid_from_mask_or_box(mask_bool: np.ndarray, box: tuple) -> tuple:
    ys, xs = np.where(mask_bool)
    if len(xs) > 0:
        return int(xs.mean()), int(ys.mean())
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def point_in_polygon(point: tuple, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def draw_polygon(frame: np.ndarray, poly: np.ndarray, color: tuple, title: str):
    cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
    x, y = poly[0]
    cv2.putText(frame, title, (int(x), max(20, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


@dataclass
class TrackState:
    centroid: tuple
    bbox: tuple
    cls_name: str
    last_seen: int
    first_seen: int
    counted: bool = False
    missed: int = 0


class ConveyorTracker:
    def __init__(self, prefix: str, timeout_frames: int, match_dist_px: float):
        self.prefix = prefix
        self.timeout_frames = timeout_frames
        self.match_dist_px = match_dist_px
        self.next_id = 1
        self.tracks = {}
        self.count_by_class = {}
        self.counted_ids = set()
        self.last_detection_ts = "-"

    def _new_id(self) -> str:
        tid = f"{self.prefix}_{self.next_id}"
        self.next_id += 1
        return tid

    def update(self, detections: list, frame_idx: int):
        assigned_tracks = set()
        assigned_dets = set()
        pairs = []
        for det_idx, det in enumerate(detections):
            dcx, dcy = det["centroid"]
            for tid, st in self.tracks.items():
                dist = np.hypot(dcx - st.centroid[0], dcy - st.centroid[1])
                if dist <= self.match_dist_px:
                    pairs.append((dist, tid, det_idx))

        pairs.sort(key=lambda x: x[0])
        for _, tid, det_idx in pairs:
            if tid in assigned_tracks or det_idx in assigned_dets:
                continue
            det = detections[det_idx]
            st = self.tracks[tid]
            st.centroid = det["centroid"]
            st.bbox = det["bbox"]
            st.cls_name = det["cls_name"]
            st.last_seen = frame_idx
            st.missed = 0
            assigned_tracks.add(tid)
            assigned_dets.add(det_idx)

        for det_idx, det in enumerate(detections):
            if det_idx in assigned_dets:
                continue
            tid = self._new_id()
            self.tracks[tid] = TrackState(
                centroid=det["centroid"],
                bbox=det["bbox"],
                cls_name=det["cls_name"],
                first_seen=frame_idx,
                last_seen=frame_idx,
            )

        for tid, st in list(self.tracks.items()):
            if st.last_seen != frame_idx:
                st.missed += 1
            if st.missed > self.timeout_frames:
                del self.tracks[tid]

    def mark_counted(self, tid: str):
        if tid in self.counted_ids:
            return False
        st = self.tracks.get(tid)
        if st is None:
            return False
        self.counted_ids.add(tid)
        st.counted = True
        self.count_by_class[st.cls_name] = self.count_by_class.get(st.cls_name, 0) + 1
        self.last_detection_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return True


class InferenceEngine:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.running = True
        self.stop_flag = False
        self.last_frame = None
        self.last_jpeg = None
        self.frame_idx = 0
        self.last_fps = 0.0
        self.total_detections = 0
        self.event_logs = deque(maxlen=200)
        self.thread = None
        self.device_name = "CPU"
        self.model_name = Path(args.weights).name
        self.version = "v1.0.0"
        self.run_dir = Path(args.project) / args.name
        self.csv_path = self.run_dir / "dashboard_count_log.csv"
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        if self.csv_path.stat().st_size == 0:
            self.csv_writer.writerow(["timestamp", "conveyor_id", "tracker_id"])
        self.snapshot_dir = self.run_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.c1_tracker = ConveyorTracker("C1", args.track_timeout, args.match_dist)
        self.c2_tracker = ConveyorTracker("C2", args.track_timeout, args.match_dist)
        self.c1_poly = None
        self.c2_poly = None
        self.inference_speed_ms = 0.0

    def _parse_roi(self, roi_str, w, h):
        if not roi_str:
            return None
        pts = []
        for p in roi_str.split(";"):
            x, y = p.split(",")
            pts.append((int(x), int(y)))
        if len(pts) != 4:
            return None
        return np.array(pts, dtype=np.int32)

    def _default_rois(self, w, h):
        c1 = np.array(
            [(int(0.08 * w), int(0.22 * h)), (int(0.48 * w), int(0.22 * h)), (int(0.48 * w), int(0.86 * h)), (int(0.08 * w), int(0.86 * h))],
            dtype=np.int32,
        )
        c2 = np.array(
            [(int(0.52 * w), int(0.22 * h)), (int(0.92 * w), int(0.22 * h)), (int(0.92 * w), int(0.86 * h)), (int(0.52 * w), int(0.86 * h))],
            dtype=np.int32,
        )
        return c1, c2

    def add_event(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            self.event_logs.appendleft(f"[{ts}] {text}")

    def start(self):
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        device_obj = select_device(self.args.device)
        self.device_name = str(device_obj)
        model = DetectMultiBackend(self.args.weights, device=device_obj)
        stride, names = model.stride, model.names
        imgsz = check_img_size(self.args.imgsz, s=stride)
        model.warmup(imgsz=(1, 3, imgsz, imgsz))
        is_webcam = self.args.source.isnumeric()
        dataset = LoadStreams(self.args.source, img_size=imgsz, stride=stride) if is_webcam else LoadImages(self.args.source, img_size=imgsz, stride=stride)

        prev_t = time.time()
        while not self.stop_flag:
            if not self.running:
                time.sleep(0.1)
                continue

            for data in dataset:
                if self.stop_flag:
                    break
                if not self.running:
                    break

                t0 = time.time()
                _, im, im0s, _, _ = data
                self.frame_idx += 1
                frame = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()

                if self.c1_poly is None or self.c2_poly is None:
                    h, w = frame.shape[:2]
                    self.c1_poly = self._parse_roi(self.args.c1_roi, w, h)
                    self.c2_poly = self._parse_roi(self.args.c2_roi, w, h)
                    if self.c1_poly is None or self.c2_poly is None:
                        self.c1_poly, self.c2_poly = self._default_rois(w, h)
                        self.add_event("Using default ROI polygons")
                    else:
                        self.add_event("Loaded ROI polygons from CLI arguments")

                im_tensor = torch.from_numpy(im).to(device_obj).float() / 255.0
                if im_tensor.ndim == 3:
                    im_tensor = im_tensor[None]

                with torch.no_grad():
                    pred, proto = model(im_tensor, augment=False, visualize=False)[:2]
                    pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres, nm=32)

                detections = []
                if len(pred[0]):
                    pred[0][:, :4] = scale_boxes(im_tensor.shape[2:], pred[0][:, :4], frame.shape).round()
                    masks = process_mask(proto[0], pred[0][:, 6:], pred[0][:, :4], frame.shape[:2], upsample=True)
                    for i, (*xyxy, conf, cls_idx) in enumerate(pred[0][:, :6]):
                        x1, y1, x2, y2 = map(int, xyxy)
                        cls_name = names[int(cls_idx)]
                        mask = masks[i]
                        if isinstance(mask, torch.Tensor):
                            mask = mask.detach().cpu().numpy()
                        mask_bool = mask.astype(bool)
                        centroid = centroid_from_mask_or_box(mask_bool, (x1, y1, x2, y2))
                        detections.append(
                            {
                                "bbox": (x1, y1, x2, y2),
                                "cls_name": cls_name,
                                "mask": mask_bool,
                                "centroid": centroid,
                                "conf": float(conf.item()),
                            }
                        )

                c1_dets, c2_dets = [], []
                for d in detections:
                    if point_in_polygon(d["centroid"], self.c1_poly):
                        c1_dets.append(d)
                    if point_in_polygon(d["centroid"], self.c2_poly):
                        c2_dets.append(d)

                self.c1_tracker.update(c1_dets, self.frame_idx)
                self.c2_tracker.update(c2_dets, self.frame_idx)

                for tid, st in self.c1_tracker.tracks.items():
                    if st.last_seen == self.frame_idx and not st.counted and self.c1_tracker.mark_counted(tid):
                        self.csv_writer.writerow([datetime.now().isoformat(timespec="seconds"), "C1", tid])
                        self.csv_file.flush()
                        self.add_event(f"{tid} Counted")
                for tid, st in self.c2_tracker.tracks.items():
                    if st.last_seen == self.frame_idx and not st.counted and self.c2_tracker.mark_counted(tid):
                        self.csv_writer.writerow([datetime.now().isoformat(timespec="seconds"), "C2", tid])
                        self.csv_file.flush()
                        self.add_event(f"{tid} Counted")

                overlay = frame.copy()
                # for d in detections:
                #     color = get_class_color(d["cls_name"])
                #     overlay[d["mask"]] = (0.62 * overlay[d["mask"]] + 0.38 * np.array(color)).astype(np.uint8)
                # frame = overlay

                for d in detections:
                    x1, y1, x2, y2 = d["bbox"]
                    color = get_class_color(d["cls_name"])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, d["cls_name"], (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                for tid, st in self.c1_tracker.tracks.items():
                    if st.last_seen != self.frame_idx:
                        continue
                    cv2.circle(frame, st.centroid, 4, (0, 255, 255), -1)
                    cv2.putText(frame, tid, (st.bbox[0], max(20, st.bbox[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                for tid, st in self.c2_tracker.tracks.items():
                    if st.last_seen != self.frame_idx:
                        continue
                    cv2.circle(frame, st.centroid, 4, (255, 255, 0), -1)
                    cv2.putText(frame, tid, (st.bbox[0], max(20, st.bbox[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

                draw_polygon(frame, self.c1_poly, (0, 255, 255), "Conveyor 1 ROI")
                draw_polygon(frame, self.c2_poly, (255, 255, 0), "Conveyor 2 ROI")

                now_t = time.time()
                dt = max(now_t - prev_t, 1e-6)
                prev_t = now_t
                self.last_fps = 1.0 / dt
                self.inference_speed_ms = (time.time() - t0) * 1000.0
                self.total_detections = len(detections)

                cv2.putText(frame, f"FPS: {self.last_fps:.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Active IDs: {len(self.c1_tracker.tracks) + len(self.c2_tracker.tracks)}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Detections: {len(detections)}", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2, cv2.LINE_AA)

                _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                with self.lock:
                    self.last_frame = frame
                    self.last_jpeg = jpeg.tobytes()

            if not is_webcam:
                break

        self.csv_file.close()

    def get_stats(self):
        with self.lock:
            c1_total = int(sum(self.c1_tracker.count_by_class.values()))
            c2_total = int(sum(self.c2_tracker.count_by_class.values()))
            c1_active = [tid for tid, st in self.c1_tracker.tracks.items() if st.last_seen == self.frame_idx]
            c2_active = [tid for tid, st in self.c2_tracker.tracks.items() if st.last_seen == self.frame_idx]
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fps": round(self.last_fps, 2),
            "total_detections": self.total_detections,
            "active_ids_total": len(c1_active) + len(c2_active),
            "c1": {"total_count": c1_total, "active_ids": c1_active, "status": "RUNNING" if self.running else "PAUSED", "last_detection_time": self.c1_tracker.last_detection_ts},
            "c2": {"total_count": c2_total, "active_ids": c2_active, "status": "RUNNING" if self.running else "PAUSED", "last_detection_time": self.c2_tracker.last_detection_ts},
            "recording": self.running,
            "model_name": self.model_name,
            "gpu_device": self.device_name,
            "version": self.version,
        }

    def get_events(self):
        with self.lock:
            return list(self.event_logs)

    def get_health(self):
        cpu = psutil.cpu_percent(interval=None) if psutil else 0.0
        ram = psutil.virtual_memory().percent if psutil else 0.0
        gpu = 0.0
        if torch.cuda.is_available():
            try:
                gpu = float(torch.cuda.memory_allocated() / max(torch.cuda.max_memory_allocated(), 1) * 100.0)
            except Exception:
                gpu = 0.0
        return {
            "gpu_usage": round(gpu, 1),
            "cpu_usage": round(cpu, 1),
            "ram_usage": round(ram, 1),
            "camera_status": "ONLINE" if self.last_jpeg is not None else "WAITING",
            "inference_speed_ms": round(self.inference_speed_ms, 1),
        }

    def generate_mjpeg(self):
        while not self.stop_flag:
            frame = None
            with self.lock:
                frame = self.last_jpeg
            if frame is None:
                time.sleep(0.03)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    def toggle_running(self):
        self.running = not self.running
        self.add_event("Inference resumed" if self.running else "Inference paused")
        return self.running

    def snapshot(self):
        with self.lock:
            frame = None if self.last_frame is None else self.last_frame.copy()
        if frame is None:
            return None
        snap_path = self.snapshot_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(snap_path), frame)
        self.add_event(f"Snapshot saved: {snap_path.name}")
        return snap_path

    def shutdown(self):
        self.stop_flag = True
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)


def create_app(args):
    app = Flask(__name__, template_folder="templates", static_folder="static")
    engine = InferenceEngine(args)
    engine.start()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(engine.generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stats")
    def stats():
        return jsonify(engine.get_stats())

    @app.route("/events")
    def events():
        return jsonify({"events": engine.get_events()})

    @app.route("/health")
    def health():
        return jsonify(engine.get_health())

    @app.route("/toggle_inference", methods=["POST"])
    def toggle_inference():
        running = engine.toggle_running()
        return jsonify({"running": running})

    @app.route("/snapshot", methods=["POST"])
    def snapshot():
        snap_path = engine.snapshot()
        if not snap_path:
            return jsonify({"ok": False, "message": "No frame available yet"}), 400
        return jsonify({"ok": True, "path": str(snap_path)})

    @app.route("/export_csv")
    def export_csv():
        return send_file(engine.csv_path, as_attachment=True)

    @app.route("/shutdown", methods=["POST"])
    def shutdown():
        engine.shutdown()
        return jsonify({"ok": True})

    return app


def parse_opt():
    parser = argparse.ArgumentParser(description="Industrial Flask YOLOv5 Segmentation Dashboard")
    parser.add_argument("--weights", required=True, help="YOLOv5 segmentation weights")
    parser.add_argument("--source", required=True, help="Video path or webcam index")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="", help="cuda device id or cpu")
    parser.add_argument("--project", default="runs/seg-dashboard", help="Output directory")
    parser.add_argument("--name", default="exp", help="Run name inside project directory")
    parser.add_argument("--track-timeout", type=int, default=90)
    parser.add_argument("--match-dist", type=float, default=80.0)
    parser.add_argument(
        "--c1-roi",
        default="161,501;298,472;263,275;231,216",
        help="Conveyor1 ROI: x1,y1;x2,y2;x3,y3;x4,y4",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    options = parse_opt()
    app = create_app(options)
    app.run(host=options.host, port=options.port, threaded=True, debug=False)
