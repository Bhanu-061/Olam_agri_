import argparse
import atexit
import csv
import os
import signal
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pathlib
import torch
import ctypes
from flask import Flask, Response, jsonify, render_template, send_file
from PIL import ImageGrab

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
        self.video_dir = self.run_dir / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        suffix = "screen" if args.save_mode == "screen" else "inference"
        self.video_path = self.video_dir / f"{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        self.video_writer = None
        self.output_frame_size = None

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
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU is required, but CUDA is not available on this system.")
            if str(self.args.device).lower() == "cpu":
                raise RuntimeError("GPU-only mode: --device cpu is not allowed. Use --device 0 (or another CUDA device id).")

            device_obj = select_device(self.args.device)
            if "cpu" in str(device_obj).lower():
                raise RuntimeError(f"GPU-only mode: selected device is '{device_obj}'. Please set a valid CUDA device (for example, --device 0).")
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

                    if self.video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        if self.args.save_mode == "screen":
                            if self.args.screen_region is None:
                                time.sleep(max(0.0, float(self.args.selection_delay)))
                                self.args.screen_region = select_screen_region_interactive()
                            sw, sh = self.args.device_screen_size
                            self.output_frame_size = (int(sw), int(sh))
                            self.video_writer = cv2.VideoWriter(str(self.video_path), fourcc, self.args.save_fps, self.output_frame_size)
                        else:
                            h, w = frame.shape[:2]
                            self.output_frame_size = (int(w), int(h))
                            self.video_writer = cv2.VideoWriter(str(self.video_path), fourcc, self.args.save_fps, self.output_frame_size)
                        self.add_event(f"Video recording started: {self.video_path}")

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
                    draw_polygon(frame, self.c2_poly, (0, 165, 255), "Conveyor 2 ROI")

                    now_t = time.time()
                    dt = max(now_t - prev_t, 1e-6)
                    prev_t = now_t
                    self.last_fps = 1.0 / dt
                    self.inference_speed_ms = (time.time() - t0) * 1000.0
                    self.total_detections = len(detections)

                    cv2.putText(frame, f"FPS: {self.last_fps:.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Active IDs: {len(self.c1_tracker.tracks) + len(self.c2_tracker.tracks)}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Detections: {len(detections)}", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 255), 2, cv2.LINE_AA)
                    if self.video_writer is not None:
                        if self.args.save_mode == "screen":
                            x, y, w, h = self.args.screen_region
                            bbox = (x, y, x + w, y + h)
                            screen_img = ImageGrab.grab(bbox=bbox)
                            screen_frame = cv2.cvtColor(np.array(screen_img), cv2.COLOR_RGB2BGR)
                            if self.output_frame_size and (screen_frame.shape[1], screen_frame.shape[0]) != self.output_frame_size:
                                screen_frame = cv2.resize(screen_frame, self.output_frame_size, interpolation=cv2.INTER_AREA)
                            self.video_writer.write(screen_frame)
                        else:
                            self.video_writer.write(frame)

                    _, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    with self.lock:
                        self.last_frame = frame
                        self.last_jpeg = jpeg.tobytes()

                if not is_webcam:
                    break
        finally:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception:
                pass
            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception:
                    pass
            print(f"[OUTPUT] CSV saved at: {self.csv_path}")
            print(f"[OUTPUT] Video saved at: {self.video_path}")

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
    app.config["engine"] = engine

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
    parser.add_argument("--device", default="0", help="CUDA device id (GPU-only mode, e.g. 0)")
    parser.add_argument("--project", default="runs/seg-dashboard", help="Output directory")
    parser.add_argument("--name", default="exp", help="Run name inside project directory")
    parser.add_argument("--track-timeout", type=int, default=90)
    parser.add_argument("--match-dist", type=float, default=80.0)
    parser.add_argument(
        "--c1-roi",
        default="636,577;971,562;1006,700;649,706",
        help="Conveyor1 ROI: x1,y1;x2,y2;x3,y3;x4,y4",
    )
    parser.add_argument(
        "--c2-roi",
        default="1071,80;1090,320;1240,290;1237,80",
        help="Conveyor2 ROI: x1,y1;x2,y2;x3,y3;x4,y4",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--save-fps", type=float, default=20.0, help="Saved output video FPS")
    parser.add_argument("--save-mode", choices=["inference", "screen"], default="inference", help="Save inference output or selected screen region")
    parser.add_argument("--screen-region", default="", help="Screen region for screen mode: x,y,w,h")
    parser.add_argument("--select-screen-region", action="store_true", help="Select screen region interactively after localhost opens")
    parser.add_argument("--browser-open-delay", type=float, default=1.2, help="Delay in seconds before opening localhost in browser")
    parser.add_argument("--selection-delay", type=float, default=2.0, help="Delay in seconds before showing selection tool")
    return parser.parse_args()


def resolve_screen_region(options):
    if options.save_mode != "screen":
        return (0, 0, 0, 0)

    if options.select_screen_region:
        return None

    if options.screen_region:
        vals = [int(v.strip()) for v in options.screen_region.split(",")]
        if len(vals) != 4 or vals[2] <= 0 or vals[3] <= 0:
            raise ValueError("--screen-region must be x,y,w,h with w>0 and h>0")
        return tuple(vals)

    raise ValueError("Screen mode requires --select-screen-region or --screen-region x,y,w,h")


def select_screen_region_interactive():
    hwnd = None
    shown_state = 0
    if os.name == "nt":
        try:
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                shown_state = int(ctypes.windll.user32.IsWindowVisible(hwnd))
                if shown_state:
                    ctypes.windll.user32.ShowWindow(hwnd, 6)
                    time.sleep(0.4)
        except Exception:
            hwnd = None
            shown_state = 0

    full = ImageGrab.grab(all_screens=True)
    preview = cv2.cvtColor(np.array(full), cv2.COLOR_RGB2BGR)
    win = "Select screen region and press ENTER"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    roi = cv2.selectROI(win, preview, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)

    if os.name == "nt" and hwnd and shown_state:
        hwnd = None
        try:
            ctypes.windll.user32.ShowWindow(hwnd, 9)
        except Exception:
            pass

    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        raise ValueError("No valid screen region selected.")
    print(f"[OUTPUT] Selected screen region: x={x}, y={y}, w={w}, h={h}")
    return (x, y, w, h)


def get_device_screen_size():
    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        except Exception:
            pass
    return 1920, 1080


if __name__ == "__main__":
    options = parse_opt()
    options.screen_region = resolve_screen_region(options)
    options.device_screen_size = get_device_screen_size()
    app = create_app(options)
    engine = app.config["engine"]
    url = f"http://127.0.0.1:{options.port}/"

    def _graceful_exit(*_):
        engine.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)
    atexit.register(engine.shutdown)
    threading.Timer(max(0.0, float(options.browser_open_delay)), lambda: webbrowser.open(url)).start()
    try:
        app.run(host=options.host, port=options.port, threaded=True, debug=False)
    finally:
        engine.shutdown()




###### python client_video_dash_board_.py   --imgsz 640 --weights "C:\Users\admin\Downloads\olam_vid_2_seg_.pt" --conf-thres 0.40 --project "D:\bhanu\olam_agri\ai_infernces" --name "olam_agri_v1_" --source "D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\input_videos\olam_client_video_v2.mp4" --device 0 --save-mode screen --select-screen-region
