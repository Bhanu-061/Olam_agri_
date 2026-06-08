# import argparse
# import csv
# import os
# import signal
# import sys
# import time
# from dataclasses import dataclass, field
# from datetime import datetime
# from pathlib import Path

# import cv2
# import numpy as np
# import pathlib
# import torch
# from tqdm import tqdm

# # ================= WINDOWS PATH FIX =================
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # ================= ROOT =================
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# # ================= YOLOv5 SEG =================
# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import check_img_size, non_max_suppression, scale_boxes
# from utils.segment.general import process_mask
# from utils.torch_utils import select_device, smart_inference_mode


# STOP_REQUESTED = False
# ROI_POINTS = []


# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\nExit requested. Saving outputs...")


# signal.signal(signal.SIGINT, request_stop)
# signal.signal(signal.SIGTERM, request_stop)


# def get_class_color(cls_name: str) -> tuple:
#     np.random.seed(abs(hash(cls_name)) % (2 ** 32))
#     return tuple(int(c) for c in np.random.randint(40, 255, 3))


# def get_next_path(save_dir: Path, prefix: str, suffix: str) -> Path:
#     save_dir.mkdir(parents=True, exist_ok=True)
#     existing = list(save_dir.glob(f"{prefix}_*{suffix}"))
#     if not existing:
#         return save_dir / f"{prefix}_0001{suffix}"
#     nums = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
#     return save_dir / f"{prefix}_{max(nums) + 1:04d}{suffix}"


# def centroid_from_mask_or_box(mask_bool: np.ndarray, box: tuple) -> tuple:
#     ys, xs = np.where(mask_bool)
#     if len(xs) > 0:
#         return int(xs.mean()), int(ys.mean())
#     x1, y1, x2, y2 = box
#     return int((x1 + x2) / 2), int((y1 + y2) / 2)


# def point_in_polygon(point: tuple, polygon: np.ndarray) -> bool:
#     return cv2.pointPolygonTest(polygon, point, False) >= 0


# def draw_polygon(frame: np.ndarray, poly: np.ndarray, color: tuple, title: str):
#     cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
#     x, y = poly[0]
#     cv2.putText(frame, title, (int(x), max(20, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


# @dataclass
# class TrackState:
#     centroid: tuple
#     bbox: tuple
#     cls_name: str
#     last_seen: int
#     first_seen: int
#     counted: bool = False
#     counted_at: int = -1
#     missed: int = 0


# class ConveyorTracker:
#     def __init__(self, prefix: str, timeout_frames: int, match_dist_px: float):
#         self.prefix = prefix
#         self.timeout_frames = timeout_frames
#         self.match_dist_px = match_dist_px
#         self.next_id = 1
#         self.tracks = {}
#         self.count_by_class = {}
#         self.counted_ids = set()

#     def _new_id(self) -> str:
#         tid = f"{self.prefix}_{self.next_id}"
#         self.next_id += 1
#         return tid

#     def update(self, detections: list, frame_idx: int) -> list:
#         assigned_tracks = set()
#         assigned_dets = set()

#         # Greedy nearest-neighbor matching
#         pairs = []
#         for det_idx, det in enumerate(detections):
#             dcx, dcy = det["centroid"]
#             for tid, st in self.tracks.items():
#                 dist = np.hypot(dcx - st.centroid[0], dcy - st.centroid[1])
#                 if dist <= self.match_dist_px:
#                     pairs.append((dist, tid, det_idx))

#         pairs.sort(key=lambda x: x[0])
#         for _, tid, det_idx in pairs:
#             if tid in assigned_tracks or det_idx in assigned_dets:
#                 continue
#             det = detections[det_idx]
#             st = self.tracks[tid]
#             st.centroid = det["centroid"]
#             st.bbox = det["bbox"]
#             st.cls_name = det["cls_name"]
#             st.last_seen = frame_idx
#             st.missed = 0
#             assigned_tracks.add(tid)
#             assigned_dets.add(det_idx)

#         # Create tracks for unmatched detections
#         for det_idx, det in enumerate(detections):
#             if det_idx in assigned_dets:
#                 continue
#             tid = self._new_id()
#             self.tracks[tid] = TrackState(
#                 centroid=det["centroid"],
#                 bbox=det["bbox"],
#                 cls_name=det["cls_name"],
#                 first_seen=frame_idx,
#                 last_seen=frame_idx,
#             )

#         # Age unmatched tracks
#         for tid, st in list(self.tracks.items()):
#             if st.last_seen != frame_idx:
#                 st.missed += 1
#             if st.missed > self.timeout_frames:
#                 del self.tracks[tid]

#         return list(self.tracks.keys())

#     def mark_counted(self, tid: str):
#         if tid in self.counted_ids:
#             return False
#         st = self.tracks.get(tid)
#         if st is None:
#             return False
#         self.counted_ids.add(tid)
#         st.counted = True
#         self.count_by_class[st.cls_name] = self.count_by_class.get(st.cls_name, 0) + 1
#         return True



# def _roi_mouse_callback(event, x, y, flags, param):
#     global ROI_POINTS
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if len(ROI_POINTS) < 8:
#             ROI_POINTS.append((x, y))
#     elif event == cv2.EVENT_RBUTTONDOWN:
#         ROI_POINTS = []


# def select_two_rois(first_frame: np.ndarray, window_name: str) -> tuple:
#     global ROI_POINTS
#     ROI_POINTS = []

#     cv2.namedWindow(window_name)
#     cv2.setMouseCallback(window_name, _roi_mouse_callback)

#     while True:
#         preview = first_frame.copy()

#         for i, pt in enumerate(ROI_POINTS):
#             c = (0, 255, 255) if i < 4 else (255, 255, 0)
#             cv2.circle(preview, pt, 5, c, -1)

#         if len(ROI_POINTS) >= 4:
#             c1 = np.array(ROI_POINTS[:4], dtype=np.int32)
#             draw_polygon(preview, c1, (0, 255, 255), "Conveyor 1 ROI")
#         if len(ROI_POINTS) >= 8:
#             c2 = np.array(ROI_POINTS[4:8], dtype=np.int32)
#             draw_polygon(preview, c2, (255, 255, 0), "Conveyor 2 ROI")

#         cv2.putText(preview, "Left click: C1 4 points, then C2 4 points", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
#         cv2.putText(preview, "Right click: reset all | Enter: confirm", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
#         cv2.putText(preview, f"Points: {len(ROI_POINTS)}/8", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

#         cv2.imshow(window_name, preview)
#         key = cv2.waitKey(20) & 0xFF
#         if key == 13 and len(ROI_POINTS) == 8:
#             break
#         if key in [27, ord("q")]:
#             request_stop()
#             break

#     if len(ROI_POINTS) != 8:
#         raise RuntimeError("ROI selection cancelled. 8 points required.")

#     c1_poly = np.array(ROI_POINTS[:4], dtype=np.int32)
#     c2_poly = np.array(ROI_POINTS[4:8], dtype=np.int32)
#     return c1_poly, c2_poly


# @smart_inference_mode()
# def run(
#     weights,
#     source,
#     imgsz=640,
#     conf_thres=0.25,
#     iou_thres=0.45,
#     device="",
#     project="runs/seg-count",
#     name="exp",
#     track_timeout=30,
#     match_dist=80.0,
# ):
#     raw_writer = None
#     ann_writer = None
#     frame_idx = 0
#     window_name = "YOLOv5 Seg Multi-Conveyor ROI Counting"

#     csv_file = None
#     csv_writer = None

#     try:
#         if not os.path.exists(weights):
#             raise FileNotFoundError(f"Weights not found: {weights}")

#         is_webcam = source.isnumeric()
#         save_dir = Path(project) / name
#         raw_video = get_next_path(save_dir, "raw", ".mp4")
#         ann_video = get_next_path(save_dir, "annotated", ".mp4")
#         csv_path = get_next_path(save_dir, "count_log", ".csv")

#         csv_file = open(csv_path, "w", newline="", encoding="utf-8")
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow(["timestamp", "conveyor_id", "tracker_id"])

#         device_obj = select_device(device)
#         model = DetectMultiBackend(weights, device=device_obj)
#         stride, names = model.stride, model.names
#         imgsz = check_img_size(imgsz, s=stride)
#         model.warmup(imgsz=(1, 3, imgsz, imgsz))

#         dataset = LoadStreams(source, img_size=imgsz, stride=stride) if is_webcam else LoadImages(source, img_size=imgsz, stride=stride)

#         tracker_c1 = ConveyorTracker(prefix="C1", timeout_frames=track_timeout, match_dist_px=match_dist)
#         tracker_c2 = ConveyorTracker(prefix="C2", timeout_frames=track_timeout, match_dist_px=match_dist)

#         c1_poly, c2_poly = None, None

#         for data in tqdm(dataset, desc="ROI Seg Counting"):
#             if STOP_REQUESTED:
#                 break

#             path, im, im0s, vid_cap, _ = data
#             frame_idx += 1

#             raw = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
#             frame = raw.copy()

#             if c1_poly is None or c2_poly is None:
#                 c1_poly, c2_poly = select_two_rois(frame, window_name)
#                 print(f"Conveyor 1 ROI: {c1_poly.tolist()}")
#                 print(f"Conveyor 2 ROI: {c2_poly.tolist()}")

#             im_tensor = torch.from_numpy(im).to(device_obj).float() / 255.0
#             if im_tensor.ndim == 3:
#                 im_tensor = im_tensor[None]

#             pred, proto = model(im_tensor, augment=False, visualize=False)[:2]
#             pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

#             detections = []
#             if len(pred[0]):
#                 pred[0][:, :4] = scale_boxes(im_tensor.shape[2:], pred[0][:, :4], frame.shape).round()
#                 masks = process_mask(proto[0], pred[0][:, 6:], pred[0][:, :4], frame.shape[:2], upsample=True)

#                 for i, (*xyxy, conf, cls_idx) in enumerate(pred[0][:, :6]):
#                     x1, y1, x2, y2 = map(int, xyxy)
#                     cls_name = names[int(cls_idx)]
#                     mask = masks[i]
#                     if isinstance(mask, torch.Tensor):
#                         mask = mask.detach().cpu().numpy()
#                     mask_bool = mask.astype(bool)
#                     centroid = centroid_from_mask_or_box(mask_bool, (x1, y1, x2, y2))
#                     detections.append({
#                         "bbox": (x1, y1, x2, y2),
#                         "conf": float(conf.item()),
#                         "cls_name": cls_name,
#                         "mask": mask_bool,
#                         "centroid": centroid,
#                     })

#             # ROI split
#             c1_dets, c2_dets = [], []
#             for d in detections:
#                 pt = d["centroid"]
#                 in_c1 = point_in_polygon(pt, c1_poly)
#                 in_c2 = point_in_polygon(pt, c2_poly)
#                 if in_c1:
#                     c1_dets.append(d)
#                 if in_c2:
#                     c2_dets.append(d)

#             tracker_c1.update(c1_dets, frame_idx)
#             tracker_c2.update(c2_dets, frame_idx)

#             # Count each active track once and log event
#             for tid, st in tracker_c1.tracks.items():
#                 if st.last_seen == frame_idx and (not st.counted):
#                     if tracker_c1.mark_counted(tid):
#                         csv_writer.writerow([datetime.now().isoformat(timespec="seconds"), "C1", tid])

#             for tid, st in tracker_c2.tracks.items():
#                 if st.last_seen == frame_idx and (not st.counted):
#                     if tracker_c2.mark_counted(tid):
#                         csv_writer.writerow([datetime.now().isoformat(timespec="seconds"), "C2", tid])

#             # Draw masks + boxes from detections
#             overlay = frame.copy()
#             for d in detections:
#                 color = get_class_color(d["cls_name"])
#                 overlay[d["mask"]] = (0.65 * overlay[d["mask"]] + 0.35 * np.array(color)).astype(np.uint8)
#             frame = overlay

#             # Draw tracked objects C1
#             for tid, st in tracker_c1.tracks.items():
#                 if st.last_seen != frame_idx:
#                     continue
#                 x1, y1, x2, y2 = st.bbox
#                 color = get_class_color(st.cls_name)
#                 dwell = frame_idx - st.first_seen + 1
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.circle(frame, st.centroid, 4, color, -1)
#                 cv2.putText(frame, f"{st.cls_name} {tid} t={dwell}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

#             # Draw tracked objects C2
#             for tid, st in tracker_c2.tracks.items():
#                 if st.last_seen != frame_idx:
#                     continue
#                 x1, y1, x2, y2 = st.bbox
#                 color = get_class_color(st.cls_name)
#                 dwell = frame_idx - st.first_seen + 1
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.circle(frame, st.centroid, 4, color, -1)
#                 cv2.putText(frame, f"{st.cls_name} {tid} t={dwell}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

#             # Draw ROIs and counts
#             draw_polygon(frame, c1_poly, (0, 255, 255), "Conveyor 1 ROI")
#             draw_polygon(frame, c2_poly, (255, 255, 0), "Conveyor 2 ROI")

#             c1_total = sum(tracker_c1.count_by_class.values())
#             c2_total = sum(tracker_c2.count_by_class.values())
#             cv2.putText(frame, f"Conveyor 1 Count: {c1_total}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
#             cv2.putText(frame, f"Conveyor 2 Count: {c2_total}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

#             if raw_writer is None:
#                 h, w = frame.shape[:2]
#                 fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
#                 raw_writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
#                 ann_writer = cv2.VideoWriter(str(ann_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

#             raw_writer.write(raw)
#             ann_writer.write(frame)

#             cv2.imshow(window_name, frame)
#             if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
#                 request_stop()

#     finally:
#         if raw_writer:
#             raw_writer.release()
#         if ann_writer:
#             ann_writer.release()
#         if csv_file:
#             csv_file.close()
#         cv2.destroyAllWindows()

#         if 'raw_video' in locals():
#             print(f"Raw video: {raw_video}")
#         if 'ann_video' in locals():
#             print(f"Annotated video: {ann_video}")
#         if 'csv_path' in locals():
#             print(f"CSV log: {csv_path}")


# def parse_opt():
#     parser = argparse.ArgumentParser(description="YOLOv5 Segmentation multi-conveyor polygon ROI counter")
#     parser.add_argument("--weights", required=True, help="YOLOv5 segmentation weights (.pt)")
#     parser.add_argument("--source", required=True, help="Video path or webcam index")
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="", help="cuda device (e.g. 0) or cpu")
#     parser.add_argument("--project", default="runs/seg-count")
#     parser.add_argument("--name", default="exp")
#     parser.add_argument("--track-timeout", type=int, default=30, help="Frames to keep lost tracks alive")
#     parser.add_argument("--match-dist", type=float, default=80.0, help="Centroid distance threshold for track matching")
#     return parser.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))

























































import argparse
import csv
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pathlib
import torch
from tqdm import tqdm

# ================= WINDOWS PATH FIX =================
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# ================= ROOT =================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# ================= YOLOv5 SEG =================
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.segment.general import process_mask
from utils.torch_utils import select_device, smart_inference_mode


STOP_REQUESTED = False
ROI_POINTS = []


def request_stop(sig=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\nExit requested. Saving outputs...")


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)


def get_class_color(cls_name: str) -> tuple:
    np.random.seed(abs(hash(cls_name)) % (2 ** 32))
    return tuple(int(c) for c in np.random.randint(40, 255, 3))


def get_next_path(save_dir: Path, prefix: str, suffix: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = list(save_dir.glob(f"{prefix}_*{suffix}"))
    if not existing:
        return save_dir / f"{prefix}_0001{suffix}"
    nums = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
    return save_dir / f"{prefix}_{max(nums) + 1:04d}{suffix}"


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
    counted_at: int = -1
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

    def _new_id(self) -> str:
        tid = f"{self.prefix}_{self.next_id}"
        self.next_id += 1
        return tid

    def update(self, detections: list, frame_idx: int) -> list:
        assigned_tracks = set()
        assigned_dets = set()

        # Greedy nearest-neighbor matching
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

        # Create tracks for unmatched detections
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

        # Age unmatched tracks
        for tid, st in list(self.tracks.items()):
            if st.last_seen != frame_idx:
                st.missed += 1
            if st.missed > self.timeout_frames:
                del self.tracks[tid]

        return list(self.tracks.keys())

    def mark_counted(self, tid: str):
        if tid in self.counted_ids:
            return False
        st = self.tracks.get(tid)
        if st is None:
            return False
        self.counted_ids.add(tid)
        st.counted = True
        self.count_by_class[st.cls_name] = self.count_by_class.get(st.cls_name, 0) + 1
        return True



def _roi_mouse_callback(event, x, y, flags, param):
    global ROI_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ROI_POINTS) < 8:
            ROI_POINTS.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        ROI_POINTS = []


def select_two_rois(first_frame: np.ndarray, window_name: str) -> tuple:
    global ROI_POINTS
    ROI_POINTS = []

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _roi_mouse_callback)

    while True:
        preview = first_frame.copy()

        for i, pt in enumerate(ROI_POINTS):
            c = (0, 255, 255) if i < 4 else (255, 255, 0)
            cv2.circle(preview, pt, 5, c, -1)

        if len(ROI_POINTS) >= 4:
            c1 = np.array(ROI_POINTS[:4], dtype=np.int32)
            draw_polygon(preview, c1, (0, 255, 255), "Conveyor 1 ROI")
        if len(ROI_POINTS) >= 8:
            c2 = np.array(ROI_POINTS[4:8], dtype=np.int32)
            draw_polygon(preview, c2, (255, 255, 0), "Conveyor 2 ROI")

        cv2.putText(preview, "Left click: C1 4 points, then C2 4 points", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(preview, "Right click: reset all | Enter: confirm", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(preview, f"Points: {len(ROI_POINTS)}/8", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(ROI_POINTS) == 8:
            break
        if key in [27, ord("q")]:
            request_stop()
            break

    if len(ROI_POINTS) != 8:
        raise RuntimeError("ROI selection cancelled. 8 points required.")

    c1_poly = np.array(ROI_POINTS[:4], dtype=np.int32)
    c2_poly = np.array(ROI_POINTS[4:8], dtype=np.int32)
    return c1_poly, c2_poly


@smart_inference_mode()
def run(
    weights,
    source,
    imgsz=640,
    conf_thres=0.25,
    iou_thres=0.45,
    device="",
    project="runs/seg-count",
    name="exp",
    track_timeout=90,
    match_dist=80.0,
):
    raw_writer = None
    ann_writer = None
    frame_idx = 0
    window_name = "YOLOv5 Seg Multi-Conveyor ROI Counting"

    csv_file = None
    csv_writer = None

    try:
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Weights not found: {weights}")

        is_webcam = source.isnumeric()
        save_dir = Path(project) / name
        raw_video = get_next_path(save_dir, "raw", ".mp4")
        ann_video = get_next_path(save_dir, "annotated", ".mp4")
        csv_path = get_next_path(save_dir, "count_log", ".csv")

        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "conveyor_id", "tracker_id"])

        device_obj = select_device(device)
        model = DetectMultiBackend(weights, device=device_obj)
        stride, names = model.stride, model.names
        imgsz = check_img_size(imgsz, s=stride)
        model.warmup(imgsz=(1, 3, imgsz, imgsz))

        dataset = LoadStreams(source, img_size=imgsz, stride=stride) if is_webcam else LoadImages(source, img_size=imgsz, stride=stride)

        tracker_c1 = ConveyorTracker(prefix="C1", timeout_frames=track_timeout, match_dist_px=match_dist)
        tracker_c2 = ConveyorTracker(prefix="C2", timeout_frames=track_timeout, match_dist_px=match_dist)

        c1_poly, c2_poly = None, None

        for data in tqdm(dataset, desc="ROI Seg Counting"):
            if STOP_REQUESTED:
                break

            path, im, im0s, vid_cap, _ = data
            frame_idx += 1

            raw = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
            frame = raw.copy()

            if c1_poly is None or c2_poly is None:
                c1_poly, c2_poly = select_two_rois(frame, window_name)
                print(f"Conveyor 1 ROI: {c1_poly.tolist()}")
                print(f"Conveyor 2 ROI: {c2_poly.tolist()}")

            im_tensor = torch.from_numpy(im).to(device_obj).float() / 255.0
            if im_tensor.ndim == 3:
                im_tensor = im_tensor[None]

            pred, proto = model(im_tensor, augment=False, visualize=False)[:2]
            pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

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
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": float(conf.item()),
                        "cls_name": cls_name,
                        "mask": mask_bool,
                        "centroid": centroid,
                    })

            # ROI split
            c1_dets, c2_dets = [], []
            for d in detections:
                pt = d["centroid"]
                in_c1 = point_in_polygon(pt, c1_poly)
                in_c2 = point_in_polygon(pt, c2_poly)
                if in_c1:
                    c1_dets.append(d)
                if in_c2:
                    c2_dets.append(d)

            tracker_c1.update(c1_dets, frame_idx)
            tracker_c2.update(c2_dets, frame_idx)

            # Count each active track once and log event
            for tid, st in tracker_c1.tracks.items():
                if st.last_seen == frame_idx and (not st.counted):
                    if tracker_c1.mark_counted(tid):
                        csv_writer.writerow([datetime.now().isoformat(timespec="seconds"), "C1", tid])

            for tid, st in tracker_c2.tracks.items():
                if st.last_seen == frame_idx and (not st.counted):
                    if tracker_c2.mark_counted(tid):
                        csv_writer.writerow([datetime.now().isoformat(timespec="seconds"), "C2", tid])

            # Draw masks + detection bboxes
            # overlay = frame.copy()
            # for d in detections:
            #     color = get_class_color(d["cls_name"])
            #     overlay[d["mask"]] = (0.65 * overlay[d["mask"]] + 0.35 * np.array(color)).astype(np.uint8)
            # frame = overlay
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                color = get_class_color(d["cls_name"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            # Draw tracked objects C1
            for tid, st in tracker_c1.tracks.items():
                if st.last_seen != frame_idx:
                    continue
                x1, y1, x2, y2 = st.bbox
                color = get_class_color(st.cls_name)
                dwell = frame_idx - st.first_seen + 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, st.centroid, 4, color, -1)
                cv2.putText(frame, f"{st.cls_name} {tid} t={dwell}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Draw tracked objects C2
            for tid, st in tracker_c2.tracks.items():
                if st.last_seen != frame_idx:
                    continue
                x1, y1, x2, y2 = st.bbox
                color = get_class_color(st.cls_name)
                dwell = frame_idx - st.first_seen + 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, st.centroid, 4, color, -1)
                cv2.putText(frame, f"{st.cls_name} {tid} t={dwell}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Draw ROIs and counts
            draw_polygon(frame, c1_poly, (0, 255, 255), "Conveyor 1 ROI")
            draw_polygon(frame, c2_poly, (255, 255, 0), "Conveyor 2 ROI")

            c1_total = sum(tracker_c1.count_by_class.values())
            c2_total = sum(tracker_c2.count_by_class.values())
            cv2.putText(frame, f"Conveyor 1 Count: {c1_total}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Conveyor 2 Count: {c2_total}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            if raw_writer is None:
                h, w = frame.shape[:2]
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
                raw_writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                ann_writer = cv2.VideoWriter(str(ann_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            raw_writer.write(raw)
            ann_writer.write(frame)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                request_stop()

    finally:
        if raw_writer:
            raw_writer.release()
        if ann_writer:
            ann_writer.release()
        if csv_file:
            csv_file.close()
        cv2.destroyAllWindows()

        if 'raw_video' in locals():
            print(f"Raw video: {raw_video}")
        if 'ann_video' in locals():
            print(f"Annotated video: {ann_video}")
        if 'csv_path' in locals():
            print(f"CSV log: {csv_path}")


def parse_opt():
    parser = argparse.ArgumentParser(description="YOLOv5 Segmentation multi-conveyor polygon ROI counter")
    parser.add_argument("--weights", required=True, help="YOLOv5 segmentation weights (.pt)")
    parser.add_argument("--source", required=True, help="Video path or webcam index")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="", help="cuda device (e.g. 0) or cpu")
    parser.add_argument("--project", default="runs/seg-count")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--track-timeout", type=int, default=90, help="Frames to keep lost tracks alive")
    parser.add_argument("--match-dist", type=float, default=80.0, help="Centroid distance threshold for track matching")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
