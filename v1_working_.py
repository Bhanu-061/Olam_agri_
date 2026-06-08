








































# import argparse
# import sys
# import time
# import traceback
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# import os
# import pathlib
# from tqdm import tqdm
# import signal

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
# from sort.sort import Sort

# # ================= CONFIG =================
# LINE_X = 1500
# LINE_Y = 300
# LINE_P1_X = 161
# LINE_P1_Y = 501
# LINE_P2_X = 298
# LINE_P2_Y = 472
# BUFFER_PX = 10
# BUFFER_SECONDS = 0.2
# STOP_REQUESTED = False
# LINE_POINTS = []


# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\n⚠ Exit requested — saving videos safely...")


# signal.signal(signal.SIGINT, request_stop)
# signal.signal(signal.SIGTERM, request_stop)

# # ================= UTILITIES =================
# def get_class_color(cls):
#     np.random.seed(abs(hash(cls)) % (2**32))
#     return tuple(int(c) for c in np.random.randint(40, 255, 3))


# def point_line_side(px, py, x1, y1, x2, y2):
#     # Cross product sign tells which side of directed line the point lies on.
#     return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


# def get_next_video_path(save_dir, prefix):
#     save_dir.mkdir(parents=True, exist_ok=True)
#     existing = list(save_dir.glob(f"{prefix}_*.mp4"))
#     if not existing:
#         return save_dir / f"{prefix}_0001.mp4"
#     nums = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
#     return save_dir / f"{prefix}_{max(nums) + 1:04d}.mp4"


# def draw_text_with_gold_box(img, text, pos, color):
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     scale = 0.7
#     thickness = 2
#     padding = 6

#     (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
#     x, y = pos
#     cv2.rectangle(img, (x - padding, y - h - padding),
#                   (x + w + padding, y + padding), (0, 0, 0), -1)
#     cv2.rectangle(img, (x - padding, y - h - padding),
#                   (x + w + padding, y + padding), (0, 215, 255), 2)
#     cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# def _line_mouse_callback(event, x, y, flags, param):
#     global LINE_POINTS
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if len(LINE_POINTS) >= 2:
#             LINE_POINTS = []
#         LINE_POINTS.append((x, y))


# # ==================================================
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
#     axis="y",
#     line_p1_x=LINE_P1_X,
#     line_p1_y=LINE_P1_Y,
#     line_p2_x=LINE_P2_X,
#     line_p2_y=LINE_P2_Y,
#     draw_line=True
# ):
#     raw_writer = None
#     ann_writer = None
#     frame_idx = 0
#     axis = axis.lower()
#     if axis not in ("x", "y"):
#         raise ValueError(f"Invalid axis '{axis}'. Use 'x' or 'y'.")
#     p1x = int(line_p1_x)
#     p1y = int(line_p1_y)
#     p2x = int(line_p2_x)
#     p2y = int(line_p2_y)
#     line_warned = False
#     line_selected = not bool(draw_line)
#     window_name = "YOLOv5 Seg Counting"

#     try:
#         if not os.path.exists(weights):
#             raise FileNotFoundError(f"Weights not found: {weights}")

#         is_webcam = source.isnumeric()
#         save_dir = Path(project) / name
#         raw_video = get_next_video_path(save_dir, "raw")
#         ann_video = get_next_video_path(save_dir, "annotated")

#         device = select_device(device)
#         model = DetectMultiBackend(weights, device=device)
#         stride, names = model.stride, model.names
#         imgsz = check_img_size(imgsz, s=stride)
#         model.warmup(imgsz=(1, 3, imgsz, imgsz))

#         dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#             if is_webcam else LoadImages(source, img_size=imgsz, stride=stride)

#         tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

#         count_out = {v: 0 for v in names.values()}
#         track_state = {}
#         last_time = {}
#         track_class = {}

#         for data in tqdm(dataset, desc="Segmentation Counting"):
#             if STOP_REQUESTED:
#                 break

#             path, im, im0s, vid_cap, _ = data
#             frame_idx += 1

#             raw = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
#             frame = raw.copy()
#             h, w = frame.shape[:2]
#             p1x = max(0, min(p1x, w - 1))
#             p1y = max(0, min(p1y, h - 1))
#             p2x = max(0, min(p2x, w - 1))
#             p2y = max(0, min(p2y, h - 1))
#             if not line_warned and ((p1x, p1y) != (int(line_p1_x), int(line_p1_y)) or (p2x, p2y) != (int(line_p2_x), int(line_p2_y))):
#                 print(f"Warning: Line points clamped to frame bounds. Using ({p1x},{p1y})-({p2x},{p2y}).")
#                 line_warned = True

#             if not line_selected:
#                 global LINE_POINTS
#                 LINE_POINTS = [(p1x, p1y), (p2x, p2y)]
#                 cv2.namedWindow(window_name)
#                 cv2.setMouseCallback(window_name, _line_mouse_callback)

#                 while True:
#                     preview = frame.copy()
#                     if len(LINE_POINTS) >= 1:
#                         cv2.circle(preview, LINE_POINTS[0], 4, (0, 255, 255), -1)
#                     if len(LINE_POINTS) >= 2:
#                         cv2.line(preview, LINE_POINTS[0], LINE_POINTS[1], (0, 255, 255), 2)
#                     cv2.putText(preview, "Click 2 points for line, ENTER to confirm", (15, 30),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#                     cv2.imshow(window_name, preview)
#                     key = cv2.waitKey(20) & 0xFF
#                     if key == 13 and len(LINE_POINTS) == 2:  # Enter
#                         break
#                     if key in [27, ord("q")]:
#                         request_stop()
#                         break

#                 if len(LINE_POINTS) == 2:
#                     (p1x, p1y), (p2x, p2y) = LINE_POINTS
#                     print(f"Selected line points: P1=({p1x}, {p1y}), P2=({p2x}, {p2y})")
#                 line_selected = True

#             im = torch.from_numpy(im).to(device).float() / 255.0
#             if im.ndim == 3:
#                 im = im[None]

#             # Seg models can return extra tensors depending on backend/version.
#             # We only need the first two outputs: predictions and mask protos.
#             pred, proto = model(im, augment=False, visualize=False)[:2]
#             pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

#             detections = []
#             masks = []

#             if len(pred[0]):
#                 pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], frame.shape).round()
#                 masks = process_mask(proto[0], pred[0][:, 6:], pred[0][:, :4], frame.shape[:2], upsample=True)

#                 for i, (*xyxy, conf, cls) in enumerate(pred[0][:, :6]):
#                     x1, y1, x2, y2 = map(int, xyxy)
#                     detections.append([x1, y1, x2, y2, conf.item(), int(cls), masks[i]])

#             tracks = tracker.update(
#                 np.array([d[:5] for d in detections]) if detections else np.empty((0, 5))
#             )

#             now = time.time()

#             for trk in tracks.astype(int):
#                 x1, y1, x2, y2, tid = trk

#                 # Find matching detection
#                 best_iou, det = 0, None
#                 for d in detections:
#                     xx1, yy1 = max(x1, d[0]), max(y1, d[1])
#                     xx2, yy2 = min(x2, d[2]), min(y2, d[3])
#                     inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
#                     area1 = (x2 - x1) * (y2 - y1)
#                     area2 = (d[2] - d[0]) * (d[3] - d[1])
#                     iou = inter / (area1 + area2 - inter + 1e-6)
#                     if iou > best_iou:
#                         best_iou, det = iou, d

#                 if det is None:
#                     continue

#                 cls_name = names[det[5]]
#                 mask = det[6]
#                 if isinstance(mask, torch.Tensor):
#                     mask = mask.detach().cpu().numpy()
#                 mask = mask.astype(bool)

#                 track_class.setdefault(tid, cls_name)

#                 # Mask centroid
#                 ys, xs = np.where(mask)
#                 if len(xs) == 0:
#                     continue
#                 cx = int(xs.mean())
#                 cy = int(ys.mean())
#                 side_val = point_line_side(cx, cy, p1x, p1y, p2x, p2y)
#                 side_sign = 0 if abs(side_val) <= BUFFER_PX else (1 if side_val > 0 else -1)
#                 last_t = last_time.get(tid, 0)
#                 state = track_state.setdefault(tid, {"last_sign": 0})
#                 prev_sign = state["last_sign"]

#                 # Count a crossing when the sign flips across the line.
#                 if prev_sign != 0 and side_sign != 0 and prev_sign != side_sign and (now - last_t) >= BUFFER_SECONDS:
#                     if prev_sign < 0 and side_sign > 0:
#                         count_out[cls_name] += 1
#                     last_time[tid] = now
    
#                 if side_sign != 0:
#                     state["last_sign"] = side_sign

#                 # Display-only: draw bbox only (no mask fill, no ID text).
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(frame, cls_name, (x1, max(20, y1 - 8)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

#             # Display-only: keep counting line hidden.

#             y = 40
#             for cls in count_out:
#                 draw_text_with_gold_box(
#                     frame,
#                     f"{cls} OUT:{count_out[cls]}",
#                     (15, y),
#                     get_class_color(cls)
#                 )
#                 y += 34

#             if raw_writer is None:
#                 h, w = frame.shape[:2]
#                 fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
#                 raw_writer = cv2.VideoWriter(str(raw_video),
#                                              cv2.VideoWriter_fourcc(*"mp4v"),
#                                              fps, (w, h))
#                 ann_writer = cv2.VideoWriter(str(ann_video),
#                                              cv2.VideoWriter_fourcc(*"mp4v"),
#                                              fps, (w, h))

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

#         cv2.destroyAllWindows()
#         print(f"\n✅ Raw video: {raw_video}")
#         print(f"✅ Annotated video: {ann_video}")
#         print(f"📊 Frames processed: {frame_idx}")


# # ================= CLI =================
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", required=True)
#     parser.add_argument("--source", required=True)
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--project", default="runs/seg-count")
#     parser.add_argument("--name", default="exp")
#     parser.add_argument("--axis", choices=["x", "y"], default="y", help="x=vertical line, y=horizontal line")
#     parser.add_argument("--line-p1-x", type=int, default=LINE_P1_X, help="line point 1 x")
#     parser.add_argument("--line-p1-y", type=int, default=LINE_P1_Y, help="line point 1 y")
#     parser.add_argument("--line-p2-x", type=int, default=LINE_P2_X, help="line point 2 x")
#     parser.add_argument("--line-p2-y", type=int, default=LINE_P2_Y, help="line point 2 y")
#     parser.add_argument("--draw-line", action="store_true", help="draw 2-point line with mouse on first frame")
#     return parser.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))





















########## v2 working


















import argparse
import sys
import time
import traceback
from pathlib import Path
import cv2
import torch
import numpy as np
import os
import pathlib
from tqdm import tqdm
import signal

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
from sort.sort import Sort

# ================= CONFIG =================
LINE_X = 1500
LINE_Y = 300
LINE_P1_X = 161
LINE_P1_Y = 501
LINE_P2_X = 298
LINE_P2_Y = 472
LINE_P3_X = 340
LINE_P3_Y = 500
LINE_P4_X = 520
LINE_P4_Y = 470
BUFFER_PX = 10
BUFFER_SECONDS = 0.2
STOP_REQUESTED = False
LINE_POINTS = []


def request_stop(sig=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠ Exit requested — saving videos safely...")


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)

# ================= UTILITIES =================
def get_class_color(cls):
    np.random.seed(abs(hash(cls)) % (2**32))
    return tuple(int(c) for c in np.random.randint(40, 255, 3))


def point_line_side(px, py, x1, y1, x2, y2):
    # Cross product sign tells which side of directed line the point lies on.
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def get_next_video_path(save_dir, prefix):
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = list(save_dir.glob(f"{prefix}_*.mp4"))
    if not existing:
        return save_dir / f"{prefix}_0001.mp4"
    nums = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
    return save_dir / f"{prefix}_{max(nums) + 1:04d}.mp4"


def draw_text_with_gold_box(img, text, pos, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    padding = 6

    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - padding, y - h - padding),
                  (x + w + padding, y + padding), (0, 0, 0), -1)
    cv2.rectangle(img, (x - padding, y - h - padding),
                  (x + w + padding, y + padding), (0, 215, 255), 2)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _line_mouse_callback(event, x, y, flags, param):
    global LINE_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(LINE_POINTS) >= 2:
            LINE_POINTS = []
        LINE_POINTS.append((x, y))


# ==================================================
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
    axis="y",
    line_p1_x=LINE_P1_X,
    line_p1_y=LINE_P1_Y,
    line_p2_x=LINE_P2_X,
    line_p2_y=LINE_P2_Y,
    line_p3_x=LINE_P3_X,
    line_p3_y=LINE_P3_Y,
    line_p4_x=LINE_P4_X,
    line_p4_y=LINE_P4_Y,
    draw_line=True
):
    raw_writer = None
    ann_writer = None
    frame_idx = 0
    axis = axis.lower()
    if axis not in ("x", "y"):
        raise ValueError(f"Invalid axis '{axis}'. Use 'x' or 'y'.")
    p1x = int(line_p1_x)
    p1y = int(line_p1_y)
    p2x = int(line_p2_x)
    p2y = int(line_p2_y)
    p3x = int(line_p3_x)
    p3y = int(line_p3_y)
    p4x = int(line_p4_x)
    p4y = int(line_p4_y)
    line_warned = False
    line_selected = False
    window_name = "YOLOv5 Seg Counting"

    try:
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Weights not found: {weights}")

        is_webcam = source.isnumeric()
        save_dir = Path(project) / name
        raw_video = get_next_video_path(save_dir, "raw")
        ann_video = get_next_video_path(save_dir, "annotated")

        device = select_device(device)
        model = DetectMultiBackend(weights, device=device)
        stride, names = model.stride, model.names
        imgsz = check_img_size(imgsz, s=stride)
        model.warmup(imgsz=(1, 3, imgsz, imgsz))

        dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
            if is_webcam else LoadImages(source, img_size=imgsz, stride=stride)

        tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

        count_out_c1 = {v: 0 for v in names.values()}
        count_out_c2 = {v: 0 for v in names.values()}
        track_state_c1 = {}
        track_state_c2 = {}
        last_time_c1 = {}
        last_time_c2 = {}
        track_class = {}

        for data in tqdm(dataset, desc="Segmentation Counting"):
            if STOP_REQUESTED:
                break

            path, im, im0s, vid_cap, _ = data
            frame_idx += 1

            raw = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
            frame = raw.copy()
            h, w = frame.shape[:2]
            p1x = max(0, min(p1x, w - 1))
            p1y = max(0, min(p1y, h - 1))
            p2x = max(0, min(p2x, w - 1))
            p2y = max(0, min(p2y, h - 1))
            p3x = max(0, min(p3x, w - 1))
            p3y = max(0, min(p3y, h - 1))
            p4x = max(0, min(p4x, w - 1))
            p4y = max(0, min(p4y, h - 1))
            if not line_warned and ((p1x, p1y) != (int(line_p1_x), int(line_p1_y)) or (p2x, p2y) != (int(line_p2_x), int(line_p2_y))):
                print(f"Warning: Line points clamped to frame bounds. Using ({p1x},{p1y})-({p2x},{p2y}).")
                line_warned = True

            if not line_selected:
                global LINE_POINTS
                print(f"Conveyor 1 fixed points: P1=({p1x}, {p1y}), P2=({p2x}, {p2y})")
                LINE_POINTS = [(p3x, p3y), (p4x, p4y)]
                cv2.namedWindow(window_name)
                cv2.setMouseCallback(window_name, _line_mouse_callback)

                while True:
                    preview = frame.copy()
                    #cv2.line(preview, (p1x, p1y), (p2x, p2y), (0, 255, 255), 2)
                    cv2.putText(preview, "Conveyor 1", (p1x, max(20, p1y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if len(LINE_POINTS) >= 1:
                        cv2.circle(preview, LINE_POINTS[0], 4, (255, 255, 0), -1)
                    if len(LINE_POINTS) >= 2:
                        #cv2.line(preview, LINE_POINTS[0], LINE_POINTS[1], (255, 255, 0), 2)
                        cv2.putText(preview, "Conveyor 2", (LINE_POINTS[0][0], max(20, LINE_POINTS[0][1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(preview, "Click P3 and P4 for Conveyor 2, ENTER to confirm", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow(window_name, preview)
                    key = cv2.waitKey(20) & 0xFF
                    if key == 13 and len(LINE_POINTS) == 2:  # Enter
                        break
                    if key in [27, ord("q")]:
                        request_stop()
                        break

                if len(LINE_POINTS) == 2:
                    (p3x, p3y), (p4x, p4y) = LINE_POINTS
                    print(f"Selected conveyor 2 points: P3=({p3x}, {p3y}), P4=({p4x}, {p4y})")
                line_selected = True

            im = torch.from_numpy(im).to(device).float() / 255.0
            if im.ndim == 3:
                im = im[None]

            # Seg models can return extra tensors depending on backend/version.
            # We only need the first two outputs: predictions and mask protos.
            pred, proto = model(im, augment=False, visualize=False)[:2]
            pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

            detections = []
            masks = []

            if len(pred[0]):
                pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], frame.shape).round()
                masks = process_mask(proto[0], pred[0][:, 6:], pred[0][:, :4], frame.shape[:2], upsample=True)

                for i, (*xyxy, conf, cls) in enumerate(pred[0][:, :6]):
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append([x1, y1, x2, y2, conf.item(), int(cls), masks[i]])

            tracks = tracker.update(
                np.array([d[:5] for d in detections]) if detections else np.empty((0, 5))
            )

            now = time.time()

            for trk in tracks.astype(int):
                x1, y1, x2, y2, tid = trk

                # Find matching detection
                best_iou, det = 0, None
                for d in detections:
                    xx1, yy1 = max(x1, d[0]), max(y1, d[1])
                    xx2, yy2 = min(x2, d[2]), min(y2, d[3])
                    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (d[2] - d[0]) * (d[3] - d[1])
                    iou = inter / (area1 + area2 - inter + 1e-6)
                    if iou > best_iou:
                        best_iou, det = iou, d

                if det is None:
                    continue

                cls_name = names[det[5]]
                mask = det[6]
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                mask = mask.astype(bool)

                track_class.setdefault(tid, cls_name)

                # Mask centroid
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                cx = int(xs.mean())
                cy = int(ys.mean())
                # Conveyor 1 crossing
                side_val_1 = point_line_side(cx, cy, p1x, p1y, p2x, p2y)
                side_sign_1 = 0 if abs(side_val_1) <= BUFFER_PX else (1 if side_val_1 > 0 else -1)
                last_t_1 = last_time_c1.get(tid, 0)
                state_1 = track_state_c1.setdefault(tid, {"last_sign": 0})
                prev_sign_1 = state_1["last_sign"]
                if prev_sign_1 != 0 and side_sign_1 != 0 and prev_sign_1 != side_sign_1 and (now - last_t_1) >= BUFFER_SECONDS:
                    if prev_sign_1 < 0 and side_sign_1 > 0:
                        count_out_c1[cls_name] += 1
                    last_time_c1[tid] = now
                if side_sign_1 != 0:
                    state_1["last_sign"] = side_sign_1

                # Conveyor 2 crossing
                side_val_2 = point_line_side(cx, cy, p3x, p3y, p4x, p4y)
                side_sign_2 = 0 if abs(side_val_2) <= BUFFER_PX else (1 if side_val_2 > 0 else -1)
                last_t_2 = last_time_c2.get(tid, 0)
                state_2 = track_state_c2.setdefault(tid, {"last_sign": 0})
                prev_sign_2 = state_2["last_sign"]
                if prev_sign_2 != 0 and side_sign_2 != 0 and prev_sign_2 != side_sign_2 and (now - last_t_2) >= BUFFER_SECONDS:
                    if prev_sign_2 < 0 and side_sign_2 > 0:
                        count_out_c2[cls_name] += 1
                    last_time_c2[tid] = now
                if side_sign_2 != 0:
                    state_2["last_sign"] = side_sign_2

                # Display-only: draw bbox only (no mask fill, no ID text).
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, cls_name, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Show both conveyor lines on display.
            cv2.line(frame, (p1x, p1y), (p2x, p2y), (0, 255, 255), 2)
            cv2.putText(frame, "Conveyor 1", (p1x, max(20, p1y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.line(frame, (p3x, p3y), (p4x, p4y), (255, 255, 0), 2)
            cv2.putText(frame, "Conveyor 2", (p3x, max(20, p3y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            y = 40
            for cls in count_out_c1:
                draw_text_with_gold_box(
                    frame,
                    f"{cls}",
                    (15, y),
                    get_class_color(cls)
                )
                y += 30
                draw_text_with_gold_box(
                    frame,
                    f"C1_OUT:{count_out_c1[cls]}",
                    (15, y),
                    get_class_color(cls)
                )
                y += 30
                draw_text_with_gold_box(
                    frame,
                    f"C2_OUT:{count_out_c2[cls]}",
                    (15, y),
                    get_class_color(cls)
                )
                y += 36

            if raw_writer is None:
                h, w = frame.shape[:2]
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
                raw_writer = cv2.VideoWriter(str(raw_video),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (w, h))
                ann_writer = cv2.VideoWriter(str(ann_video),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (w, h))

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

        cv2.destroyAllWindows()
        print(f"\n✅ Raw video: {raw_video}")
        print(f"✅ Annotated video: {ann_video}")
        print(f"📊 Frames processed: {frame_idx}")


# ================= CLI =================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/seg-count")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--axis", choices=["x", "y"], default="y", help="x=vertical line, y=horizontal line")
    parser.add_argument("--line-p1-x", type=int, default=LINE_P1_X, help="line point 1 x")
    parser.add_argument("--line-p1-y", type=int, default=LINE_P1_Y, help="line point 1 y")
    parser.add_argument("--line-p2-x", type=int, default=LINE_P2_X, help="line point 2 x")
    parser.add_argument("--line-p2-y", type=int, default=LINE_P2_Y, help="line point 2 y")
    parser.add_argument("--line-p3-x", type=int, default=LINE_P3_X, help="line point 3 x (conveyor 2)")
    parser.add_argument("--line-p3-y", type=int, default=LINE_P3_Y, help="line point 3 y (conveyor 2)")
    parser.add_argument("--line-p4-x", type=int, default=LINE_P4_X, help="line point 4 x (conveyor 2)")
    parser.add_argument("--line-p4-y", type=int, default=LINE_P4_Y, help="line point 4 y (conveyor 2)")
    parser.add_argument("--draw-line", action="store_true", help="draw 2-point line with mouse on first frame")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))





















