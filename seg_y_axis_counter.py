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
# BUFFER_PX = 50
# BUFFER_SECONDS = 10
# STOP_REQUESTED = False


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


# def get_zone(cx, line_x):
#     if cx < line_x - BUFFER_PX:
#         return "left"
#     elif cx > line_x + BUFFER_PX:
#         return "right"
#     else:
#         return "buffer"


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
#     name="exp"
# ):
#     raw_writer = None
#     ann_writer = None
#     frame_idx = 0

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

#         count_in = {v: 0 for v in names.values()}
#         count_out = {v: 0 for v in names.values()}
#         last_side = {}
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
#             line_x = LINE_X if 0 <= LINE_X < w else w // 2

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

#                 zone = get_zone(cx, line_x)
#                 prev = last_side.get(tid)
#                 last_t = last_time.get(tid, 0)

#                 if prev and zone != prev and (now - last_t) > BUFFER_SECONDS:
#                     if prev == "left" and zone in ["buffer", "right"]:
#                         count_in[cls_name] += 1
#                         last_time[tid] = now
#                         last_side[tid] = "right"
#                     elif prev == "right" and zone in ["buffer", "left"]:
#                         count_out[cls_name] += 1
#                         last_time[tid] = now
#                         last_side[tid] = "left"

#                 if zone in ["left", "right"]:
#                     last_side[tid] = zone

#                 color = get_class_color(cls_name)
#                 frame[mask] = frame[mask] * 0.5 + np.array(color) * 0.5

#                 cv2.putText(frame, f"{cls_name} ID:{tid}",
#                             (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             # Draw lines
#             cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), 2)
#             cv2.line(frame, (line_x - BUFFER_PX, 0), (line_x - BUFFER_PX, frame.shape[0]), (255, 215, 0), 1)
#             cv2.line(frame, (line_x + BUFFER_PX, 0), (line_x + BUFFER_PX, frame.shape[0]), (255, 215, 0), 1)

#             y = 40
#             for cls in count_in:
#                 draw_text_with_gold_box(
#                     frame,
#                     f"{cls} IN:{count_in[cls]} OUT:{count_out[cls]}",
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

#             cv2.imshow("YOLOv5 Seg Counting", frame)
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
#     return parser.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))































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
# BUFFER_PX = 10
# BUFFER_SECONDS = 10
# STOP_REQUESTED = False


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


# def get_zone(cx, line_x):
#     if cx < line_x - BUFFER_PX:
#         return "left"
#     elif cx > line_x + BUFFER_PX:
#         return "right"
#     else:
#         return "buffer"


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
#     line_x=LINE_X
# ):
#     raw_writer = None
#     ann_writer = None
#     frame_idx = 0
#     requested_line_x = int(line_x)
#     line_x_warned = False

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

#         count_in = {v: 0 for v in names.values()}
#         count_out = {v: 0 for v in names.values()}
#         last_side = {}
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
#             line_x = max(0, min(requested_line_x, w - 1))
#             if not line_x_warned and requested_line_x != line_x:
#                 print(f"Warning: --line-x {requested_line_x} is outside frame width {w}. Using {line_x}.")
#                 line_x_warned = True

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

#                 zone = get_zone(cx, line_x)
#                 prev = last_side.get(tid)
#                 last_t = last_time.get(tid, 0)

#                 if prev and zone != prev and (now - last_t) > BUFFER_SECONDS:
#                     if prev == "left" and zone in ["buffer", "right"]:
#                         count_in[cls_name] += 1
#                         last_time[tid] = now
#                         last_side[tid] = "right"
#                     elif prev == "right" and zone in ["buffer", "left"]:
#                         count_out[cls_name] += 1
#                         last_time[tid] = now
#                         last_side[tid] = "left"

#                 if zone in ["left", "right"]:
#                     last_side[tid] = zone

#                 color = get_class_color(cls_name)
#                 frame[mask] = frame[mask] * 0.5 + np.array(color) * 0.5

#                 cv2.putText(frame, f"{cls_name} ID:{tid}",
#                             (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             # Draw lines
#             cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), 2)
#             cv2.line(frame, (line_x - BUFFER_PX, 0), (line_x - BUFFER_PX, frame.shape[0]), (255, 215, 0), 1)
#             cv2.line(frame, (line_x + BUFFER_PX, 0), (line_x + BUFFER_PX, frame.shape[0]), (255, 215, 0), 1)

#             y = 40
#             for cls in count_in:
#                 draw_text_with_gold_box(
#                     frame,
#                     f"{cls} IN:{count_in[cls]} OUT:{count_out[cls]}",
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

#             cv2.imshow("YOLOv5 Seg Counting", frame)
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
#     parser.add_argument("--line-x", type=int, default=LINE_X, help="x position of vertical counting line")
#     return parser.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))





















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
# BUFFER_PX = 10
# BUFFER_SECONDS = 10
# STOP_REQUESTED = False


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


# def get_zone(coord, line_pos):
#     if coord < line_pos - BUFFER_PX:
#         return "left"
#     elif coord > line_pos + BUFFER_PX:
#         return "right"
#     else:
#         return "buffer"


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
#     line_x=LINE_X,
#     line_y=LINE_Y,
#     axis="y"
# ):
#     raw_writer = None
#     ann_writer = None
#     frame_idx = 0
#     axis = axis.lower()
#     if axis not in ("x", "y"):
#         raise ValueError(f"Invalid axis '{axis}'. Use 'x' or 'y'.")
#     requested_line_x = int(line_x)
#     requested_line_y = int(line_y)
#     line_x_warned = False
#     line_y_warned = False

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

#         count_in = {v: 0 for v in names.values()}
#         count_out = {v: 0 for v in names.values()}
#         last_side = {}
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
#             line_x = max(0, min(requested_line_x, w - 1))
#             line_y = max(0, min(requested_line_y, h - 1))
#             if not line_x_warned and requested_line_x != line_x:
#                 print(f"Warning: --line-x {requested_line_x} is outside frame width {w}. Using {line_x}.")
#                 line_x_warned = True
#             if not line_y_warned and requested_line_y != line_y:
#                 print(f"Warning: --line-y {requested_line_y} is outside frame height {h}. Using {line_y}.")
#                 line_y_warned = True

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
#                 coord = cx if axis == "x" else cy
#                 line_pos = line_x if axis == "x" else line_y

#                 zone = get_zone(coord, line_pos)
#                 prev = last_side.get(tid)
#                 last_t = last_time.get(tid, 0)

#                 if prev and zone != prev and (now - last_t) > BUFFER_SECONDS:
#                     if prev == "left" and zone in ["buffer", "right"]:
#                         count_in[cls_name] += 1
#                         last_time[tid] = now
#                         last_side[tid] = "right"
#                     elif prev == "right" and zone in ["buffer", "left"]:
#                         count_out[cls_name] += 1
#                         last_time[tid] = now
#                         last_side[tid] = "left"

#                 if zone in ["left", "right"]:
#                     last_side[tid] = zone

#                 color = get_class_color(cls_name)
#                 frame[mask] = frame[mask] * 0.5 + np.array(color) * 0.5

#                 cv2.putText(frame, f"{cls_name} ID:{tid}",
#                             (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             # Draw only one counting axis based on user selection
#             if axis == "x":
#                 cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), 2)
#                 cv2.line(frame, (line_x - BUFFER_PX, 0), (line_x - BUFFER_PX, frame.shape[0]), (255, 215, 0), 1)
#                 cv2.line(frame, (line_x + BUFFER_PX, 0), (line_x + BUFFER_PX, frame.shape[0]), (255, 215, 0), 1)
#             else:
#                 cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)
#                 cv2.line(frame, (0, line_y - BUFFER_PX), (frame.shape[1], line_y - BUFFER_PX), (255, 215, 0), 1)
#                 cv2.line(frame, (0, line_y + BUFFER_PX), (frame.shape[1], line_y + BUFFER_PX), (255, 215, 0), 1)

#             y = 40
#             for cls in count_in:
#                 draw_text_with_gold_box(
#                     frame,
#                     f"{cls} IN:{count_in[cls]} OUT:{count_out[cls]}",
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

#             cv2.imshow("YOLOv5 Seg Counting", frame)
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
#     parser.add_argument("--line-x", type=int, default=LINE_X, help="x position of vertical counting line")
#     parser.add_argument("--line-y", type=int, default=LINE_Y, help="y position of horizontal counting line")
#     return parser.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))






































    





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
LINE_P3_X = 263
LINE_P3_Y = 275
LINE_P4_X = 231
LINE_P4_Y = 216

### below are example points for a conveyor belt counting line olam_agri_vid_1
# LINE_P1_Y = 501
# LINE_P2_X = 298
# LINE_P2_Y = 472
# LINE_P3_X = 263
# LINE_P3_Y = 275
# LINE_P4_X = 231
# LINE_P4_Y = 216
BUFFER_PX = 10
BUFFER_SECONDS = 0.2
MIN_MATCH_IOU = 0.1
DEDUP_SECONDS = 1.0
DEDUP_DISTANCE_PX = 50
STOP_REQUESTED = False
LINE_POINTS = []
DRAW_SCALE = 1.0


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


def update_conveyor_out_count(
    cx, cy, tid, cls_name, line_pts, track_state, last_time, count_out, counted_ids,
    recent_events, dedup_seconds, dedup_distance_px
):
    if tid in counted_ids:
        return False

    x1, y1, x2, y2 = line_pts
    side_val = point_line_side(cx, cy, x1, y1, x2, y2)
    side_sign = 0 if abs(side_val) <= BUFFER_PX else (1 if side_val > 0 else -1)
    now = time.time()
    last_t = last_time.get(tid, 0)
    state = track_state.setdefault(tid, {"last_sign": 0})
    prev_sign = state["last_sign"]

    # Same rule for all conveyors: count once when side flips across the line.
    if prev_sign != 0 and side_sign != 0 and prev_sign != side_sign and (now - last_t) >= BUFFER_SECONDS:
        is_duplicate = False
        for ex, ey, et in recent_events:
            if (now - et) <= dedup_seconds:
                if ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5 <= dedup_distance_px:
                    is_duplicate = True
                    break

        if not is_duplicate:
            count_out[cls_name] += 1
            last_time[tid] = now
            counted_ids.add(tid)
            recent_events.append((cx, cy, now))
            recent_events[:] = [e for e in recent_events if (now - e[2]) <= dedup_seconds]
            return True

    if side_sign != 0:
        state["last_sign"] = side_sign
    return False


def to_norm_points(p1x, p1y, p2x, p2y, w, h):
    wx = max(1, w - 1)
    hy = max(1, h - 1)
    return (p1x / wx, p1y / hy, p2x / wx, p2y / hy)


def from_norm_points(norm_pts, w, h):
    wx = max(1, w - 1)
    hy = max(1, h - 1)
    n1x, n1y, n2x, n2y = norm_pts
    p1x = int(round(n1x * wx))
    p1y = int(round(n1y * hy))
    p2x = int(round(n2x * wx))
    p2y = int(round(n2y * hy))
    return p1x, p1y, p2x, p2y


def get_fit_scale(w, h, max_w=1280, max_h=720):
    return min(max_w / max(1, w), max_h / max(1, h), 1.0)


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
    global LINE_POINTS, DRAW_SCALE
    sx = int(round(x / max(DRAW_SCALE, 1e-6)))
    sy = int(round(y / max(DRAW_SCALE, 1e-6)))
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(LINE_POINTS) >= 4:
            LINE_POINTS = []
        LINE_POINTS.append((sx, sy))
    elif event == cv2.EVENT_RBUTTONDOWN and LINE_POINTS:
        LINE_POINTS.pop()


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
    draw_line=True,
    persistence_seconds=0.5,
    dedup_seconds=DEDUP_SECONDS,
    dedup_distance_px=DEDUP_DISTANCE_PX
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
    norm_c1 = None
    norm_c2 = None
    persistence_seconds = float(persistence_seconds)
    dedup_seconds = float(dedup_seconds)
    dedup_distance_px = float(dedup_distance_px)

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
        counted_ids_c1 = set()
        counted_ids_c2 = set()
        counted_ids_global = set()
        recent_events_c1 = []
        recent_events_c2 = []
        first_seen_time = {}
        track_class = {}

        for data in tqdm(dataset, desc="Segmentation Counting"):
            if STOP_REQUESTED:
                break

            path, im, im0s, vid_cap, _ = data
            frame_idx += 1

            raw = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
            frame = raw.copy()
            h, w = frame.shape[:2]
            if norm_c1 is not None and norm_c2 is not None:
                p1x, p1y, p2x, p2y = from_norm_points(norm_c1, w, h)
                p3x, p3y, p4x, p4y = from_norm_points(norm_c2, w, h)
            else:
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
                global LINE_POINTS, DRAW_SCALE
                print("Edit first-frame points: click P1,P2 for Conveyor 1 and P3,P4 for Conveyor 2.")
                LINE_POINTS = [(p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y)]
                cv2.namedWindow(window_name)
                cv2.setMouseCallback(window_name, _line_mouse_callback)
                DRAW_SCALE = get_fit_scale(w, h)
                pw, ph = int(w * DRAW_SCALE), int(h * DRAW_SCALE)

                while True:
                    preview = frame.copy()
                    if len(LINE_POINTS) >= 1:
                        cv2.circle(preview, LINE_POINTS[0], 5, (0, 255, 255), -1)
                        cv2.putText(preview, "P1", (LINE_POINTS[0][0] + 4, LINE_POINTS[0][1] - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                    if len(LINE_POINTS) >= 2:
                        cv2.circle(preview, LINE_POINTS[1], 5, (0, 255, 255), -1)
                        cv2.putText(preview, "P2", (LINE_POINTS[1][0] + 4, LINE_POINTS[1][1] - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                        cv2.line(preview, LINE_POINTS[0], LINE_POINTS[1], (0, 255, 255), 2)
                        cv2.putText(preview, "Conveyor 1", (LINE_POINTS[0][0], max(20, LINE_POINTS[0][1] - 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if len(LINE_POINTS) >= 3:
                        cv2.circle(preview, LINE_POINTS[2], 5, (255, 255, 0), -1)
                        cv2.putText(preview, "P3", (LINE_POINTS[2][0] + 4, LINE_POINTS[2][1] - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                    if len(LINE_POINTS) >= 4:
                        cv2.circle(preview, LINE_POINTS[3], 5, (255, 255, 0), -1)
                        cv2.putText(preview, "P4", (LINE_POINTS[3][0] + 4, LINE_POINTS[3][1] - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                        cv2.line(preview, LINE_POINTS[2], LINE_POINTS[3], (255, 255, 0), 2)
                        cv2.putText(preview, "Conveyor 2", (LINE_POINTS[2][0], max(20, LINE_POINTS[2][1] - 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(preview, "Click P1,P2,P3,P4 | Right-click undo | ENTER confirm", (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    preview_show = cv2.resize(preview, (pw, ph), interpolation=cv2.INTER_AREA) if DRAW_SCALE < 1.0 else preview
                    cv2.imshow(window_name, preview_show)
                    key = cv2.waitKey(20) & 0xFF
                    if key == 13 and len(LINE_POINTS) == 4:  # Enter
                        break
                    if key in [27, ord("q")]:
                        request_stop()
                        break

                if len(LINE_POINTS) == 4:
                    (p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y) = LINE_POINTS
                    print(f"Selected conveyor 1 points: P1=({p1x}, {p1y}), P2=({p2x}, {p2y})")
                    print(f"Selected conveyor 2 points: P3=({p3x}, {p3y}), P4=({p4x}, {p4y})")

                # Lock both conveyor lines from first-frame coordinates and scale to future resolutions.
                norm_c1 = to_norm_points(p1x, p1y, p2x, p2y, w, h)
                norm_c2 = to_norm_points(p3x, p3y, p4x, p4y, w, h)
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

            used_det_indices = set()
            for trk in tracks.astype(int):
                x1, y1, x2, y2, tid = trk

                # Find matching detection
                best_iou, best_idx, det = 0, -1, None
                for di, d in enumerate(detections):
                    if di in used_det_indices:
                        continue
                    xx1, yy1 = max(x1, d[0]), max(y1, d[1])
                    xx2, yy2 = min(x2, d[2]), min(y2, d[3])
                    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (d[2] - d[0]) * (d[3] - d[1])
                    iou = inter / (area1 + area2 - inter + 1e-6)
                    if iou > best_iou:
                        best_iou, best_idx, det = iou, di, d

                if det is None:
                    continue
                if best_iou < MIN_MATCH_IOU:
                    continue
                used_det_indices.add(best_idx)

                cls_name = names[det[5]]
                mask = det[6]
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                mask = mask.astype(bool)

                track_class.setdefault(tid, cls_name)
                if tid not in first_seen_time:
                    first_seen_time[tid] = time.time()

                # Mask centroid
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                cx = int(xs.mean())
                cy = int(ys.mean())
                if (time.time() - first_seen_time[tid]) >= persistence_seconds and tid not in counted_ids_global:
                    counted_c1 = update_conveyor_out_count(
                        cx, cy, tid, cls_name,
                        (p1x, p1y, p2x, p2y),
                        track_state_c1, last_time_c1, count_out_c1, counted_ids_c1,
                        recent_events_c1, dedup_seconds, dedup_distance_px
                    )
                    counted_c2 = False
                    if not counted_c1:
                        counted_c2 = update_conveyor_out_count(
                            cx, cy, tid, cls_name,
                            (p3x, p3y, p4x, p4y),
                            track_state_c2, last_time_c2, count_out_c2, counted_ids_c2,
                            recent_events_c2, dedup_seconds, dedup_distance_px
                        )
                    if counted_c1 or counted_c2:
                        counted_ids_global.add(tid)

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
            draw_text_with_gold_box(frame, "Conveyor 1", (15, y), (0, 255, 255))
            y += 34
            for cls in count_out_c1:
                draw_text_with_gold_box(
                    frame,
                    f"{cls} OUT:{count_out_c1[cls]}",
                    (15, y),
                    get_class_color(cls)
                )
                y += 32

            y += 10
            draw_text_with_gold_box(frame, "Conveyor 2", (15, y), (255, 255, 0))
            y += 34
            for cls in count_out_c2:
                draw_text_with_gold_box(
                    frame,
                    f"{cls} OUT:{count_out_c2[cls]}",
                    (15, y),
                    get_class_color(cls)
                )
                y += 32

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
    parser.add_argument("--persistence-seconds", type=float, default=0.5, help="minimum tracked time before a crossing can be counted")
    parser.add_argument("--dedup-seconds", type=float, default=DEDUP_SECONDS, help="suppress duplicate events within this time window")
    parser.add_argument("--dedup-distance-px", type=float, default=DEDUP_DISTANCE_PX, help="duplicate suppression radius in pixels")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))











































































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
# LINE_P3_X = 263
# LINE_P3_Y = 275
# LINE_P4_X = 231
# LINE_P4_Y = 216

# ### below are example points for a conveyor belt counting line olam_agri_vid_1
# # LINE_P1_Y = 501
# # LINE_P2_X = 298
# # LINE_P2_Y = 472
# # LINE_P3_X = 263
# # LINE_P3_Y = 275
# # LINE_P4_X = 231
# # LINE_P4_Y = 216
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


# def update_conveyor_out_count(cx, cy, tid, cls_name, line_pts, track_state, last_time, count_out, counted_ids):
#     if tid in counted_ids:
#         return

#     x1, y1, x2, y2 = line_pts
#     side_val = point_line_side(cx, cy, x1, y1, x2, y2)
#     side_sign = 0 if abs(side_val) <= BUFFER_PX else (1 if side_val > 0 else -1)
#     now = time.time()
#     last_t = last_time.get(tid, 0)
#     state = track_state.setdefault(tid, {"last_sign": 0})
#     prev_sign = state["last_sign"]

#     # Same rule for all conveyors: count once when side flips across the line.
#     if prev_sign != 0 and side_sign != 0 and prev_sign != side_sign and (now - last_t) >= BUFFER_SECONDS:
#         count_out[cls_name] += 1
#         last_time[tid] = now
#         counted_ids.add(tid)

#     if side_sign != 0:
#         state["last_sign"] = side_sign


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
#     line_p3_x=LINE_P3_X,
#     line_p3_y=LINE_P3_Y,
#     line_p4_x=LINE_P4_X,
#     line_p4_y=LINE_P4_Y,
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
#     p3x = int(line_p3_x)
#     p3y = int(line_p3_y)
#     p4x = int(line_p4_x)
#     p4y = int(line_p4_y)
#     line_warned = False
#     line_selected = False
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

#         count_out_c1 = {v: 0 for v in names.values()}
#         count_out_c2 = {v: 0 for v in names.values()}
#         track_state_c1 = {}
#         track_state_c2 = {}
#         last_time_c1 = {}
#         last_time_c2 = {}
#         counted_ids_c1 = set()
#         counted_ids_c2 = set()
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
#             p3x = max(0, min(p3x, w - 1))
#             p3y = max(0, min(p3y, h - 1))
#             p4x = max(0, min(p4x, w - 1))
#             p4y = max(0, min(p4y, h - 1))
#             if not line_warned and ((p1x, p1y) != (int(line_p1_x), int(line_p1_y)) or (p2x, p2y) != (int(line_p2_x), int(line_p2_y))):
#                 print(f"Warning: Line points clamped to frame bounds. Using ({p1x},{p1y})-({p2x},{p2y}).")
#                 line_warned = True

#             if not line_selected:
#                 global LINE_POINTS
#                 print(f"Conveyor 1 fixed points: P1=({p1x}, {p1y}), P2=({p2x}, {p2y})")
#                 LINE_POINTS = [(p3x, p3y), (p4x, p4y)]
#                 cv2.namedWindow(window_name)
#                 cv2.setMouseCallback(window_name, _line_mouse_callback)

#                 while True:
#                     preview = frame.copy()
#                     #cv2.line(preview, (p1x, p1y), (p2x, p2y), (0, 255, 255), 2)
#                     cv2.putText(preview, "Conveyor 1", (p1x, max(20, p1y - 10)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#                     if len(LINE_POINTS) >= 1:
#                         cv2.circle(preview, LINE_POINTS[0], 4, (255, 255, 0), -1)
#                     if len(LINE_POINTS) >= 2:
#                         #cv2.line(preview, LINE_POINTS[0], LINE_POINTS[1], (255, 255, 0), 2)
#                         cv2.putText(preview, "Conveyor 2", (LINE_POINTS[0][0], max(20, LINE_POINTS[0][1] - 10)),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#                     cv2.putText(preview, "Click P3 and P4 for Conveyor 2, ENTER to confirm", (15, 30),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#                     cv2.imshow(window_name, preview)
#                     key = cv2.waitKey(20) & 0xFF
#                     if key == 13 and len(LINE_POINTS) == 2:  # Enter
#                         break
#                     if key in [27, ord("q")]:
#                         request_stop()
#                         break

#                 if len(LINE_POINTS) == 2:
#                     (p3x, p3y), (p4x, p4y) = LINE_POINTS
#                     print(f"Selected conveyor 2 points: P3=({p3x}, {p3y}), P4=({p4x}, {p4y})")
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
#                 update_conveyor_out_count(
#                     cx, cy, tid, cls_name,
#                     (p1x, p1y, p2x, p2y),
#                     track_state_c1, last_time_c1, count_out_c1, counted_ids_c1
#                 )
#                 update_conveyor_out_count(
#                     cx, cy, tid, cls_name,
#                     (p3x, p3y, p4x, p4y),
#                     track_state_c2, last_time_c2, count_out_c2, counted_ids_c2
#                 )

#                 # Display-only: draw bbox only (no mask fill, no ID text).
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(frame, cls_name, (x1, max(20, y1 - 8)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

#             # Show both conveyor lines on display.
#             cv2.line(frame, (p1x, p1y), (p2x, p2y), (0, 255, 255), 2)
#             cv2.putText(frame, "Conveyor 1", (p1x, max(20, p1y - 8)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#             cv2.line(frame, (p3x, p3y), (p4x, p4y), (255, 255, 0), 2)
#             cv2.putText(frame, "Conveyor 2", (p3x, max(20, p3y - 8)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#             y = 40
#             for cls in count_out_c1:
#                 draw_text_with_gold_box(
#                     frame,
#                     f"{cls}",
#                     (15, y),
#                     get_class_color(cls)
#                 )
#                 y += 30
#                 draw_text_with_gold_box(
#                     frame,
#                     f"C1_OUT:{count_out_c1[cls]}",
#                     (15, y),
#                     get_class_color(cls)
#                 )
#                 y += 30
#                 draw_text_with_gold_box(
#                     frame,
#                     f"C2_OUT:{count_out_c2[cls]}",
#                     (15, y),
#                     get_class_color(cls)
#                 )
#                 y += 36

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
#     parser.add_argument("--line-p3-x", type=int, default=LINE_P3_X, help="line point 3 x (conveyor 2)")
#     parser.add_argument("--line-p3-y", type=int, default=LINE_P3_Y, help="line point 3 y (conveyor 2)")
#     parser.add_argument("--line-p4-x", type=int, default=LINE_P4_X, help="line point 4 x (conveyor 2)")
#     parser.add_argument("--line-p4-y", type=int, default=LINE_P4_Y, help="line point 4 y (conveyor 2)")
#     parser.add_argument("--draw-line", action="store_true", help="draw 2-point line with mouse on first frame")
#     return parser.parse_args()


# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))



