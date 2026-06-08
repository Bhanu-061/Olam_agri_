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

# # ================= DEFAULT CONFIG =================
# LINE_P1_X = 426
# LINE_P1_Y = 978
# LINE_P2_X = 538
# LINE_P2_Y = 694
# LINE_P3_X = 794
# LINE_P3_Y = 537
# LINE_P4_X = 879
# LINE_P4_Y = 286

# # ------------------------------------------------------------------
# # BELT_BOUNDARY_Y  — the horizontal pixel row that separates
# #   Conveyor-1 zone (above, smaller Y) from Conveyor-2 zone (below,
# #   larger Y) in your camera view.
# #
# # HOW TO SET IT:
# #   Pause the video on the first frame, hover your mouse over the
# #   gap between the two conveyor belts and read the Y coordinate
# #   shown in the OpenCV window title or use a screenshot tool.
# #   Set --belt-boundary-y to that value.
# #
# # WHY IT MATTERS:
# #   Without this guard, a large sack whose bounding-box spans both
# #   belt zones has its mask centroid land on the "wrong" side and
# #   gets assigned to the wrong conveyor.  The zone guard ensures
# #   that a centroid physically above the boundary is ONLY ever
# #   tested against the C1 counting line, and vice-versa for C2.
# # ------------------------------------------------------------------
# BELT_BOUNDARY_Y = 650     # pixels — override with --belt-boundary-y

# BUFFER_PX        = 10      # pixels around line treated as "on the line"
# BUFFER_SECONDS   = 0.2     # min seconds between crossings for same tracker
# MIN_IOU_MATCH    = 0.15    # min IoU to accept SORT track <-> detection match
# CENTROID_HISTORY = 30      # centroid trail length per tracker

# STOP_REQUESTED = False
# LINE_POINTS    = []


# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\n⚠ Exit requested — saving videos safely...")


# signal.signal(signal.SIGINT,  request_stop)
# signal.signal(signal.SIGTERM, request_stop)


# # ================= UTILITIES =================

# def get_class_color(cls: str) -> tuple:
#     """Deterministic BGR colour per class name."""
#     np.random.seed(abs(hash(cls)) % (2 ** 32))
#     return tuple(int(c) for c in np.random.randint(40, 255, 3))


# def compute_iou(box_a: tuple, box_b: tuple) -> float:
#     """IoU between two (x1,y1,x2,y2) boxes."""
#     ax1, ay1, ax2, ay2 = box_a
#     bx1, by1, bx2, by2 = box_b
#     ix1, iy1 = max(ax1, bx1), max(ay1, by1)
#     ix2, iy2 = min(ax2, bx2), min(ay2, by2)
#     inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
#     area_a = (ax2 - ax1) * (ay2 - ay1)
#     area_b = (bx2 - bx1) * (by2 - by1)
#     return inter / (area_a + area_b - inter + 1e-6)


# def point_line_side(px: float, py: float,
#                     x1: float, y1: float,
#                     x2: float, y2: float) -> float:
#     """
#     Cross-product of (P2-P1) x (P-P1).
#     > 0  → left of directed line P1→P2
#     < 0  → right
#     ~ 0  → on the line
#     """
#     return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


# def draw_text_with_gold_box(img, text: str, pos: tuple, color: tuple):
#     """Text label with black fill and gold border."""
#     font, scale, thickness, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2, 6
#     (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
#     x, y = pos
#     cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
#     cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 215, 255), 2)
#     cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# def _line_mouse_callback(event, x, y, flags, param):
#     global LINE_POINTS
#     if event == cv2.EVENT_LBUTTONDOWN:
#         if len(LINE_POINTS) >= 2:
#             LINE_POINTS = []
#         LINE_POINTS.append((x, y))


# # ================= TRACKER REGISTRY =================

# class TrackerRegistry:
#     """
#     Single source of truth for every active tracker.

#     Per-tracker state dict:
#         cls_name   : str | None       — class label from first matched detection
#         prev_side  : {c1: int, c2: int} — last sign from point_line_side per conveyor
#         centroids  : list[(cx,cy)]    — recent centroid history (capped at CENTROID_HISTORY)
#         counted    : bool             — True once the object has been counted
#         conveyor   : None|"c1"|"c2"  — permanently assigned conveyor
#         display_id : None|str         — e.g. "C1_3" or "C2_7"
#         last_time  : float            — wall-clock of last crossing event

#     Key design decisions
#     --------------------
#     * Zone guard (belt_boundary_y):
#         Before testing ANY line-crossing, the centroid's Y position is compared
#         against belt_boundary_y.  Centroids above the boundary are ONLY tested
#         against the C1 line; centroids below are ONLY tested against the C2 line.
#         This prevents large objects whose bounding box spans both belt zones from
#         being assigned to the wrong conveyor due to centroid drift.

#     * One assignment, permanent:
#         Once a tracker is assigned to a conveyor, that assignment never changes.
#         The counted flag ensures it is never incremented a second time.

#     * Separate display-ID counters:
#         c1_counter and c2_counter increment independently, so IDs are always
#         "C1_1, C1_2 …" and "C2_1, C2_2 …" with no gaps or overlaps.
#     """

#     def __init__(self, centroid_history: int = CENTROID_HISTORY,
#                  belt_boundary_y: int = BELT_BOUNDARY_Y):
#         self._state: dict        = {}
#         self._c1_counter: int    = 0
#         self._c2_counter: int    = 0
#         self._centroid_history   = centroid_history
#         self._belt_boundary_y    = belt_boundary_y

#         self.count_c1: dict = {}   # cls_name -> int
#         self.count_c2: dict = {}   # cls_name -> int

#     # ------------------------------------------------------------------
#     def get(self, tid: int) -> dict:
#         """Return (creating if absent) the state dict for tracker `tid`."""
#         if tid not in self._state:
#             self._state[tid] = {
#                 "cls_name"  : None,
#                 "prev_side" : {"c1": 0, "c2": 0},
#                 "centroids" : [],
#                 "counted"   : False,
#                 "conveyor"  : None,
#                 "display_id": None,
#                 "last_time" : 0.0,
#             }
#         return self._state[tid]

#     # ------------------------------------------------------------------
#     def update_centroid(self, tid: int, cx: int, cy: int):
#         state = self.get(tid)
#         state["centroids"].append((cx, cy))
#         if len(state["centroids"]) > self._centroid_history:
#             state["centroids"].pop(0)

#     # ------------------------------------------------------------------
#     def try_count(self, tid: int, cx: int, cy: int,
#                   cls_name: str,
#                   line_pts_c1: tuple,
#                   line_pts_c2: tuple) -> bool:
#         """
#         Attempt to register a crossing for tracker `tid`.

#         Zone-guard logic
#         ----------------
#         cy < belt_boundary_y  →  centroid is in the C1 belt region
#                                   → only test C1 line, never C2
#         cy >= belt_boundary_y →  centroid is in the C2 belt region
#                                   → only test C2 line, never C1

#         This is the primary fix for the wrong-conveyor-assignment bug.
#         """
#         state = self.get(tid)
#         now   = time.time()

#         if state["counted"]:
#             return False

#         # Ensure per-class counters exist
#         if cls_name not in self.count_c1:
#             self.count_c1[cls_name] = 0
#             self.count_c2[cls_name] = 0

#         assigned = state["conveyor"]

#         # ---- Inner helper: test one conveyor line --------------------
#         def _check_conveyor(conv_key: str, line_pts: tuple) -> bool:
#             x1, y1, x2, y2 = line_pts
#             side_val  = point_line_side(cx, cy, x1, y1, x2, y2)
#             side_sign = (0 if abs(side_val) <= BUFFER_PX
#                          else (1 if side_val > 0 else -1))
#             prev_sign = state["prev_side"][conv_key]

#             crossed = (
#                 prev_sign != 0
#                 and side_sign != 0
#                 and prev_sign != side_sign
#                 and (now - state["last_time"]) >= BUFFER_SECONDS
#             )
#             if side_sign != 0:
#                 state["prev_side"][conv_key] = side_sign
#             return crossed

#         # ---- Zone determination (THE FIX) ----------------------------
#         # Use centroid Y to decide which belt this object is on.
#         # An object physically on C1 (top belt) has a SMALLER Y value.
#         in_c1_zone = (cy < self._belt_boundary_y)

#         # ---- Crossing check ------------------------------------------
#         if assigned is None:
#             # Object not yet assigned — use zone to restrict which line
#             # we test.  This prevents large cross-belt objects from being
#             # grabbed by the wrong conveyor's line-crossing event.
#             if in_c1_zone:
#                 if _check_conveyor("c1", line_pts_c1):
#                     self._register_count(state, tid, cls_name, "c1", now)
#                     return True
#             else:
#                 if _check_conveyor("c2", line_pts_c2):
#                     self._register_count(state, tid, cls_name, "c2", now)
#                     return True

#         elif assigned == "c1":
#             # Already on C1 — only recheck C1 (counted flag prevents double-count)
#             if _check_conveyor("c1", line_pts_c1):
#                 self._register_count(state, tid, cls_name, "c1", now)
#                 return True

#         elif assigned == "c2":
#             if _check_conveyor("c2", line_pts_c2):
#                 self._register_count(state, tid, cls_name, "c2", now)
#                 return True

#         return False

#     # ------------------------------------------------------------------
#     def _register_count(self, state: dict, tid: int,
#                         cls_name: str, conv_key: str, now: float):
#         """Lock in conveyor assignment, issue display ID, increment count."""
#         state["conveyor"]  = conv_key
#         state["counted"]   = True
#         state["last_time"] = now

#         if conv_key == "c1":
#             self._c1_counter += 1
#             state["display_id"] = f"C1_{self._c1_counter}"
#             self.count_c1[cls_name] += 1
#         else:
#             self._c2_counter += 1
#             state["display_id"] = f"C2_{self._c2_counter}"
#             self.count_c2[cls_name] += 1

#         print(f"  [COUNT] {cls_name} → {state['display_id']}"
#               f"  (C1_total:{self.count_c1[cls_name]}"
#               f"  C2_total:{self.count_c2[cls_name]})")


# # ================= MAIN RUN =================

# @smart_inference_mode()
# def run(
#     weights,
#     source,
#     imgsz           = 640,
#     conf_thres      = 0.25,
#     iou_thres       = 0.45,
#     device          = "",
#     project         = "runs/seg-count",
#     name            = "exp",
#     axis            = "y",
#     line_p1_x       = LINE_P1_X,
#     line_p1_y       = LINE_P1_Y,
#     line_p2_x       = LINE_P2_X,
#     line_p2_y       = LINE_P2_Y,
#     line_p3_x       = LINE_P3_X,
#     line_p3_y       = LINE_P3_Y,
#     line_p4_x       = LINE_P4_X,
#     line_p4_y       = LINE_P4_Y,
#     belt_boundary_y = BELT_BOUNDARY_Y,
#     draw_line       = True,
# ):
#     raw_writer    = None
#     ann_writer    = None
#     frame_idx     = 0
#     line_warned   = False
#     line_selected = False
#     window_name   = "YOLOv5 Seg Counting"

#     axis = axis.lower()
#     if axis not in ("x", "y"):
#         raise ValueError(f"Invalid axis '{axis}'. Use 'x' or 'y'.")

#     p1x, p1y = int(line_p1_x), int(line_p1_y)
#     p2x, p2y = int(line_p2_x), int(line_p2_y)
#     p3x, p3y = int(line_p3_x), int(line_p3_y)
#     p4x, p4y = int(line_p4_x), int(line_p4_y)
#     bby       = int(belt_boundary_y)   # short alias for the boundary Y

#     try:
#         if not os.path.exists(weights):
#             raise FileNotFoundError(f"Weights not found: {weights}")

#         # ---- Output paths --------------------------------------------
#         is_webcam = source.isnumeric()
#         save_dir  = Path(project) / name

#         def _next_video_path(prefix: str) -> Path:
#             save_dir.mkdir(parents=True, exist_ok=True)
#             existing = list(save_dir.glob(f"{prefix}_*.mp4"))
#             if not existing:
#                 return save_dir / f"{prefix}_0001.mp4"
#             nums = [int(p.stem.split("_")[-1])
#                     for p in existing if p.stem.split("_")[-1].isdigit()]
#             return save_dir / f"{prefix}_{max(nums) + 1:04d}.mp4"

#         raw_video = _next_video_path("raw")
#         ann_video = _next_video_path("annotated")

#         # ---- Model ---------------------------------------------------
#         device_obj = select_device(device)
#         model      = DetectMultiBackend(weights, device=device_obj)
#         stride, names = model.stride, model.names
#         imgsz      = check_img_size(imgsz, s=stride)
#         model.warmup(imgsz=(1, 3, imgsz, imgsz))

#         # ---- Dataset -------------------------------------------------
#         dataset = (LoadStreams(source, img_size=imgsz, stride=stride)
#                    if is_webcam
#                    else LoadImages(source, img_size=imgsz, stride=stride))

#         # ---- Tracker + registry ------------------------------------
#         tracker  = Sort(max_age=30, min_hits=2, iou_threshold=0.2)
#         registry = TrackerRegistry(
#             centroid_history=CENTROID_HISTORY,
#             belt_boundary_y=bby,
#         )

#         print(f"\n  Belt boundary Y = {bby}px")
#         print(f"  Objects with centroid Y < {bby} → Conveyor 1 zone")
#         print(f"  Objects with centroid Y >= {bby} → Conveyor 2 zone\n")

#         # ---- Main loop -----------------------------------------------
#         for data in tqdm(dataset, desc="Segmentation Counting"):
#             if STOP_REQUESTED:
#                 break

#             path, im, im0s, vid_cap, _ = data
#             frame_idx += 1

#             raw   = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
#             frame = raw.copy()
#             h, w  = frame.shape[:2]

#             # Clamp all line points to frame bounds
#             _clamp = lambda v, lo, hi: max(lo, min(v, hi))
#             p1x = _clamp(p1x, 0, w - 1);  p1y = _clamp(p1y, 0, h - 1)
#             p2x = _clamp(p2x, 0, w - 1);  p2y = _clamp(p2y, 0, h - 1)
#             p3x = _clamp(p3x, 0, w - 1);  p3y = _clamp(p3y, 0, h - 1)
#             p4x = _clamp(p4x, 0, w - 1);  p4y = _clamp(p4y, 0, h - 1)

#             if not line_warned and (
#                 (p1x, p1y) != (int(line_p1_x), int(line_p1_y)) or
#                 (p2x, p2y) != (int(line_p2_x), int(line_p2_y))
#             ):
#                 print(f"Warning: Line points clamped to frame bounds. "
#                       f"C1=({p1x},{p1y})-({p2x},{p2y})  "
#                       f"C2=({p3x},{p3y})-({p4x},{p4y})")
#                 line_warned = True

#             # ---- Interactive Conveyor-2 line selection (first frame) -
#             if not line_selected:
#                 global LINE_POINTS
#                 print(f"Conveyor 1 fixed: P1=({p1x},{p1y}), P2=({p2x},{p2y})")
#                 LINE_POINTS = [(p3x, p3y), (p4x, p4y)]
#                 cv2.namedWindow(window_name)
#                 cv2.setMouseCallback(window_name, _line_mouse_callback)

#                 while True:
#                     preview = frame.copy()
#                     cv2.line(preview, (p1x, p1y), (p2x, p2y), (0, 255, 255), 2)
#                     cv2.putText(preview, "Conveyor 1",
#                                 (p1x, max(20, p1y - 10)),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#                     # Draw belt boundary guide line
#                     cv2.line(preview, (0, bby), (w, bby), (180, 180, 180), 1)
#                     cv2.putText(preview, f"belt boundary Y={bby}",
#                                 (10, bby - 6),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
#                     if len(LINE_POINTS) >= 1:
#                         cv2.circle(preview, LINE_POINTS[0], 4, (255, 255, 0), -1)
#                     if len(LINE_POINTS) >= 2:
#                         cv2.line(preview, LINE_POINTS[0], LINE_POINTS[1],
#                                  (255, 255, 0), 2)
#                         cv2.putText(preview, "Conveyor 2",
#                                     (LINE_POINTS[0][0],
#                                      max(20, LINE_POINTS[0][1] - 10)),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#                     cv2.putText(preview,
#                                 "Click P3 & P4 for Conveyor 2, then ENTER to confirm",
#                                 (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
#                                 (255, 255, 0), 2)
#                     cv2.imshow(window_name, preview)
#                     key = cv2.waitKey(20) & 0xFF
#                     if key == 13 and len(LINE_POINTS) == 2:   # Enter
#                         break
#                     if key in [27, ord("q")]:
#                         request_stop()
#                         break

#                 if len(LINE_POINTS) == 2:
#                     (p3x, p3y), (p4x, p4y) = LINE_POINTS
#                     print(f"Conveyor 2 confirmed: P3=({p3x},{p3y}), "
#                           f"P4=({p4x},{p4y})")
#                 line_selected = True

#             # ---- Inference -------------------------------------------
#             im_tensor = torch.from_numpy(im).to(device_obj).float() / 255.0
#             if im_tensor.ndim == 3:
#                 im_tensor = im_tensor[None]

#             pred, proto = model(im_tensor, augment=False, visualize=False)[:2]
#             pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

#             # ---- Build detection list --------------------------------
#             detections = []   # [x1, y1, x2, y2, conf, cls_idx, mask]

#             if len(pred[0]):
#                 pred[0][:, :4] = scale_boxes(
#                     im_tensor.shape[2:], pred[0][:, :4], frame.shape
#                 ).round()
#                 masks_tensor = process_mask(
#                     proto[0], pred[0][:, 6:], pred[0][:, :4],
#                     frame.shape[:2], upsample=True
#                 )
#                 for i, (*xyxy, conf, cls) in enumerate(pred[0][:, :6]):
#                     x1, y1, x2, y2 = map(int, xyxy)
#                     detections.append(
#                         [x1, y1, x2, y2, conf.item(), int(cls), masks_tensor[i]]
#                     )

#             # ---- SORT update -----------------------------------------
#             sort_input = (
#                 np.array([d[:5] for d in detections])
#                 if detections else np.empty((0, 5))
#             )
#             tracks = tracker.update(sort_input)   # (N,5): x1,y1,x2,y2,tid

#             line_pts_c1 = (p1x, p1y, p2x, p2y)
#             line_pts_c2 = (p3x, p3y, p4x, p4y)

#             # ---- Per-track processing --------------------------------
#             for trk in tracks.astype(int):
#                 x1, y1, x2, y2, tid = trk
#                 tid = int(tid)

#                 # Best-matching detection by IoU
#                 best_iou, best_det = 0.0, None
#                 for d in detections:
#                     iou = compute_iou((x1, y1, x2, y2),
#                                       (d[0], d[1], d[2], d[3]))
#                     if iou > best_iou:
#                         best_iou, best_det = iou, d

#                 # Skip coasting tracks / identity swaps
#                 if best_det is None or best_iou < MIN_IOU_MATCH:
#                     continue

#                 cls_name = names[best_det[5]]
#                 mask     = best_det[6]
#                 if isinstance(mask, torch.Tensor):
#                     mask = mask.detach().cpu().numpy()
#                 mask = mask.astype(bool)

#                 # Store class on first encounter
#                 state = registry.get(tid)
#                 if state["cls_name"] is None:
#                     state["cls_name"] = cls_name

#                 # Mask centroid
#                 ys, xs = np.where(mask)
#                 if len(xs) == 0:
#                     continue
#                 cx = int(xs.mean())
#                 cy = int(ys.mean())

#                 registry.update_centroid(tid, cx, cy)

#                 # Zone-guarded crossing check (THE FIX)
#                 registry.try_count(
#                     tid, cx, cy, cls_name, line_pts_c1, line_pts_c2
#                 )

#                 # ---- Drawing -----------------------------------------
#                 state      = registry.get(tid)
#                 display_id = state["display_id"] or f"T{tid}"
#                 color      = get_class_color(cls_name)
#                 label      = f"{cls_name} {display_id}"

#                 # Bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 # Label
#                 cv2.putText(frame, label, (x1, max(20, y1 - 8)),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
#                             cv2.LINE_AA)
#                 # Centroid dot
#                 cv2.circle(frame, (cx, cy), 5, color, -1)
#                 # Centroid trail (last 10 points)
#                 trail = state["centroids"][-10:]
#                 for i in range(1, len(trail)):
#                     cv2.line(frame, trail[i - 1], trail[i], color, 1)

#                 # Show which zone the centroid is in (debug aid)
#                 zone_label = "C1-zone" if cy < bby else "C2-zone"
#                 cv2.putText(frame, zone_label, (cx + 6, cy - 6),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.45,
#                             (200, 200, 200), 1, cv2.LINE_AA)

#             # ---- Draw counting lines ---------------------------------
#             cv2.line(frame, (p1x, p1y), (p2x, p2y), (0, 255, 255), 2)
#             cv2.putText(frame, "Conveyor 1", (p1x, max(20, p1y - 8)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#             cv2.line(frame, (p3x, p3y), (p4x, p4y), (255, 255, 0), 2)
#             cv2.putText(frame, "Conveyor 2", (p3x, max(20, p3y - 8)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#             # Draw belt boundary line (thin, grey)
#             cv2.line(frame, (0, bby), (w, bby), (160, 160, 160), 1)
#             cv2.putText(frame, f"zone boundary Y={bby}",
#                         (10, bby - 6),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

#             # ---- HUD: per-class counts per conveyor ------------------
#             y_hud       = 40
#             all_classes = sorted(
#                 set(list(registry.count_c1.keys()) +
#                     list(registry.count_c2.keys()))
#             )
#             for cls in all_classes:
#                 c = get_class_color(cls)
#                 draw_text_with_gold_box(frame, cls,               (15, y_hud), c)
#                 y_hud += 28
#                 draw_text_with_gold_box(
#                     frame, f"C1_OUT: {registry.count_c1.get(cls, 0)}",
#                     (15, y_hud), c)
#                 y_hud += 28
#                 draw_text_with_gold_box(
#                     frame, f"C2_OUT: {registry.count_c2.get(cls, 0)}",
#                     (15, y_hud), c)
#                 y_hud += 36

#             # ---- Video writers ---------------------------------------
#             if raw_writer is None:
#                 h_fr, w_fr = frame.shape[:2]
#                 fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
#                 raw_writer = cv2.VideoWriter(
#                     str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"),
#                     fps, (w_fr, h_fr))
#                 ann_writer = cv2.VideoWriter(
#                     str(ann_video), cv2.VideoWriter_fourcc(*"mp4v"),
#                     fps, (w_fr, h_fr))

#             raw_writer.write(raw)
#             ann_writer.write(frame)

#             cv2.imshow(window_name, frame)
#             if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
#                 request_stop()

#     except Exception:
#         traceback.print_exc()

#     finally:
#         if raw_writer:
#             raw_writer.release()
#         if ann_writer:
#             ann_writer.release()
#         cv2.destroyAllWindows()

#         print(f"\n✅ Raw video      : {raw_video}")
#         print(f"✅ Annotated video: {ann_video}")
#         print(f"📊 Frames processed: {frame_idx}")

#         print("\n📦 Final Counts:")
#         if "registry" in dir():
#             all_cls = sorted(
#                 set(list(registry.count_c1.keys()) +
#                     list(registry.count_c2.keys()))
#             )
#             for cls in all_cls:
#                 print(f"  {cls:20s}  "
#                       f"C1: {registry.count_c1.get(cls, 0):4d}  "
#                       f"C2: {registry.count_c2.get(cls, 0):4d}")


# # ================= CLI =================
# def parse_opt():
#     parser = argparse.ArgumentParser(
#         description="YOLOv5-Seg + SORT dual-conveyor object counter"
#     )
#     parser.add_argument("--weights",          required=True,
#                         help="Path to YOLOv5-seg weights (.pt)")
#     parser.add_argument("--source",           required=True,
#                         help="Video file path or webcam index")
#     parser.add_argument("--imgsz",            type=int,   default=640)
#     parser.add_argument("--conf-thres",       type=float, default=0.25)
#     parser.add_argument("--iou-thres",        type=float, default=0.45)
#     parser.add_argument("--device",           default="",
#                         help="cuda device (e.g. 0) or cpu")
#     parser.add_argument("--project",          default="runs/seg-count")
#     parser.add_argument("--name",             default="exp")
#     parser.add_argument("--axis",             choices=["x", "y"], default="y")

#     # Conveyor 1 line
#     parser.add_argument("--line-p1-x",        type=int, default=LINE_P1_X)
#     parser.add_argument("--line-p1-y",        type=int, default=LINE_P1_Y)
#     parser.add_argument("--line-p2-x",        type=int, default=LINE_P2_X)
#     parser.add_argument("--line-p2-y",        type=int, default=LINE_P2_Y)

#     # Conveyor 2 line
#     parser.add_argument("--line-p3-x",        type=int, default=LINE_P3_X)
#     parser.add_argument("--line-p3-y",        type=int, default=LINE_P3_Y)
#     parser.add_argument("--line-p4-x",        type=int, default=LINE_P4_X)
#     parser.add_argument("--line-p4-y",        type=int, default=LINE_P4_Y)

#     # Zone guard — THE KEY FIX
#     parser.add_argument(
#         "--belt-boundary-y", type=int, default=BELT_BOUNDARY_Y,
#         help=(
#             "Pixel Y row separating Conveyor-1 zone (above, lower Y) from "
#             "Conveyor-2 zone (below, higher Y).  Set to the Y value of the "
#             "gap between the two conveyor belts in your camera view.  "
#             "This prevents large objects spanning both belts from being "
#             "assigned to the wrong conveyor."
#         )
#     )
#     parser.add_argument("--draw-line", action="store_true")
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

# ================= DEFAULT CONFIG =================
LINE_P1_X = 426
LINE_P1_Y = 978
LINE_P2_X = 538
LINE_P2_Y = 694
LINE_P3_X = 794
LINE_P3_Y = 537
LINE_P4_X = 879
LINE_P4_Y = 286

# ------------------------------------------------------------------
# BELT_BOUNDARY_Y  — the horizontal pixel row that separates
#   Conveyor-1 zone (above, smaller Y) from Conveyor-2 zone (below,
#   larger Y) in your camera view.
#
# HOW TO SET IT:
#   Pause the video on the first frame, hover your mouse over the
#   gap between the two conveyor belts and read the Y coordinate
#   shown in the OpenCV window title or use a screenshot tool.
#   Set --belt-boundary-y to that value.
#
# WHY IT MATTERS:
#   Without this guard, a large sack whose bounding-box spans both
#   belt zones has its mask centroid land on the "wrong" side and
#   gets assigned to the wrong conveyor.  The zone guard ensures
#   that a centroid physically above the boundary is ONLY ever
#   tested against the C1 counting line, and vice-versa for C2.
# ------------------------------------------------------------------
BELT_BOUNDARY_Y = 650      # pixels — override with --belt-boundary-y

BUFFER_PX        = 10      # pixels around line treated as "on the line"
BUFFER_SECONDS   = 0.2     # min seconds between crossings for same tracker
MIN_IOU_MATCH    = 0.15    # min IoU to accept SORT track <-> detection match
CENTROID_HISTORY = 30      # centroid trail length per tracker

STOP_REQUESTED = False
LINE_POINTS    = []


def request_stop(sig=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠ Exit requested — saving videos safely...")


signal.signal(signal.SIGINT,  request_stop)
signal.signal(signal.SIGTERM, request_stop)


# ================= UTILITIES =================

def get_class_color(cls: str) -> tuple:
    """Deterministic BGR colour per class name."""
    np.random.seed(abs(hash(cls)) % (2 ** 32))
    return tuple(int(c) for c in np.random.randint(40, 255, 3))


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """IoU between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def point_line_side(px: float, py: float,
                    x1: float, y1: float,
                    x2: float, y2: float) -> float:
    """
    Cross-product of (P2-P1) x (P-P1).
    > 0  → left of directed line P1→P2
    < 0  → right
    ~ 0  → on the line
    """
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def draw_text_with_gold_box(img, text: str, pos: tuple, color: tuple):
    """Large text label with black fill and gold border."""

    # ===== FONT SETTINGS =====
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.15          # Bigger text
    thickness = 3         # Thicker text

    # ===== SPACING SETTINGS =====
    pad_x = 18            # Left/right padding
    pad_y = 14            # Top/bottom padding
    text_gap = 8          # Space between text and border

    # ===== TEXT SIZE =====
    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    x, y = pos

    # ===== BOX COORDINATES =====
    top_left = (
        x - pad_x,
        y - th - pad_y - text_gap
    )

    bottom_right = (
        x + tw + pad_x,
        y + pad_y
    )

    # ===== BLACK FILLED BOX =====
    cv2.rectangle(
        img,
        top_left,
        bottom_right,
        (0, 0, 0),
        -1
    )

    # ===== GOLD BORDER =====
    cv2.rectangle(
        img,
        top_left,
        bottom_right,
        (0, 215, 255),
        4
    )

    # ===== DRAW TEXT =====
    cv2.putText(
        img,
        text,
        (x, y - text_gap),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )

def _line_mouse_callback(event, x, y, flags, param):
    global LINE_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(LINE_POINTS) >= 2:
            LINE_POINTS = []
        LINE_POINTS.append((x, y))


# ================= TRACKER REGISTRY =================

class TrackerRegistry:
    """
    Single source of truth for every active tracker.

    Per-tracker state dict:
        cls_name   : str | None       — class label from first matched detection
        prev_side  : {c1: int, c2: int} — last sign from point_line_side per conveyor
        centroids  : list[(cx,cy)]    — recent centroid history (capped at CENTROID_HISTORY)
        counted    : bool             — True once the object has been counted
        conveyor   : None|"c1"|"c2"  — permanently assigned conveyor
        display_id : None|str         — e.g. "C1_3" or "C2_7"
        last_time  : float            — wall-clock of last crossing event

    Key design decisions
    --------------------
    * Zone guard (belt_boundary_y):
        Before testing ANY line-crossing, the centroid's Y position is compared
        against belt_boundary_y.  Centroids above the boundary are ONLY tested
        against the C1 line; centroids below are ONLY tested against the C2 line.
        This prevents large objects whose bounding box spans both belt zones from
        being assigned to the wrong conveyor due to centroid drift.

    * One assignment, permanent:
        Once a tracker is assigned to a conveyor, that assignment never changes.
        The counted flag ensures it is never incremented a second time.

    * Separate display-ID counters:
        c1_counter and c2_counter increment independently, so IDs are always
        "C1_1, C1_2 …" and "C2_1, C2_2 …" with no gaps or overlaps.
    """

    def __init__(self, centroid_history: int = CENTROID_HISTORY,
                 belt_boundary_y: int = BELT_BOUNDARY_Y):
        self._state: dict        = {}
        self._c1_counter: int    = 0
        self._c2_counter: int    = 0
        self._centroid_history   = centroid_history
        self._belt_boundary_y    = belt_boundary_y

        self.count_c1: dict = {}   # cls_name -> int
        self.count_c2: dict = {}   # cls_name -> int

    # ------------------------------------------------------------------
    def get(self, tid: int) -> dict:
        """Return (creating if absent) the state dict for tracker `tid`."""
        if tid not in self._state:
            self._state[tid] = {
                "cls_name"  : None,
                "prev_side" : {"c1": 0, "c2": 0},
                "centroids" : [],
                "counted"   : False,
                "conveyor"  : None,
                "display_id": None,
                "last_time" : 0.0,
            }
        return self._state[tid]

    # ------------------------------------------------------------------
    def update_centroid(self, tid: int, cx: int, cy: int):
        state = self.get(tid)
        state["centroids"].append((cx, cy))
        if len(state["centroids"]) > self._centroid_history:
            state["centroids"].pop(0)

    # ------------------------------------------------------------------
    def try_count(self, tid: int, cx: int, cy: int,
                  cls_name: str,
                  line_pts_c1: tuple,
                  line_pts_c2: tuple) -> bool:
        """
        Attempt to register a crossing for tracker `tid`.

        Zone-guard logic
        ----------------
        cy < belt_boundary_y  →  centroid is in the C1 belt region
                                  → only test C1 line, never C2
        cy >= belt_boundary_y →  centroid is in the C2 belt region
                                  → only test C2 line, never C1

        This is the primary fix for the wrong-conveyor-assignment bug.
        """
        state = self.get(tid)
        now   = time.time()

        if state["counted"]:
            return False

        # Ensure per-class counters exist
        if cls_name not in self.count_c1:
            self.count_c1[cls_name] = 0
            self.count_c2[cls_name] = 0

        assigned = state["conveyor"]

        # ---- Inner helper: test one conveyor line --------------------
        def _check_conveyor(conv_key: str, line_pts: tuple) -> bool:
            x1, y1, x2, y2 = line_pts
            side_val  = point_line_side(cx, cy, x1, y1, x2, y2)
            side_sign = (0 if abs(side_val) <= BUFFER_PX
                         else (1 if side_val > 0 else -1))
            prev_sign = state["prev_side"][conv_key]

            crossed = (
                prev_sign != 0
                and side_sign != 0
                and prev_sign != side_sign
                and (now - state["last_time"]) >= BUFFER_SECONDS
            )
            if side_sign != 0:
                state["prev_side"][conv_key] = side_sign
            return crossed

        # ---- Zone determination (THE FIX) ----------------------------
        # Use centroid Y to decide which belt this object is on.
        # An object physically on C1 (top belt) has a SMALLER Y value.
        in_c1_zone = (cy < self._belt_boundary_y)

        # ---- Crossing check ------------------------------------------
        if assigned is None:
            # Object not yet assigned — use zone to restrict which line
            # we test.  This prevents large cross-belt objects from being
            # grabbed by the wrong conveyor's line-crossing event.
            if in_c1_zone:
                if _check_conveyor("c1", line_pts_c1):
                    self._register_count(state, tid, cls_name, "c1", now)
                    return True
            else:
                if _check_conveyor("c2", line_pts_c2):
                    self._register_count(state, tid, cls_name, "c2", now)
                    return True

        elif assigned == "c1":
            # Already on C1 — only recheck C1 (counted flag prevents double-count)
            if _check_conveyor("c1", line_pts_c1):
                self._register_count(state, tid, cls_name, "c1", now)
                return True

        elif assigned == "c2":
            if _check_conveyor("c2", line_pts_c2):
                self._register_count(state, tid, cls_name, "c2", now)
                return True

        return False

    # ------------------------------------------------------------------
    def _register_count(self, state: dict, tid: int,
                        cls_name: str, conv_key: str, now: float):
        """Lock in conveyor assignment, issue display ID, increment count."""
        state["conveyor"]  = conv_key
        state["counted"]   = True
        state["last_time"] = now

        if conv_key == "c1":
            self._c1_counter += 1
            state["display_id"] = f"C1_{self._c1_counter}"
            self.count_c1[cls_name] += 1
        else:
            self._c2_counter += 1
            state["display_id"] = f"C2_{self._c2_counter}"
            self.count_c2[cls_name] += 1

        print(f"  [COUNT] {cls_name} → {state['display_id']}"
              f"  (C1_total:{self.count_c1[cls_name]}"
              f"  C2_total:{self.count_c2[cls_name]})")


# ================= MAIN RUN =================

@smart_inference_mode()
def run(
    weights,
    source,
    imgsz           = 640,
    conf_thres      = 0.25,
    iou_thres       = 0.45,
    device          = "",
    project         = "runs/seg-count",
    name            = "exp",
    axis            = "y",
    line_p1_x       = LINE_P1_X,
    line_p1_y       = LINE_P1_Y,
    line_p2_x       = LINE_P2_X,
    line_p2_y       = LINE_P2_Y,
    line_p3_x       = LINE_P3_X,
    line_p3_y       = LINE_P3_Y,
    line_p4_x       = LINE_P4_X,
    line_p4_y       = LINE_P4_Y,
    belt_boundary_y = BELT_BOUNDARY_Y,
    draw_line       = True,
):
    raw_writer    = None
    ann_writer    = None
    frame_idx     = 0
    line_warned   = False
    line_selected = False
    window_name   = "YOLOv5 Seg Counting"

    axis = axis.lower()
    if axis not in ("x", "y"):
        raise ValueError(f"Invalid axis '{axis}'. Use 'x' or 'y'.")

    p1x, p1y = int(line_p1_x), int(line_p1_y)
    p2x, p2y = int(line_p2_x), int(line_p2_y)
    p3x, p3y = int(line_p3_x), int(line_p3_y)
    p4x, p4y = int(line_p4_x), int(line_p4_y)
    bby       = int(belt_boundary_y)   # short alias for the boundary Y

    try:
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Weights not found: {weights}")

        # ---- Output paths --------------------------------------------
        is_webcam = source.isnumeric()
        save_dir  = Path(project) / name

        def _next_video_path(prefix: str) -> Path:
            save_dir.mkdir(parents=True, exist_ok=True)
            existing = list(save_dir.glob(f"{prefix}_*.mp4"))
            if not existing:
                return save_dir / f"{prefix}_0001.mp4"
            nums = [int(p.stem.split("_")[-1])
                    for p in existing if p.stem.split("_")[-1].isdigit()]
            return save_dir / f"{prefix}_{max(nums) + 1:04d}.mp4"

        raw_video = _next_video_path("raw")
        ann_video = _next_video_path("annotated")

        # ---- Model ---------------------------------------------------
        device_obj = select_device(device)
        model      = DetectMultiBackend(weights, device=device_obj)
        stride, names = model.stride, model.names
        imgsz      = check_img_size(imgsz, s=stride)
        model.warmup(imgsz=(1, 3, imgsz, imgsz))

        # ---- Dataset -------------------------------------------------
        dataset = (LoadStreams(source, img_size=imgsz, stride=stride)
                   if is_webcam
                   else LoadImages(source, img_size=imgsz, stride=stride))

        # ---- Tracker + registry ------------------------------------
        tracker  = Sort(max_age=30, min_hits=2, iou_threshold=0.2)
        registry = TrackerRegistry(
            centroid_history=CENTROID_HISTORY,
            belt_boundary_y=bby,
        )

        print(f"\n  Belt boundary Y = {bby}px")
        print(f"  Objects with centroid Y < {bby} → Conveyor 1 zone")
        print(f"  Objects with centroid Y >= {bby} → Conveyor 2 zone\n")

        # ---- Main loop -----------------------------------------------
        for data in tqdm(dataset, desc="Segmentation Counting"):
            if STOP_REQUESTED:
                break

            path, im, im0s, vid_cap, _ = data
            frame_idx += 1

            raw   = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
            frame = raw.copy()
            h, w  = frame.shape[:2]

            # Clamp all line points to frame bounds
            _clamp = lambda v, lo, hi: max(lo, min(v, hi))
            p1x = _clamp(p1x, 0, w - 1);  p1y = _clamp(p1y, 0, h - 1)
            p2x = _clamp(p2x, 0, w - 1);  p2y = _clamp(p2y, 0, h - 1)
            p3x = _clamp(p3x, 0, w - 1);  p3y = _clamp(p3y, 0, h - 1)
            p4x = _clamp(p4x, 0, w - 1);  p4y = _clamp(p4y, 0, h - 1)

            if not line_warned and (
                (p1x, p1y) != (int(line_p1_x), int(line_p1_y)) or
                (p2x, p2y) != (int(line_p2_x), int(line_p2_y))
            ):
                print(f"Warning: Line points clamped to frame bounds. "
                      f"C1=({p1x},{p1y})-({p2x},{p2y})  "
                      f"C2=({p3x},{p3y})-({p4x},{p4y})")
                line_warned = True

            # ---- Interactive Conveyor-2 line selection (first frame) -
            if not line_selected:
                global LINE_POINTS
                print(f"Conveyor 1 fixed: P1=({p1x},{p1y}), P2=({p2x},{p2y})")
                LINE_POINTS = [(p3x, p3y), (p4x, p4y)]
                cv2.namedWindow(window_name)
                cv2.setMouseCallback(window_name, _line_mouse_callback)

                while True:
                    preview = frame.copy()
                    cv2.line(preview, (p1x, p1y), (p2x, p2y), (0, 255, 255), 2)
                    # cv2.putText(preview, "Conveyor 2", 
                    #             (p1x, max(20, p1y - 10)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    # Draw belt boundary guide line
                    cv2.line(preview, (0, bby), (w, bby), (180, 180, 180), 1)
                    cv2.putText(preview, f"belt boundary Y={bby}",
                                (10, bby - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                    if len(LINE_POINTS) >= 1:
                        cv2.circle(preview, LINE_POINTS[0], 4, (255, 255, 0), -1)
                    if len(LINE_POINTS) >= 2:
                        cv2.line(preview, LINE_POINTS[0], LINE_POINTS[1],
                                 (255, 255, 0), 2)
                        # cv2.putText(preview, "Conveyor 1",
                        #             (LINE_POINTS[0][0],
                        #              max(20, LINE_POINTS[0][1] - 10)),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(preview,
                                "Click P3 & P4 for Conveyor 2, then ENTER to confirm",
                                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (255, 255, 0), 2)
                    cv2.imshow(window_name, preview)
                    key = cv2.waitKey(20) & 0xFF
                    if key == 13 and len(LINE_POINTS) == 2:   # Enter
                        break
                    if key in [27, ord("q")]:
                        request_stop()
                        break

                if len(LINE_POINTS) == 2:
                    (p3x, p3y), (p4x, p4y) = LINE_POINTS
                    print(f"Conveyor 2 confirmed: P3=({p3x},{p3y}), "
                          f"P4=({p4x},{p4y})")
                line_selected = True

            # ---- Inference -------------------------------------------
            im_tensor = torch.from_numpy(im).to(device_obj).float() / 255.0
            if im_tensor.ndim == 3:
                im_tensor = im_tensor[None]

            pred, proto = model(im_tensor, augment=False, visualize=False)[:2]
            pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

            # ---- Build detection list --------------------------------
            detections = []   # [x1, y1, x2, y2, conf, cls_idx, mask]

            if len(pred[0]):
                pred[0][:, :4] = scale_boxes(
                    im_tensor.shape[2:], pred[0][:, :4], frame.shape
                ).round()
                masks_tensor = process_mask(
                    proto[0], pred[0][:, 6:], pred[0][:, :4],
                    frame.shape[:2], upsample=True
                )
                for i, (*xyxy, conf, cls) in enumerate(pred[0][:, :6]):
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append(
                        [x1, y1, x2, y2, conf.item(), int(cls), masks_tensor[i]]
                    )

            # ---- SORT update -----------------------------------------
            sort_input = (
                np.array([d[:5] for d in detections])
                if detections else np.empty((0, 5))
            )
            tracks = tracker.update(sort_input)   # (N,5): x1,y1,x2,y2,tid

            line_pts_c1 = (p1x, p1y, p2x, p2y)
            line_pts_c2 = (p3x, p3y, p4x, p4y)

            # ---- Per-track processing --------------------------------
            for trk in tracks.astype(int):
                x1, y1, x2, y2, tid = trk
                tid = int(tid)

                # Best-matching detection by IoU
                best_iou, best_det = 0.0, None
                for d in detections:
                    iou = compute_iou((x1, y1, x2, y2),
                                      (d[0], d[1], d[2], d[3]))
                    if iou > best_iou:
                        best_iou, best_det = iou, d

                # Skip coasting tracks / identity swaps
                if best_det is None or best_iou < MIN_IOU_MATCH:
                    continue

                cls_name = names[best_det[5]]
                mask     = best_det[6]
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                mask = mask.astype(bool)

                # Store class on first encounter
                state = registry.get(tid)
                if state["cls_name"] is None:
                    state["cls_name"] = cls_name

                # Mask centroid
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    continue
                cx = int(xs.mean())
                cy = int(ys.mean())

                registry.update_centroid(tid, cx, cy)

                # Zone-guarded crossing check (THE FIX)
                registry.try_count(
                    tid, cx, cy, cls_name, line_pts_c1, line_pts_c2
                )

                # ---- Drawing -----------------------------------------
                state      = registry.get(tid)
                display_id = state["display_id"] or f"T{tid}"
                color      = get_class_color(cls_name)
                label      = f"{cls_name} {display_id}"

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Label
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                            cv2.LINE_AA)
                # Centroid dot
                cv2.circle(frame, (cx, cy), 5, color, -1)
                # Centroid trail (last 10 points)
                trail = state["centroids"][-10:]
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i - 1], trail[i], color, 1)

                # Show which conveyor zone the centroid is in (debug aid)
                # cy < bby  → top belt    → Conveyor 1  (cyan,   matches C1 line)
                # cy >= bby → bottom belt → Conveyor 2  (yellow, matches C2 line)
                if cy < bby:
                    zone_label = "Conveyor 1"
                    zone_color = (0, 255, 255)    # cyan
                else:
                    zone_label = "Conveyor 2"
                    zone_color = (255, 255, 0)    # yellow
                cv2.putText(frame, zone_label, (cx + 6, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            zone_color, 1, cv2.LINE_AA)

            # ---- Draw counting lines ---------------------------------
            cv2.line(frame, (p1x, p1y), (p2x, p2y), (0, 255, 255), 2)
            cv2.putText(frame, "Conveyor 2", (p1x, max(20, p1y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.line(frame, (p3x, p3y), (p4x, p4y), (255, 255, 0), 2)
            cv2.putText(frame, "Conveyor 1", (p3x, max(20, p3y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # # Draw belt boundary line (thin, grey)
            # cv2.line(frame, (0, bby), (w, bby), (160, 160, 160), 1)
            # cv2.putText(frame, f"zone boundary Y={bby}",
            #             (10, bby - 6),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

            # ---- HUD: per-class counts per conveyor ------------------
            y_hud       = 40
            all_classes = sorted(
                set(list(registry.count_c1.keys()) +
                    list(registry.count_c2.keys()))
            )
            for cls in all_classes:
                c = get_class_color(cls)

                draw_text_with_gold_box(
                    frame,
                    cls,
                    (15, y_hud),
                    c
                )
                y_hud += 65   # Increased spacing

                draw_text_with_gold_box(
                    frame,
                    f"C1_OUT: {registry.count_c1.get(cls, 0)}",
                    (15, y_hud),
                    c
                )
                y_hud += 65   # Increased spacing

                draw_text_with_gold_box(
                    frame,
                    f"C2_OUT: {registry.count_c2.get(cls, 0)}",
                    (15, y_hud),
                    c
                )
                y_hud += 85   # Extra gap between classes

            # ---- Video writers ---------------------------------------
            if raw_writer is None:
                h_fr, w_fr = frame.shape[:2]
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
                raw_writer = cv2.VideoWriter(
                    str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (w_fr, h_fr))
                ann_writer = cv2.VideoWriter(
                    str(ann_video), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (w_fr, h_fr))

            raw_writer.write(raw)
            ann_writer.write(frame)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                request_stop()

    except Exception:
        traceback.print_exc()

    finally:
        if raw_writer:
            raw_writer.release()
        if ann_writer:
            ann_writer.release()
        cv2.destroyAllWindows()

        print(f"\n✅ Raw video      : {raw_video}")
        print(f"✅ Annotated video: {ann_video}")
        print(f"📊 Frames processed: {frame_idx}")

        print("\n📦 Final Counts:")
        if "registry" in dir():
            all_cls = sorted(
                set(list(registry.count_c1.keys()) +
                    list(registry.count_c2.keys()))
            )
            for cls in all_cls:
                print(f"  {cls:20s}  "
                      f"C1: {registry.count_c1.get(cls, 0):4d}  "
                      f"C2: {registry.count_c2.get(cls, 0):4d}")


# ================= CLI =================
def parse_opt():
    parser = argparse.ArgumentParser(
        description="YOLOv5-Seg + SORT dual-conveyor object counter"
    )
    parser.add_argument("--weights",          required=True,
                        help="Path to YOLOv5-seg weights (.pt)")
    parser.add_argument("--source",           required=True,
                        help="Video file path or webcam index")
    parser.add_argument("--imgsz",            type=int,   default=640)
    parser.add_argument("--conf-thres",       type=float, default=0.25)
    parser.add_argument("--iou-thres",        type=float, default=0.45)
    parser.add_argument("--device",           default="",
                        help="cuda device (e.g. 0) or cpu")
    parser.add_argument("--project",          default="runs/seg-count")
    parser.add_argument("--name",             default="exp")
    parser.add_argument("--axis",             choices=["x", "y"], default="y")

    # Conveyor 1 line
    parser.add_argument("--line-p1-x",        type=int, default=LINE_P1_X)
    parser.add_argument("--line-p1-y",        type=int, default=LINE_P1_Y)
    parser.add_argument("--line-p2-x",        type=int, default=LINE_P2_X)
    parser.add_argument("--line-p2-y",        type=int, default=LINE_P2_Y)

    # Conveyor 2 line
    parser.add_argument("--line-p3-x",        type=int, default=LINE_P3_X)
    parser.add_argument("--line-p3-y",        type=int, default=LINE_P3_Y)
    parser.add_argument("--line-p4-x",        type=int, default=LINE_P4_X)
    parser.add_argument("--line-p4-y",        type=int, default=LINE_P4_Y)

    # Zone guard — THE KEY FIX
    parser.add_argument(
        "--belt-boundary-y", type=int, default=BELT_BOUNDARY_Y,
        help=(
            "Pixel Y row separating Conveyor-1 zone (above, lower Y) from "
            "Conveyor-2 zone (below, higher Y).  Set to the Y value of the "
            "gap between the two conveyor belts in your camera view.  "
            "This prevents large objects spanning both belts from being "
            "assigned to the wrong conveyor."
        )
    )
    parser.add_argument("--draw-line", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))



####python video_3_olam_seg_2_cony_.py  --imgsz 640 --weights "C:\Users\admin\Downloads\olam_agri_video_3_v1.pt" --conf-thre 0.40 --project "D:\bhanu\olam_agri\ai_infernces" --name olam_agri_v1_ --source "D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\input_videos\olam_v2.mov" --draw-line