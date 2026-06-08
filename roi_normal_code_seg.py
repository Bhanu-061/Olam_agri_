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
from collections import deque

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
STOP_REQUESTED = False

# ---- ROI selection state (shared across mouse callbacks) ----
_roi_points: list = []
_roi_confirmed: bool = False


def request_stop(sig=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠ Exit requested — saving videos safely...")


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)


# ================= UTILITIES =================

def get_class_color(cls: str):
    np.random.seed(abs(hash(cls)) % (2 ** 32))
    return tuple(int(c) for c in np.random.randint(40, 255, 3))


def point_in_polygon(px: int, py: int, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, (float(px), float(py)), False) >= 0


def get_total_frames(source: str) -> int:
    """Return total frame count for a video file; -1 for webcam / unknown."""
    if source.isnumeric():
        return -1
    cap = cv2.VideoCapture(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else -1
    cap.release()
    return total if total > 0 else -1


def get_next_video_path(save_dir: Path, prefix: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = list(save_dir.glob(f"{prefix}_*.mp4"))
    if not existing:
        return save_dir / f"{prefix}_0001.mp4"
    nums = [int(p.stem.split("_")[-1]) for p in existing if p.stem.split("_")[-1].isdigit()]
    return save_dir / f"{prefix}_{max(nums) + 1:04d}.mp4"


def draw_text_with_gold_box(img, text: str, pos: tuple, color: tuple):

    font = cv2.FONT_HERSHEY_SIMPLEX

    # ==========================================
    # BIGGER TEXT
    # ==========================================
    scale = 1.3

    thickness = 3

    padding = 12

    # ==========================================
    # GET TEXT SIZE
    # ==========================================
    (w, h), _ = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    x, y = pos

    # ==========================================
    # BLACK BOX
    # ==========================================
    cv2.rectangle(
        img,
        (x - padding, y - h - padding),
        (x + w + padding, y + padding),
        (0, 0, 0),
        -1
    )

    # ==========================================
    # GOLD BORDER
    # ==========================================
    cv2.rectangle(
        img,
        (x - padding, y - h - padding),
        (x + w + padding, y + padding),
        (0, 215, 255),
        4
    )

    # ==========================================
    # TEXT
    # ==========================================
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_perf_overlay(img, infer_ms: float, postproc_ms: float,
                      avg_fps: float, frame_idx: int, total_frames: int):
    """
    Draws a performance HUD in the top-right corner:
      • Infer   : XX.X ms
      • PostProc: XX.X ms
      • FPS     : XX.X
      • Frame   : XXXX / XXXX  (or  XXXX  for webcam)
    """
    lines = [
        f"Infer   : {infer_ms:6.1f} ms",
        f"PostProc: {postproc_ms:6.1f} ms",
        f"FPS     : {avg_fps:6.1f}",
        f"Frame   : {frame_idx}" + (f" / {total_frames}" if total_frames > 0 else ""),
    ]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 0.58
    thick  = 1
    pad    = 6
    line_h = 22
    h_img, w_img = img.shape[:2]

    # Measure widest line
    max_w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines)
    box_w = max_w + pad * 2
    box_h = line_h * len(lines) + pad * 2
    x0 = w_img - box_w - 10
    y0 = 10

    # Dark semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # Border
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (0, 215, 255), 1)

    # Text lines
    for i, line in enumerate(lines):
        ty = y0 + pad + line_h * (i + 1) - 4
        cv2.putText(img, line, (x0 + pad, ty), font, scale, (0, 255, 200), thick, cv2.LINE_AA)


# ---- Mouse callback used during ROI selection ----
def _roi_mouse_callback(event, x, y, flags, param):
    global _roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        _roi_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if _roi_points:
            _roi_points.pop()


def select_roi_polygon(frame: np.ndarray, window_name: str,
                       label: str, color: tuple) -> np.ndarray:
    """Interactive polygon-ROI picker on a single frame."""
    global _roi_points
    _roi_points = []

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, _roi_mouse_callback)

    instructions = [
        f"Drawing ROI for: {label}",
        "Left-click: add point  |  Right-click: undo last",
        "Enter: confirm (>=3 pts)  |  R: reset  |  Q/Esc: skip",
    ]

    while True:
        preview = frame.copy()
        pts = _roi_points

        for pt in pts:
            cv2.circle(preview, pt, 5, color, -1)
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(preview, pts[i], pts[i + 1], color, 2)
        if len(pts) >= 3:
            cv2.line(preview, pts[-1], pts[0], color, 1)
            overlay = preview.copy()
            cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
            cv2.addWeighted(overlay, 0.25, preview, 0.75, 0, preview)

        for i, txt in enumerate(instructions):
            cv2.putText(preview, txt, (15, 30 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, f"Points selected: {len(pts)}",
                    (15, 30 + len(instructions) * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF

        if key == 13 and len(pts) >= 3:
            break
        elif key in [ord("r"), ord("R")]:
            _roi_points = []
        elif key in [27, ord("q"), ord("Q")]:
            print(f"  ROI for '{label}' skipped.")
            return np.empty((0, 1, 2), dtype=np.int32)

    poly = np.array(_roi_points, dtype=np.int32).reshape((-1, 1, 2))
    print(f"  ROI '{label}' confirmed with {len(_roi_points)} points.")
    return poly


# ================= COUNTING HELPER =================

def update_roi_count(cx: int, cy: int, tid: int, cls_name: str,
                     roi_poly: np.ndarray, count_dict: dict, counted_ids: set):
    if tid in counted_ids:
        return
    if roi_poly is None or len(roi_poly) == 0:
        return
    if point_in_polygon(cx, cy, roi_poly):
        count_dict[cls_name] = count_dict.get(cls_name, 0) + 1
        counted_ids.add(tid)


# ==================================================
@smart_inference_mode()
def run(
    weights: str,
    source: str,
    imgsz: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = "",
    project: str = "runs/seg-count",
    name: str = "exp",
):
    raw_writer = None
    ann_writer = None
    frame_idx  = 0
    window_name = "YOLOv5 Seg ROI Counting"

    # ---- Timing accumulators (rolling window of 30 frames) ----
    infer_times    = deque(maxlen=30)   # model forward pass  (ms)
    postproc_times = deque(maxlen=30)   # NMS + mask + track  (ms)
    frame_times    = deque(maxlen=30)   # total frame wall-time (s)

    # Last recorded values for overlay
    last_infer_ms    = 0.0
    last_postproc_ms = 0.0
    avg_fps          = 0.0

    try:
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Weights not found: {weights}")

        is_webcam   = source.isnumeric()
        total_frames = get_total_frames(source)   # -1 for webcam
        save_dir     = Path(project) / name
        raw_video    = get_next_video_path(save_dir, "raw")
        ann_video    = get_next_video_path(save_dir, "annotated")

        device_obj = select_device(device)
        model      = DetectMultiBackend(weights, device=device_obj)
        stride, names = model.stride, model.names
        imgsz = check_img_size(imgsz, s=stride)
        model.warmup(imgsz=(1, 3, imgsz, imgsz))

        dataset = (
            LoadStreams(source, img_size=imgsz, stride=stride)
            if is_webcam
            else LoadImages(source, img_size=imgsz, stride=stride)
        )

        # ---- Separate SORT trackers — completely independent ID spaces ----
        tracker_c1 = Sort(max_age=30, min_hits=2, iou_threshold=0.2)
        tracker_c2 = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

        count_c1: dict       = {v: 0 for v in names.values()}
        count_c2: dict       = {v: 0 for v in names.values()}
        counted_ids_c1: set  = set()
        counted_ids_c2: set  = set()

        roi_c1: np.ndarray | None = None
        roi_c2: np.ndarray | None = None
        roi_selected = False

        # ---- tqdm bar — shows total if video file, else infinite ----
        pbar = tqdm(
            dataset,
            desc="Inferencing",
            total=total_frames if total_frames > 0 else None,
            unit="frame",
            dynamic_ncols=True,
            colour="cyan",
        )

        for data in pbar:
            if STOP_REQUESTED:
                break

            t_frame_start = time.perf_counter()

            path, im, im0s, vid_cap, _ = data
            frame_idx += 1

            raw   = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
            frame = raw.copy()

            # ---- First-frame ROI setup (pauses the loop) ----
            if not roi_selected:
                print("\n=== ROI Setup ===")
                print("You will draw two ROI polygons — one per conveyor.")
                cv2.namedWindow(window_name)
                print("\n[1/2] Draw ROI for Conveyor 1 (cyan)")
                roi_c1 = select_roi_polygon(frame, window_name, "Conveyor 1", (0, 255, 255))
                print("\n[2/2] Draw ROI for Conveyor 2 (yellow)")
                roi_c2 = select_roi_polygon(frame, window_name, "Conveyor 2", (255, 255, 0))
                roi_selected = True
                print("\nROI setup complete. Starting inference…\n")

            # ---- Pre-process ----
            im_tensor = torch.from_numpy(im).to(device_obj).float() / 255.0
            if im_tensor.ndim == 3:
                im_tensor = im_tensor[None]

            # ===================== INFERENCE TIMER =====================
            t_infer_start = time.perf_counter()
            pred, proto   = model(im_tensor, augment=False, visualize=False)[:2]
            t_infer_end   = time.perf_counter()
            infer_ms       = (t_infer_end - t_infer_start) * 1000
            infer_times.append(infer_ms)
            last_infer_ms  = infer_ms

            # ================== POST-PROCESS TIMER ====================
            t_post_start = time.perf_counter()
            pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

            detections = []
            if len(pred[0]):
                pred[0][:, :4] = scale_boxes(
                    im_tensor.shape[2:], pred[0][:, :4], frame.shape
                ).round()
                masks_batch = process_mask(
                    proto[0], pred[0][:, 6:], pred[0][:, :4],
                    frame.shape[:2], upsample=True
                )
                for i, (*xyxy, conf, cls) in enumerate(pred[0][:, :6]):
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections.append([x1, y1, x2, y2, conf.item(), int(cls), masks_batch[i]])

            det_boxes  = (
                np.array([d[:5] for d in detections]) if detections else np.empty((0, 5))
            )
            tracks_c1  = tracker_c1.update(det_boxes).astype(int)
            tracks_c2  = tracker_c2.update(det_boxes).astype(int)
            t_post_end = time.perf_counter()
            postproc_ms       = (t_post_end - t_post_start) * 1000
            postproc_times.append(postproc_ms)
            last_postproc_ms  = postproc_ms
            # ===========================================================

            def _find_best_det(tx1, ty1, tx2, ty2):
                best_iou, best_det = 0.0, None
                for d in detections:
                    ix1, iy1 = max(tx1, d[0]), max(ty1, d[1])
                    ix2, iy2 = min(tx2, d[2]), min(ty2, d[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    a1 = (tx2 - tx1) * (ty2 - ty1)
                    a2 = (d[2] - d[0]) * (d[3] - d[1])
                    iou  = inter / (a1 + a2 - inter + 1e-6)
                    if iou > best_iou:
                        best_iou, best_det = iou, d
                return best_det

            def _get_mask_centroid(mask_tensor):
                mask = mask_tensor
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                mask = mask.astype(bool)
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    return None, None
                return int(xs.mean()), int(ys.mean())

            # ---- Conveyor 1 ----
            for trk in tracks_c1:
                x1, y1, x2, y2, tid = trk
                det = _find_best_det(x1, y1, x2, y2)
                if det is None:
                    continue
                cls_name = names[det[5]]
                cx, cy   = _get_mask_centroid(det[6])
                if cx is None:
                    continue
                update_roi_count(cx, cy, tid, cls_name, roi_c1, count_c1, counted_ids_c1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{cls_name} C1", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1, cv2.LINE_AA)

            # ---- Conveyor 2 ----
            for trk in tracks_c2:
                x1, y1, x2, y2, tid = trk
                det = _find_best_det(x1, y1, x2, y2)
                if det is None:
                    continue
                cls_name = names[det[5]]
                cx, cy   = _get_mask_centroid(det[6])
                if cx is None:
                    continue
                update_roi_count(cx, cy, tid, cls_name, roi_c2, count_c2, counted_ids_c2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{cls_name} C2 ", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1, cv2.LINE_AA)

            # ---- Draw ROI polygons ----
            if roi_c1 is not None and len(roi_c1) >= 3:
                cv2.polylines(frame, [roi_c1], isClosed=True, color=(0, 255, 255), thickness=1)
                lp = tuple(roi_c1[0][0])
                cv2.putText(frame, "ROI C1", (lp[0], max(20, lp[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            if roi_c2 is not None and len(roi_c2) >= 3:
                cv2.polylines(frame, [roi_c2], isClosed=True, color=(255, 255, 0), thickness=1)
                lp = tuple(roi_c2[0][0])
                cv2.putText(frame, "ROI C2", (lp[0], max(20, lp[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

            # ---- Count overlay (top-left) ----

            y_pos = 60

            for cls in sorted(set(list(count_c1) + list(count_c2))):

                color = get_class_color(cls)

                # ==========================================
                # CLASS NAME
                # ==========================================
                draw_text_with_gold_box(
                    frame,
                    f"{cls}",
                    (20, y_pos),
                    color
                )

                y_pos += 55

                # ==========================================
                # CONVEYOR 1 COUNT
                # ==========================================
                draw_text_with_gold_box(
                    frame,
                    f"C1: {count_c1.get(cls,0)}",
                    (20, y_pos),
                    color
                )

                y_pos += 55

                # ==========================================
                # CONVEYOR 2 COUNT
                # ==========================================
                draw_text_with_gold_box(
                    frame,
                    f"C2: {count_c2.get(cls,0)}",
                    (20, y_pos),
                    color
                )

                # ==========================================
                # GAP BETWEEN CLASSES
                # ==========================================
                y_pos += 70

            # ---- Total frame time → rolling FPS ----
            t_frame_end = time.perf_counter()
            frame_times.append(t_frame_end - t_frame_start)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

            # # ---- Performance HUD (top-right) ----
            # draw_perf_overlay(frame, last_infer_ms, last_postproc_ms,
            #                   avg_fps, frame_idx, total_frames)

            # ---- Update tqdm postfix (shows in terminal bar) ----
            pbar.set_postfix(
                infer_ms  = f"{last_infer_ms:.1f}",
                post_ms   = f"{last_postproc_ms:.1f}",
                fps       = f"{avg_fps:.1f}",
                c1        = sum(count_c1.values()),
                c2        = sum(count_c2.values()),
            )

            # ---- Video writers ----
            if raw_writer is None:
                h, w = frame.shape[:2]
                fps_src = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
                raw_writer = cv2.VideoWriter(
                    str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h)
                )
                ann_writer = cv2.VideoWriter(
                    str(ann_video), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h)
                )

            raw_writer.write(raw)
            ann_writer.write(frame)

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                request_stop()

        pbar.close()

    finally:
        if raw_writer:
            raw_writer.release()
        if ann_writer:
            ann_writer.release()
        cv2.destroyAllWindows()

        avg_infer    = sum(infer_times)    / len(infer_times)    if infer_times    else 0
        avg_postproc = sum(postproc_times) / len(postproc_times) if postproc_times else 0

        print(f"\n{'='*50}")
        print(f"✅  Raw video      : {raw_video}")
        print(f"✅  Annotated video: {ann_video}")
        print(f"📊  Frames processed : {frame_idx}")
        print(f"⚡  Avg inference    : {avg_infer:.1f} ms  ({1000/avg_infer:.1f} FPS)" if avg_infer else "")
        print(f"⚡  Avg post-proc    : {avg_postproc:.1f} ms")
        print(f"⚡  Avg total FPS    : {avg_fps:.1f}")
        print(f"{'='*50}")


# ================= CLI =================
def parse_opt():
    parser = argparse.ArgumentParser(
        description="YOLOv5-Seg ROI-based conveyor counter — two independent trackers"
    )
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--source",     required=True)
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres",  type=float, default=0.45)
    parser.add_argument("--device",     default="")
    parser.add_argument("--project",    default="runs/seg-count")
    parser.add_argument("--name",       default="exp")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))

# Example:
# python roi_normal_code_seg.py  --imgsz 640 --weights "C:\Users\admin\Downloads\olam_vid_2_seg_.pt" --conf-thres 0.60 --project "D:\bhanu\olam_agri\ai_infernces" --name "olam_agri_v1_" --source "D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Packaging\Olamagri\engineering\input_videos\olam_client_video_v2.mp4" --device 0