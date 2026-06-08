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
LINE_Y = 400
BUFFER_PX = 100
BUFFER_SECONDS = 10
STOP_REQUESTED = False


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


def get_zone(cy, line_y):
    if cy < line_y - BUFFER_PX:
        return "left"
    elif cy > line_y + BUFFER_PX:
        return "right"
    else:
        return "buffer"


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
    name="exp"
):
    raw_writer = None
    ann_writer = None
    frame_idx = 0

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

        count_in = {v: 0 for v in names.values()}
        count_out = {v: 0 for v in names.values()}
        last_side = {}
        last_time = {}
        track_class = {}

        for data in tqdm(dataset, desc="Segmentation Counting"):
            if STOP_REQUESTED:
                break

            path, im, im0s, vid_cap, _ = data
            frame_idx += 1

            raw = im0s[0].copy() if isinstance(im0s, list) else im0s.copy()
            frame = raw.copy()
            h, w = frame.shape[:2]
            line_y = LINE_Y if 0 <= LINE_Y < h else h // 2

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
                cy = int(ys.mean())

                zone = get_zone(cy, line_y)
                prev = last_side.get(tid)
                last_t = last_time.get(tid, 0)

                if prev and zone != prev and (now - last_t) > BUFFER_SECONDS:
                    if prev == "left" and zone in ["buffer", "right"]:
                        count_in[cls_name] += 1
                        last_time[tid] = now
                        last_side[tid] = "right"
                    elif prev == "right" and zone in ["buffer", "left"]:
                        count_out[cls_name] += 1
                        last_time[tid] = now
                        last_side[tid] = "left"

                if zone in ["left", "right"]:
                    last_side[tid] = zone

                color = get_class_color(cls_name)
                frame[mask] = frame[mask] * 0.5 + np.array(color) * 0.5

                cv2.putText(frame, f"{cls_name} ID:{tid}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw horizontal Y-axis line and buffer lines
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)
            cv2.line(frame, (0, line_y - BUFFER_PX), (frame.shape[1], line_y - BUFFER_PX), (255, 215, 0), 1)
            cv2.line(frame, (0, line_y + BUFFER_PX), (frame.shape[1], line_y + BUFFER_PX), (255, 215, 0), 1)

            y = 40
            for cls in count_in:
                draw_text_with_gold_box(
                    frame,
                    f"{cls} IN:{count_in[cls]} OUT:{count_out[cls]}",
                    (15, y),
                    get_class_color(cls)
                )
                y += 34

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

            cv2.imshow("YOLOv5 Seg Counting", frame)
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
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
