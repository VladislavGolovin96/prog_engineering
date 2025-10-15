#!/usr/bin/env python3


import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline


MODEL_ID = "facebook/detr-resnet-50"


@dataclass
class Detection:
    frame: int
    idx:   int
    label: str
    score: float
    box:   Tuple[int, int, int, int]  # x1, y1, x2, y2


def draw_boxes(pil_img: Image.Image, dets: List[Detection]) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for d in dets:
        x1, y1, x2, y2 = d.box
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        text = f"{d.label} {d.score:.2f}"
        if font:
            tw, th = draw.textbbox((0,0), text, font=font)[2:]
        else:
            tw, th = (len(text) * 6, 12)
        pad = 3
        draw.rectangle([x1, y1 - th - 2*pad, x1 + tw + 2*pad, y1], fill=(0, 255, 0))
        draw.text((x1 + pad, y1 - th - pad), text, fill=(0, 0, 0), font=font)
    return img


def write_csv(rows: List[Detection], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "index", "label", "score", "x1", "y1", "x2", "y2"])
        for d in rows:
            x1, y1, x2, y2 = d.box
            w.writerow([d.frame, d.idx, d.label, f"{d.score:.6f}", x1, y1, x2, y2])


def detections_to_json(rows: List[Detection], video: Path) -> Dict[str, Any]:
    return {
        "model": MODEL_ID,
        "video": str(video),
        "detections": [
            {
                "frame": d.frame,
                "index": d.idx,
                "label": d.label,
                "score": d.score,
                "box": {"x1": d.box[0], "y1": d.box[1], "x2": d.box[2], "y2": d.box[3]},
            }
            for d in rows
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="Object detection on video with DETR")
    ap.add_argument("--video", type=Path, required=True, help="Path to input video (e.g., .mp4)")
    ap.add_argument("--stride", type=int, default=5, help="Process every N-th frame (default: 5)")
    ap.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    ap.add_argument("--json", action="store_true", help="Print JSON output")
    ap.add_argument("--save-csv", type=Path, help="Save detections to CSV")
    ap.add_argument("--save-preview", type=Path, help="Save annotated preview (first processed frame)")
    args = ap.parse_args()

    # 1) Open video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    # 2) HF pipeline
    det_pipe = pipeline("object-detection", model=MODEL_ID, framework="pt")

    rows: List[Detection] = []
    first_preview_saved = False
    frame_id = -1
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        # Sample frames
        if frame_id % args.stride != 0:
            continue

        # BGR -> RGB -> PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # 3) Run detector
        preds = det_pipe(pil_img)

        # 4) Collect filtered detections
        frame_dets: List[Detection] = []
        for i, p in enumerate(preds):
            score = float(p.get("score", 0.0))
            if score < args.conf_thres:
                continue
            box = p.get("box", {})
            x1, y1, x2, y2 = int(box.get("xmin", 0)), int(box.get("ymin", 0)), int(box.get("xmax", 0)), int(box.get("ymax", 0))
            label = str(p.get("label", "object"))
            det = Detection(frame=frame_id, idx=i, label=label, score=score, box=(x1, y1, x2, y2))
            rows.append(det)
            frame_dets.append(det)

        # 5) Save preview (first processed frame with boxes that passed threshold)
        if args.save_preview and not first_preview_saved:
            annotated = draw_boxes(pil_img, frame_dets)
            args.save_preview.parent.mkdir(parents=True, exist_ok=True)
            annotated.save(args.save_preview)
            first_preview_saved = True

        processed += 1

    cap.release()

  
    if args.save_csv:
        write_csv(rows, args.save_csv)

    if args.json:
        print(json.dumps(detections_to_json(rows, args.video), ensure_ascii=False, indent=2))
    else:
        print(f"[DETR] Video: {args.video} | frames_processed={processed} | stride={args.stride} | conf>={args.conf_thres}")
        # print brief per-frame summary
        by_frame = {}
        for d in rows:
            by_frame.setdefault(d.frame, 0)
            by_frame[d.frame] += 1
        for f in sorted(by_frame)[:20]:  # limit stdout spam
            print(f"  frame {f}: {by_frame[f]} detections")
        if len(by_frame) > 20:
            print(f"  ... {len(by_frame)-20} more frames with detections")

if __name__ == "__main__":
    main()
