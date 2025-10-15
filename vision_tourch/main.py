#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()  
    categories = weights.meta.get("categories", [])
    return model, preprocess, categories


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


@torch.inference_mode()
def predict_topk(model, preprocess, device, img: Image.Image, categories: List[str], topk: int) -> List[Tuple[str, float, int]]:
    batch = preprocess(img).unsqueeze(0).to(device)
    logits = model(batch)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    k = max(1, min(topk, probs.shape[0]))
    vals, idxs = probs.topk(k)
    result = []
    for p, i in zip(vals.tolist(), idxs.tolist()):
        label = categories[i] if i < len(categories) else f"class_{i}"
        result.append((label, float(p), int(i)))
    return result


def save_visual(img: Image.Image, top1: Tuple[str, float, int], out_path: Path):
    """Overlay Top-1 label on the image and save."""
    label, prob, _ = top1
    txt = f"{label}  ({prob:.2f})"
    draw = ImageDraw.Draw(img)
    pad = 8
    try:
        font = ImageFont.load_default()
        w, h = draw.textbbox((0,0), txt, font=font)[2:]
    except Exception:
        font = None
        w, h = draw.textlength(txt), 18  # fallback
    box = [10, 10, 10 + w + 2*pad, 10 + h + 2*pad]
    draw.rectangle(box, fill=(0, 0, 0, 160))
    draw.text((10 + pad, 10 + pad), txt, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def to_json_result(path: Path, preds: List[Tuple[str, float, int]]) -> dict:
    return {
        "model": "torchvision.resnet50 (ImageNet)",
        "file": str(path),
        "predictions": [
            {"label": label, "score": prob, "class_index": idx} for (label, prob, idx) in preds
        ],
    }


def main():
    ap = argparse.ArgumentParser(description="Image classification with ResNet-50 (PyTorch)")
    ap.add_argument("--image", type=Path, required=True, help="Path to image file (jpg/png/webp, etc.)")
    ap.add_argument("--topk", type=int, default=5, help="How many top classes to show")
    ap.add_argument("--json", action="store_true", help="Print JSON output")
    ap.add_argument("--save-vis", type=Path, help="Optional path to save image with Top-1 label overlay")
    args = ap.parse_args()

    device = get_device()
    model, preprocess, categories = load_model(device)
    img = load_image(args.image)

    preds = predict_topk(model, preprocess, device, img, categories, topk=args.topk)

    if args.save_vis:
        save_visual(img.copy(), preds[0], args.save_vis)

    if args.json:
        print(json.dumps(to_json_result(args.image, preds), ensure_ascii=False, indent=2))
    else:
        print(f"[ResNet50] File: {args.image} | device={device} | topK={args.topk}")
        for i, (label, prob, idx) in enumerate(preds, start=1):
            print(f"{i:>2}. {label:35s}  score={prob:.4f} (class_id={idx})")


if __name__ == "__main__":
    main()
