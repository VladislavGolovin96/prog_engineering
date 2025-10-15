#!/usr/bin/env python3

import argparse
import json
import io
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import requests


YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YAMNET_CLASS_MAP_CSV = "https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv"
TARGET_SR = 16000


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Read audio file with soundfile. Returns float32 waveform in [-1, 1] and sample rate."""
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")  # (n, channels)
    # stereo -> mono
    mono = np.mean(data, axis=1)
    return mono, sr


def resample_linear(x: np.ndarray, sr_src: int, sr_dst: int) -> np.ndarray:
    """Lightweight linear resampling to sr_dst using numpy (no extra deps)."""
    if sr_src == sr_dst:
        return x
    duration = x.shape[0] / float(sr_src)
    n_dst = int(round(duration * sr_dst))
    if n_dst <= 1:
        return np.zeros((1,), dtype=x.dtype)
    t_src = np.linspace(0.0, duration, num=x.shape[0], endpoint=False)
    t_dst = np.linspace(0.0, duration, num=n_dst, endpoint=False)
    y = np.interp(t_dst, t_src, x).astype(np.float32)
    return y


def load_yamnet():
    # TF Hub SavedModel
    model = hub.load(YAMNET_HANDLE)
    return model


def load_labels(model) -> list[str]:
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    names = []
    with tf.io.gfile.GFile(class_map_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["display_name"])
    if len(names) < 100:
        raise ValueError("Unexpectedly short class map in model assets")
    return names


def top_k(scores_mean: np.ndarray, labels: List[str], k: int) -> List[Tuple[str, float, int]]:
    """Return top-k (label, prob, index) sorted by prob desc."""
    idx = np.argsort(scores_mean)[::-1][:k]
    return [(labels[i] if i < len(labels) else f"class_{i}", float(scores_mean[i]), int(i)) for i in idx]


def to_json_result(path: Path, top: List[Tuple[str, float, int]]) -> dict:
    return {
        "model": "yamnet/1 (TF Hub)",
        "file": str(path),
        "predictions": [
            {"label": label, "score": prob, "class_index": i}
            for (label, prob, i) in top
        ],
    }


def save_csv(top: List[Tuple[str, float, int]], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "label", "score", "class_index"])
        for r, (label, prob, i) in enumerate(top, start=1):
            w.writerow([r, label, f"{prob:.6f}", i])


def main():
    ap = argparse.ArgumentParser(description="Audio classification with YAMNet (TF Hub)")
    ap.add_argument("--audio", type=Path, required=True, help="Path to input WAV/MP3")
    ap.add_argument("--topk", type=int, default=5, help="How many top classes to show")
    ap.add_argument("--json", action="store_true", help="Print JSON output")
    ap.add_argument("--save-csv", type=Path, help="Optional path to save CSV with top-K")
    args = ap.parse_args()

    # 1) Load audio
    wav, sr = read_audio(args.audio)
    wav = np.asarray(wav, dtype=np.float32)
    # Normalize (safety)
    wav = np.clip(wav, -1.0, 1.0)

    # 2) Resample to 16 kHz (YAMNet requirement)
    wav16 = resample_linear(wav, sr, TARGET_SR)

    # 3) Load model & labels
    model = load_yamnet()
    labels = load_labels(model)

    # 4) Inference
    # YAMNet returns (scores, embeddings, spectrogram)
    scores, embeddings, log_mel = model(wav16)
    # Average over time frames to get a single distribution
    scores_mean = tf.reduce_mean(scores, axis=0).numpy()

    # 5) Top-K
    top = top_k(scores_mean, labels, k=max(1, args.topk))

    # 6) Output
    if args.save_csv:
        save_csv(top, args.save_csv)

    if args.json:
        print(json.dumps(to_json_result(args.audio, top), ensure_ascii=False, indent=2))
    else:
        print(f"[YAMNet] File: {args.audio}")
        for r, (label, prob, i) in enumerate(top, start=1):
            print(f"{r:>2}. {label:30s}  score={prob:.4f} (class_id={i})")


if __name__ == "__main__":
    # Use Apple GPU when available (no special flags needed for TF metal plugin)
    # Just ensure tensorflow-metal is installed.
    main()
