#!/usr/bin/env python3
import argparse, json, sys, pathlib
from transformers import pipeline, AutoTokenizer

MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def load_text_from_file(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Анализатор текста (Hugging Face Transformers)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Текст для анализа")
    src.add_argument("--file", type=pathlib.Path, help="Путь до .txt файла")
    p.add_argument("--json", action="store_true", help="Вывод в формате JSON")
    args = p.parse_args()

    text = args.text if args.text else load_text_from_file(args.file)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    clf = pipeline("sentiment-analysis", model=MODEL_ID, tokenizer=tokenizer, framework="pt")

    preds = clf(text if isinstance(text, str) else str(text))
    label_raw = preds[0].get("label")
    score = float(preds[0].get("score", 0.0))

    out = {
        "model": MODEL_ID,
        "input": text,
        "prediction": {"label": label_raw, "score": score},
    }

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"[{MODEL_ID}] → {label_raw} (score={score:.3f})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
