"""Бенчмарк моделей перевода EN → RU по BLEU.

Скрипт повторяет эксперимент из notebooks/Automatic_text_translation.ipynb:
берёт параллельный корпус (TSV: английский <TAB> русский), сэмплирует N пар
и считает corpus BLEU через sacrebleu для каждой модели.

Пример:
    python evaluation/evaluate_bleu.py --data data/rus.txt --sample 10000
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from sacrebleu import BLEU
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    FSMTForConditionalGeneration,
    FSMTTokenizer,
)

DEFAULT_MODELS = [
    "Helsinki-NLP/opus-mt-en-ru",
    "google-t5/t5-base",
    "facebook/wmt19-en-ru",
]


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def translate_batch(
    model_name: str,
    sources: list[str],
    batch_size: int = 32,
    max_length: int = 128,
) -> list[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "wmt19" in model_name.lower():
        tokenizer = FSMTTokenizer.from_pretrained(model_name)
        model = FSMTForConditionalGeneration.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.to(device).eval()
    predictions: list[str] = []

    with torch.inference_mode():
        for start in range(0, len(sources), batch_size):
            batch = sources[start : start + batch_size]
            if "t5" in model_name.lower():
                batch = [f"translate English to Russian: {text}" for text in batch]

            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            output = model.generate(**encoded, max_new_tokens=max_length, num_beams=1)
            predictions.extend(tokenizer.batch_decode(output, skip_special_tokens=True))

    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="TSV с парами предложений")
    parser.add_argument("--sample", type=int, default=10000, help="сколько пар оценивать")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Файл не найден: {args.data}. См. data/README.md", file=sys.stderr)
        return 1

    pairs = read_pairs(args.data)
    print(f"Всего пар: {len(pairs)}")
    if len(pairs) > args.sample:
        random.seed(args.seed)
        pairs = random.sample(pairs, args.sample)
    print(f"Оцениваем пар: {len(pairs)}")

    sources = [src for src, _ in pairs]
    references = [tgt for _, tgt in pairs]
    metric = BLEU()
    results: list[tuple[str, float]] = []

    for model_name in args.models:
        print(f"\n→ {model_name}")
        try:
            predictions = translate_batch(model_name, sources, batch_size=args.batch_size)
        except Exception as error:  # noqa: BLE001 - бенчмарк не должен падать целиком
            print(f"  ошибка: {error}")
            continue
        score = metric.corpus_score(predictions, [references]).score
        results.append((model_name, score))
        print(f"  BLEU: {score:.2f}")

    print("\nИтоговая таблица")
    for model_name, score in sorted(results, key=lambda item: item[1], reverse=True):
        print(f"  {score:6.2f}  {model_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
