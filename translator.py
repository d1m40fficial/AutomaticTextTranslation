"""Нейропереводчик EN → RU.

Обёртка над Helsinki-NLP/opus-mt-en-ru (лучшая модель по BLEU в бенчмарке
проекта). Модель загружается лениво, один раз на процесс, и переиспользуется
между запросами.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "Helsinki-NLP/opus-mt-en-ru")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "512"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = None
_model = None


def load() -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """Загружает токенизатор и модель (кэшируется в модуле)."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.to(DEVICE).eval()
    return _tokenizer, _model


def device_name() -> str:
    return DEVICE.type


def translate(text: str) -> str:
    """Переводит английский текст на русский.

    Пустая строка или строка из пробелов возвращает пустую строку.
    """
    text = (text or "").strip()
    if not text:
        return ""

    tokenizer, model = load()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(DEVICE)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, num_beams=4)

    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(f"Модель: {MODEL_NAME} | устройство: {device_name()}")
    while True:
        try:
            line = input("EN> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in {"exit", "quit"}:
            break
        print(f"RU> {translate(line)}")
