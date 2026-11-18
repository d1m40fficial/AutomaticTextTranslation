"""Streamlit-песочница: сравнение моделей перевода EN → RU по BLEU.

Запуск: streamlit run streamlit_app.py
"""

import streamlit as st
from sacrebleu import sentence_bleu
from transformers import pipeline

MODEL_OPTIONS = {
    "Helsinki-NLP/opus-mt-en-ru (лучшая по BLEU)": "Helsinki-NLP/opus-mt-en-ru",
    "alirezamsh/tiny-opus-mt-en-ru (компактная)": "alirezamsh/tiny-opus-mt-en-ru",
    "facebook/wmt19-en-ru (тяжёлая, ~1.2 ГБ)": "facebook/wmt19-en-ru",
}

TEST_EXAMPLES = {
    "Hello, how are you?": "Привет, как дела?",
    "I love machine learning!": "Я обожаю машинное обучение!",
    "Artificial intelligence will change the world.": "Искусственный интеллект изменит мир.",
}


@st.cache_resource(show_spinner="Загружаю модель…")
def load_model(model_name: str):
    return pipeline(task="translation", model=model_name)


st.set_page_config(page_title="NeuroTranslate — сравнение моделей", layout="wide")
st.title("Переводчик EN → RU с оценкой качества")
st.caption("Сравнение трёх моделей Hugging Face по BLEU на эталонных переводах")

model_label = st.selectbox("Модель", list(MODEL_OPTIONS))
translator = load_model(MODEL_OPTIONS[model_label])

text_input = st.text_area("Текст на английском", height=180, placeholder="Enter English text…")

if st.button("Перевести", type="primary"):
    if not text_input.strip():
        st.warning("Введите текст для перевода.")
    else:
        with st.spinner("Перевожу…"):
            result = translator(text_input)[0]["translation_text"]
        st.subheader("Перевод")
        st.success(result)

st.sidebar.header("Оценка на эталоне")
example = st.sidebar.selectbox("Пример", list(TEST_EXAMPLES))

if st.sidebar.button("Проверить BLEU"):
    reference = TEST_EXAMPLES[example]
    hypothesis = translator(example)[0]["translation_text"]
    score = sentence_bleu(hypothesis, [reference]).score

    st.subheader("Исходный текст")
    st.write(example)
    st.subheader("Эталон")
    st.info(reference)
    st.subheader("Перевод модели")
    st.success(hypothesis)
    st.metric("BLEU", f"{score:.1f}")
