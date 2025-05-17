# streamlit_app.py

import streamlit as st
from transformers import pipeline
from sacrebleu import corpus_bleu

# Список поддерживаемых моделей
MODEL_OPTIONS = {
    "Helsinki-NLP": "Helsinki-NLP/opus-mt-en-ru",
    "TinyMT": "alirezamsh/tiny-opus-mt-en-ru",
    "NLLB": "facebook/nllb-207"
}

# Тестовые данные (можно заменить на свои)
TEST_EXAMPLES = {
    "Hello, how are you?": "Привет, как дела?",
    "I love machine learning!": "Я обожаю машинное обучение!",
    "Artificial intelligence will change the world.": "Искусственный интеллект изменит мир."
}

@st.cache_resource
def load_model(model_name):
    return pipeline("translation_en_to_ru", model=model_name)

def calculate_bleu(reference, hypothesis):
    return corpus_bleu([hypothesis], [[reference]]).score

# Интерфейс
st.set_page_config(page_title="Переводчик с анализом", layout="wide")
st.title("Переводчик EN → RU с оценкой качества")

model_choice = st.selectbox("Выберите модель:", list(MODEL_OPTIONS.keys()))
translator = load_model(MODEL_OPTIONS[model_choice])

text_input = st.text_area("Введите текст на английском:", height=200)

if st.button("Перевести"):
    if text_input.strip() == "":
        st.warning("Пожалуйста, введите текст для перевода.")
    else:
        with st.spinner("Перевожу..."):
            result = translator(text_input)[0]["translation_text"]
            st.subheader("Результат перевода:")
            st.success(result)

# Блок с тестовыми примерами
st.sidebar.header("Тестовые примеры")
example = st.sidebar.selectbox("Выберите пример", list(TEST_EXAMPLES.keys()))

if st.sidebar.button("Запустить пример"):
    reference = TEST_EXAMPLES[example]
    hypothesis = translator(example)[0]["translation_text"]
    bleu_score = calculate_bleu(reference, hypothesis)

    st.subheader("Пример текста:")
    st.write(example)
    st.subheader("Эталонный перевод:")
    st.info(reference)
    st.subheader("Перевод модели:")
    st.success(hypothesis)
    st.subheader("BLEU-оценка:")
    st.metric(label="Оценка BLEU", value=f"{bleu_score:.1f}")