"""Flask-интерфейс нейропереводчика EN → RU."""

from flask import Flask, jsonify, render_template, request

from translator import MODEL_NAME, device_name, translate

app = Flask(__name__)

MAX_CHARS = 2000
EXAMPLES = [
    "Machine translation has improved dramatically over the last decade.",
    "The model was selected by the highest BLEU score on 10,000 sentence pairs.",
    "Please enter the text you want to translate and press the button.",
]


@app.get("/")
def index():
    return render_template(
        "index.html",
        translation="",
        source_text="",
        examples=EXAMPLES,
        model_name=MODEL_NAME,
        max_chars=MAX_CHARS,
    )


@app.post("/")
def translate_form():
    source_text = (request.form.get("text") or "")[:MAX_CHARS]
    translation = translate(source_text) if source_text.strip() else ""
    return render_template(
        "index.html",
        translation=translation,
        source_text=source_text,
        examples=EXAMPLES,
        model_name=MODEL_NAME,
        max_chars=MAX_CHARS,
    )


@app.post("/api/translate")
def translate_api():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "")[:MAX_CHARS]
    return jsonify({"translation": translate(text)})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "device": device_name()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
