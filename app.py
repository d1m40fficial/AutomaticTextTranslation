from flask import Flask, render_template, request
from model import translate

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    translated_text = ""
    if request.method == "POST":
        input_text = request.form["text"]
        translated_text = translate(input_text)
    return render_template("index.html", translation=translated_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
