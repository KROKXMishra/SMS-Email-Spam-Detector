import os
import pickle
import nltk
import string

from flask import Flask, render_template, request, session, jsonify
from flask_cors import CORS

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
# ==================================
# NLTK Setup
# ==================================

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# ==================================
# Flask App
# ==================================

app = Flask(__name__)
app.secret_key = "spam_detector_secret_key"

CORS(app)

ps = PorterStemmer()

# ==================================
# Model Loading
# ==================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# ==================================
# Model Metrics
# ==================================

ACCURACY = 97.29
PRECISION = 99.16
RECALL = 81.38
F1_SCORE = 89.39

# ==================================
# Text Preprocessing
# ==================================

def transform_text(text):

    text = text.lower()

    text = wordpunct_tokenize(text)

    words = []

    for word in text:
        if word.isalnum():
            words.append(word)

    stop_words = set(stopwords.words("english"))

    filtered = []

    for word in words:
        if word not in stop_words and word not in string.punctuation:
            filtered.append(word)

    stemmed = []

    for word in filtered:
        stemmed.append(ps.stem(word))

    return " ".join(stemmed)

# ==================================
# Spam Category Detection
# ==================================

def detect_category(text):

    text = text.lower()

    if any(word in text for word in [
        "bank", "account", "otp",
        "verify", "payment",
        "credit card", "debit card"
    ]):
        return "🏦 Financial Scam"

    elif any(word in text for word in [
        "lottery", "winner", "won",
        "prize", "reward"
    ]):
        return "🎁 Lottery Scam"

    elif any(word in text for word in [
        "click", "link", "login",
        "password", "update"
    ]):
        return "🎣 Phishing Attempt"

    elif any(word in text for word in [
        "offer", "sale", "discount",
        "deal", "free"
    ]):
        return "🛍 Promotional Spam"

    return "📩 General Message"

# ==================================
# Home
# ==================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        accuracy=ACCURACY,
        precision=PRECISION,
        recall=RECALL,
        f1=F1_SCORE,
        history=session.get("history", [])
    )

# ==================================
# Website Prediction Route
# ==================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        message = request.form["message"]

        word_count = len(message.split())
        char_count = len(message)

        transformed = transform_text(message)

        vector = vectorizer.transform([transformed])

        prediction_value = model.predict(vector)[0]

        confidence = round(
            max(model.predict_proba(vector)[0]) * 100,
            2
        )

        if prediction_value == 1:
            category = detect_category(message)
        else:
            category = ""

        if prediction_value == 1:

            prediction = "Spam"

            advice = [
                "Do not click suspicious links.",
                "Never share OTP or passwords.",
                "Verify the sender independently.",
                "Report suspicious messages.",
                "Delete messages demanding urgent payment."
            ]

        else:

            prediction = "Ham"

            advice = [
                "This message appears legitimate.",
                "Always verify sensitive requests.",
                "Be cautious before sharing personal information."
            ]

        history = session.get("history", [])

        history.insert(
            0,
            {
                "message": (
                    message[:80] + "..."
                    if len(message) > 80
                    else message
                ),
                "prediction": prediction,
                "confidence": confidence
            }
        )

        history = history[:5]

        session["history"] = history

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=confidence,
            category=category,
            advice=advice,
            word_count=word_count,
            char_count=char_count,
            history=history,
            accuracy=ACCURACY,
            precision=PRECISION,
            recall=RECALL,
            f1=F1_SCORE
        )

    except Exception as e:

        return f"Error: {str(e)}"

# ==================================
# API Route For Chrome Extension
# ==================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:

        data = request.get_json()

        message = data["message"]

        transformed = transform_text(message)

        vector = vectorizer.transform([transformed])

        prediction_value = model.predict(vector)[0]

        confidence = round(
            max(model.predict_proba(vector)[0]) * 100,
            2
        )

        category = ""

        if prediction_value == 1:
            category = detect_category(message)

        return jsonify({
            "prediction":
                "Spam"
                if prediction_value == 1
                else "Ham",

            "confidence": confidence,

            "category": category
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# ==================================
# Run App
# ==================================

if __name__ == "__main__":
    app.run(debug=True)