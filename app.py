import os
import pickle
import nltk
import string

from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

# -------------------------
# Download NLTK Resources
# -------------------------

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# -------------------------
# Flask App
# -------------------------

app = Flask(__name__)

ps = PorterStemmer()

# -------------------------
# Load Model Files
# -------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# -------------------------
# Text Preprocessing
# -------------------------

def transform_text(text):

    text = text.lower()

    # No punkt / punkt_tab needed
    text = wordpunct_tokenize(text)

    y = []

    for word in text:
        if word.isalnum():
            y.append(word)

    text = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))

    for word in text:
        if word not in stop_words and word not in string.punctuation:
            y.append(word)

    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    return " ".join(y)

# -------------------------
# Routes
# -------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        message = request.form['message']

        word_count = len(message.split())
        char_count = len(message)

        transformed = transform_text(message)

        vector = vectorizer.transform([transformed])

        result = model.predict(vector)[0]

        confidence = round(
            max(model.predict_proba(vector)[0]) * 100,
            2
        )

        if result == 1:

            prediction = "Spam"

            advice = [
                "Do not click suspicious links.",
                "Never share OTP, passwords, or bank details.",
                "Verify the sender using official channels.",
                "Block and report suspicious senders.",
                "Delete messages requesting urgent payments."
            ]

        else:

            prediction = "Ham"

            advice = [
                "This message appears legitimate.",
                "Always verify sensitive requests independently.",
                "Be cautious when sharing personal information."
            ]

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=confidence,
            advice=advice,
            word_count=word_count,
            char_count=char_count
        )

    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------
# Run App
# -------------------------

if __name__ == "__main__":
    app.run(debug=True)