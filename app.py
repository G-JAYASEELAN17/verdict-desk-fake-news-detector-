import os
import re
import json
import joblib
import numpy as np

from flask import (
    Flask,
    request,
    jsonify,
    render_template
)

app = Flask(__name__)

# -------------------------------------------------------
# Files
# -------------------------------------------------------

MODEL_PATH = "model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"
METRICS_PATH = "metrics.json"

# -------------------------------------------------------
# Load trained model
# -------------------------------------------------------

print("=" * 60)
print("Loading Fake News Detector...")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "model.joblib not found.\n"
        "Run train_model.py first."
    )

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        "vectorizer.joblib not found.\n"
        "Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("✓ Model Loaded")
print("✓ Vectorizer Loaded")

# -------------------------------------------------------
# Load metrics
# -------------------------------------------------------

if os.path.exists(METRICS_PATH):

    with open(METRICS_PATH, "r") as f:

        metrics = json.load(f)

else:

    metrics = {}

# -------------------------------------------------------
# Text Cleaning
# -------------------------------------------------------

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"[^a-z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# -------------------------------------------------------
# Get Explainable Words
# -------------------------------------------------------

feature_names = np.array(
    vectorizer.get_feature_names_out()
)

coefficients = model.coef_[0]

# -------------------------------------------------------
# Prediction Function
# -------------------------------------------------------

def predict_news(news_text):

    cleaned_text = clean_text(news_text)

    vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    real_probability = round(float(probability[0] * 100), 2)
    fake_probability = round(float(probability[1] * 100), 2)

    UNCERTAIN_MARGIN = 10  

    if abs(fake_probability - 50) < UNCERTAIN_MARGIN:
        label = "UNCERTAIN"
        confidence = round(max(real_probability, fake_probability), 2)
    elif prediction == 1:
        label = "FAKE"
        confidence = fake_probability
    else:
        label = "REAL"
        confidence = real_probability

    # -------------------------------------------------------
    # Explainable AI
    # -------------------------------------------------------

    feature_index = vector.nonzero()[1]

    fake_scores = []
    real_scores = []

    for index in feature_index:

        word = feature_names[index]

        weight = coefficients[index]

        if weight > 0:

            fake_scores.append((word, weight))

        elif weight < 0:

            real_scores.append((word, abs(weight)))

    fake_scores = sorted(
        fake_scores,
        key=lambda x: x[1],
        reverse=True
    )[:8]

    real_scores = sorted(
        real_scores,
        key=lambda x: x[1],
        reverse=True
    )[:8]

    fake_words = [word for word, _ in fake_scores]

    real_words = [word for word, _ in real_scores]

    return {

        "label": label,

        "confidence": confidence,

        "probabilities": {

            "real": real_probability,

            "fake": fake_probability

        },

        "fake_indicators": fake_words,

        "real_indicators": real_words

    }


# -------------------------------------------------------
# Home Page
# -------------------------------------------------------

@app.route("/")

def home():

    return render_template("index.html")
# -------------------------------------------------------
# Prediction API
# -------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check if request is JSON
        if not request.is_json:

            return jsonify({
                "error": "Request must be JSON."
            }), 400

        data = request.get_json()

        news_text = data.get("text", "").strip()

        if len(news_text) < 20:

            return jsonify({
                "error": "Please enter at least 20 characters of text."
            }), 400

        result = predict_news(news_text)

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# -------------------------------------------------------
# Metrics API (Optional)
# -------------------------------------------------------

@app.route("/metrics")
def get_metrics():

    return jsonify(metrics)


# -------------------------------------------------------
# Health Check
# -------------------------------------------------------

@app.route("/health")
def health():

    return jsonify({
        "status": "running",
        "model_loaded": True
    })


# -------------------------------------------------------
# Run Server
# -------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print(" Fake News Detector Server Started ")
    print("=" * 60)
    print("Open your browser at:")
    print("http://127.0.0.1:5000")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )