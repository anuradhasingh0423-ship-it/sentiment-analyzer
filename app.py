# app.py (update)
from flask import Flask, render_template, request, jsonify
import logging
from pathlib import Path
from predict import predict_sentiment_with_confidence
# LIME for explanations
from lime.lime_text import LimeTextExplainer
import pickle
import numpy as np

app = Flask(__name__)

# basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# For LIME we need a prediction function that accepts list[str] and returns prob matrix
from predict import model, tfidf

def predict_proba_for_lime(texts):
    # texts: list of raw texts (we need to clean & vectorize similarly)
    from utils import clean_text
    cleaned = [clean_text(t) for t in texts]
    vecs = tfidf.transform(cleaned)
    # Some models have predict_proba; use that
    if hasattr(model, "predict_proba"):
        return model.predict_proba(vecs)
    elif hasattr(model, "decision_function"):
        # convert decision_function to two-class probs
        df = model.decision_function(vecs)
        from scipy.special import expit
        probs_pos = expit(df)  # probability of positive class
        probs = []
        for p in probs_pos:
            probs.append([1-p, p])
        return np.array(probs)
    else:
        preds = model.predict(vecs)
        probs = []
        for p in preds:
            probs.append([1.0-p, p])  # dumb fallback
        return np.array(probs)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    input_text = ""
    explanation = None
    confidence = None   # ADD THIS

    if request.method == "POST":
        input_text = request.form.get("review", "")
        if input_text.strip():
            result = predict_sentiment_with_confidence(input_text)
            prediction = result["label"]
            confidence = result["confidence"]   # ADD THIS

            # LIME explanation
            explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
            exp = explainer.explain_instance(input_text, predict_proba_for_lime, num_features=8)
            explanation = exp.as_list()

    return render_template(
        "index.html",
        prediction=prediction,
        input_text=input_text,
        explanation=explanation,
        confidence=confidence  # ADD THIS
    )


import json

@app.route("/models")
def model_dashboard():
    metrics_path = Path("models/metrics.json")
    
    if not metrics_path.exists():
        return "Metrics not found. Run train.py first.", 500
    
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    return render_template("models.html", metrics=metrics)



# Simple JSON API
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Please send JSON with a 'text' field"}), 400
    text = data["text"]
    logger.info("API predict request length=%d", len(text))
    out = predict_sentiment_with_confidence(text)
    return jsonify(out)

# Simple docs route
@app.route("/docs", methods=["GET"])
def docs():
    return jsonify({
        "description": "IMDB Sentiment API",
        "endpoints": {
            "/api/predict": {"method": "POST", "payload": {"text": "I loved the movie"}, "returns": {"label": "Positive", "confidence": 0.92}}
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
