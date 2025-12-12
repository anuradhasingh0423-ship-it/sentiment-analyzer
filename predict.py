import pickle
from pathlib import Path
from utils import clean_text
import numpy as np

MODELS_DIR = Path("models")

with open(MODELS_DIR / "best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODELS_DIR / "tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

def predict_sentiment_with_confidence(text: str):
    clean = clean_text(text)
    vec = tfidf.transform([clean])
    # Some models return predict_proba, some (rarely) don't: handle both
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))
    else:
        # fallback: use decision_function and a softmax-ish conversion
        if hasattr(model, "decision_function"):
            df = model.decision_function(vec)
            # For binary, decision_function returns shape (n_samples,)
            # Convert to probability-like via logistic
            from scipy.special import expit
            conf = float(expit(df[0]))  # between 0-1
            pred = int(conf >= 0.5)
        else:
            pred = int(model.predict(vec)[0])
            conf = 1.0
    label = "Positive" if pred == 1 else "Negative"
    return {"label": label, "pred": pred, "confidence": round(conf, 4), "clean_text": clean}
