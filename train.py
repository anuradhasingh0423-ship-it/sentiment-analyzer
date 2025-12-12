import pandas as pd
import numpy as np
import pickle
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from utils import clean_text

# ================================
# Paths
# ================================
DATA_PATH = Path("data/IMDB Dataset.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ================================
# Load Dataset
# ================================
df = pd.read_csv(DATA_PATH)
df["review"] = df["review"].astype(str).apply(clean_text)
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# ================================
# Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.25, random_state=42, stratify=df["sentiment"]
)

# ================================
# TF-IDF Vectorizer
# ================================
tfidf = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ================================
# Models
# ================================
models = {
    "MultinomialNB": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "LinearSVC": LinearSVC(C=1.0, max_iter=5000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = []
trained_models = {}

# ================================
# Training Loop
# ================================
for name, model in models.items():
    print(f"\nTraining {name} ...")
    start = time.time()

    # SVM needs calibration for predict_proba
    if name == "LinearSVC":
        calibrated = CalibratedClassifierCV(model, cv=3)
        calibrated.fit(X_train_tfidf, y_train)
        clf = calibrated
    else:
        model.fit(X_train_tfidf, y_train)
        clf = model

    train_time = time.time() - start

    # Predictions
    preds = clf.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    print(f"{name}: Acc={acc:.4f} | F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | Time={train_time:.1f}s")

    results.append({
        "model": name,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "train_time_s": train_time
    })

    trained_models[name] = clf

    # Save model
    with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(clf, f)

# ================================
# Leaderboard
# ================================
leaderboard = pd.DataFrame(results).sort_values(by="f1", ascending=False)
leaderboard.to_csv(MODELS_DIR / "leaderboard.csv", index=False)

print("\nLeaderboard:\n", leaderboard)

# ================================
# Save TF-IDF
# ================================
with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# ================================
# Save Best Model
# ================================
best_model_name = leaderboard.iloc[0]["model"]
best_model = trained_models[best_model_name]

with open(MODELS_DIR / "best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"\nSaved best model: {best_model_name} → models/best_model.pkl")
print("Saved TF-IDF vectorizer → models/tfidf_vectorizer.pkl")

# ================================
# Metrics for Dashboard (/models)
# ================================
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }

metrics = {
    name: evaluate_model(model, X_test_tfidf, y_test)
    for name, model in trained_models.items()
}

# Add best model too
metrics["best_model"] = evaluate_model(best_model, X_test_tfidf, y_test)

# Save metrics.json
import json
with open(MODELS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nSaved metrics.json for dashboard → models/metrics.json")
