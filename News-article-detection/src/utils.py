"""
utils.py — Helper functions: model persistence, single-text inference.
"""
import os
import json
import joblib
import numpy as np

from src.config import TFIDF_VECTORIZER_PATH, LABEL_MAP


# ─── Model Persistence ────────────────────────────────────────────────────────

def save_artifact(obj, path: str) -> None:
    """Serialize any Python object to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"[✓] Saved → {path}")


def load_artifact(path: str):
    """Deserialize a joblib-saved object from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artifact not found at '{path}'. "
            "Please run train.py to generate model files first."
        )
    return joblib.load(path)


def save_metrics(metrics: dict, path: str) -> None:
    """Persist a metrics dictionary as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[✓] Metrics saved → {path}")


def load_metrics(path: str) -> dict:
    """Load metrics from a JSON file."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


# ─── Inference Helper ─────────────────────────────────────────────────────────

def predict_single(text: str, model, vectorizer) -> dict:
    """
    Predict whether a single news article is Fake or Real.

    Parameters
    ----------
    text       : raw article string (will be cleaned internally)
    model      : trained sklearn classifier
    vectorizer : fitted TfidfVectorizer

    Returns
    -------
    dict with keys:
      - label        : "FAKE" | "REAL"
      - confidence   : float (probability of predicted class)
      - probabilities: {"FAKE": float, "REAL": float}
    """
    from src.data_preprocessing import clean_text

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label = LABEL_MAP[pred]
    return {
        "label": label,
        "confidence": float(round(proba[pred] * 100, 2)),
        "probabilities": {
            "FAKE": float(round(proba[0] * 100, 2)),
            "REAL": float(round(proba[1] * 100, 2)),
        },
    }


def get_top_features(text: str, model, vectorizer, top_n: int = 15) -> list:
    """
    Return the top-N TF-IDF features most influential for the prediction.
    Works only with Logistic Regression (coefficient-based explanation).

    Returns a list of (word, weight) tuples sorted by absolute contribution.
    """
    from src.data_preprocessing import clean_text

    if not hasattr(model, "coef_"):
        return []  # Not supported for NB without extra work

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    feature_names = np.array(vectorizer.get_feature_names_out())
    # Coefficients for class=1 (REAL)
    coef = model.coef_[0]

    # Only look at non-zero TF-IDF features of the input
    nonzero_indices = X.nonzero()[1]
    contributions = [(feature_names[i], float(coef[i])) for i in nonzero_indices]

    # Sort by absolute value; positive → REAL, negative → FAKE
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]
