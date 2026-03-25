"""
model_training.py — Train Logistic Regression and Naive Bayes classifiers.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.config import (
    LR_C, LR_MAX_ITER, NB_ALPHA,
    LR_MODEL_PATH, NB_MODEL_PATH, BEST_MODEL_PATH,
    RANDOM_SEED
)
from src.utils import save_artifact


# ─── Logistic Regression ──────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """Train a Logistic Regression classifier."""
    print("[*] Training Logistic Regression …")
    model = LogisticRegression(
        C=LR_C,
        max_iter=LR_MAX_ITER,
        solver="lbfgs",
        multi_class="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[✓] Logistic Regression training complete.")
    return model


# ─── Multinomial Naive Bayes ──────────────────────────────────────────────────

def train_naive_bayes(X_train, y_train) -> MultinomialNB:
    """Train a Multinomial Naive Bayes classifier."""
    print("[*] Training Multinomial Naive Bayes …")
    model = MultinomialNB(alpha=NB_ALPHA)
    model.fit(X_train, y_train)
    print("[✓] Naive Bayes training complete.")
    return model


# ─── Train All Models ─────────────────────────────────────────────────────────

def train_all_models(X_train, y_train) -> dict:
    """
    Train both classifiers and return them in a dictionary.

    Returns
    -------
    {
      "Logistic Regression": <model>,
      "Naive Bayes":         <model>,
    }
    """
    lr_model = train_logistic_regression(X_train, y_train)
    nb_model = train_naive_bayes(X_train, y_train)

    return {
        "Logistic Regression": lr_model,
        "Naive Bayes": nb_model,
    }


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_models(models: dict) -> None:
    """Save both trained models to disk."""
    save_artifact(models["Logistic Regression"], LR_MODEL_PATH)
    save_artifact(models["Naive Bayes"],         NB_MODEL_PATH)


def save_best_model(model, name: str) -> None:
    """Save the best-performing model for use in the Streamlit app."""
    save_artifact(model, BEST_MODEL_PATH)
    print(f"[★] Best model ({name}) saved → {BEST_MODEL_PATH}")
