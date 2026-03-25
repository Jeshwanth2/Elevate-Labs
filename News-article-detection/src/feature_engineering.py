"""
feature_engineering.py — TF-IDF vectorisation with fit/transform/persist support.
"""
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

from src.config import (
    TFIDF_VECTORIZER_PATH,
    MAX_FEATURES, NGRAM_RANGE, MIN_DF
)
from src.utils import save_artifact, load_artifact


# ─── Vectorizer Factory ───────────────────────────────────────────────────────

def build_vectorizer() -> TfidfVectorizer:
    """Create a new TfidfVectorizer with project-wide settings."""
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        sublinear_tf=True,      # apply 1 + log(tf) scaling
        strip_accents="unicode",
        analyzer="word",
    )


def fit_transform_vectorizer(X_train, X_test):
    """
    Fit TF-IDF on training data, transform both train and test sets.

    Returns
    -------
    vectorizer : fitted TfidfVectorizer
    X_train_tfidf : sparse matrix
    X_test_tfidf  : sparse matrix
    """
    vectorizer = build_vectorizer()

    print("[*] Fitting TF-IDF vectorizer …")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    vocab_size = len(vectorizer.vocabulary_)
    print(f"    ↳ Vocabulary size : {vocab_size:,}")
    print(f"    ↳ Train matrix    : {X_train_tfidf.shape}")
    print(f"    ↳ Test  matrix    : {X_test_tfidf.shape}")

    return vectorizer, X_train_tfidf, X_test_tfidf


def save_vectorizer(vectorizer: TfidfVectorizer) -> None:
    """Persist the fitted vectorizer to disk."""
    save_artifact(vectorizer, TFIDF_VECTORIZER_PATH)


def load_vectorizer() -> TfidfVectorizer:
    """Load the pre-fitted vectorizer from disk."""
    return load_artifact(TFIDF_VECTORIZER_PATH)


def transform_text(text: str, vectorizer: TfidfVectorizer) -> spmatrix:
    """Transform a single raw text string into a TF-IDF sparse matrix row."""
    from src.data_preprocessing import clean_text
    cleaned = clean_text(text)
    return vectorizer.transform([cleaned])
