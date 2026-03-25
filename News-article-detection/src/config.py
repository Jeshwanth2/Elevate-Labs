"""
config.py — Project-wide configuration: paths, hyperparameters, constants.
"""
import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# ─── Data Files ───────────────────────────────────────────────────────────────
FAKE_CSV  = os.path.join(DATA_DIR, "Fake.csv")
TRUE_CSV  = os.path.join(DATA_DIR, "True.csv")
COMBINED_CSV = os.path.join(DATA_DIR, "combined_news.csv")

# ─── Model Artifacts ──────────────────────────────────────────────────────────
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LR_MODEL_PATH         = os.path.join(MODELS_DIR, "logistic_regression.pkl")
NB_MODEL_PATH         = os.path.join(MODELS_DIR, "naive_bayes.pkl")
BEST_MODEL_PATH       = os.path.join(MODELS_DIR, "best_model.pkl")
METRICS_PATH          = os.path.join(MODELS_DIR, "metrics.json")

# ─── Preprocessing ────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
TEST_SIZE     = 0.20      # 80/20 train-test split
MAX_FEATURES  = 50_000    # TF-IDF vocabulary size
NGRAM_RANGE   = (1, 2)    # unigrams + bigrams
MIN_DF        = 2         # ignore terms that appear only once

# ─── Model Hyperparameters ────────────────────────────────────────────────────
LR_C        = 5.0         # Logistic Regression regularisation
LR_MAX_ITER = 1000
NB_ALPHA    = 0.1         # Laplace smoothing

# ─── Labels ───────────────────────────────────────────────────────────────────
LABEL_MAP   = {0: "FAKE", 1: "REAL"}
LABEL_COLORS = {"FAKE": "#FF4B4B", "REAL": "#21C55D"}
