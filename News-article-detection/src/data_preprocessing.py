"""
data_preprocessing.py — Text cleaning and dataset preparation using NLTK.
"""
import re
import os
import string
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from src.config import (
    FAKE_CSV, TRUE_CSV, COMBINED_CSV,
    RANDOM_SEED, TEST_SIZE
)

# ─── NLTK resource downloads (idempotent) ────────────────────────────────────
def _download_nltk_resources():
    resources = [
        ("corpora/stopwords",     "stopwords"),
        ("corpora/wordnet",       "wordnet"),
        ("tokenizers/punkt",      "punkt"),
        ("corpora/omw-1.4",       "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[NLTK] Downloading '{name}' …")
            nltk.download(name, quiet=True)

_download_nltk_resources()

# ─── Globals ─────────────────────────────────────────────────────────────────
_STOP_WORDS  = set(stopwords.words("english"))
_LEMMATIZER  = WordNetLemmatizer()
_URL_PATTERN = re.compile(r"http\S+|www\.\S+")
_HTML_PATTERN = re.compile(r"<.*?>")


# ─── Core cleaner ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full text-cleaning pipeline:
      1. Lowercase
      2. Remove URLs
      3. Remove HTML tags
      4. Remove punctuation & digits
      5. Tokenize
      6. Remove stopwords
      7. Lemmatize
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _HTML_PATTERN.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)


# ─── Dataset Loading ─────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """
    Load Fake.csv and True.csv from the data/ directory and combine them.
    Expects columns: title, text, subject, date (standard Kaggle format).
    Labels: 0 = Fake, 1 = Real.
    """
    if not os.path.exists(FAKE_CSV) or not os.path.exists(TRUE_CSV):
        raise FileNotFoundError(
            "Dataset files not found!\n"
            f"  Expected: {FAKE_CSV}\n"
            f"           {TRUE_CSV}\n"
            "Please run data/download_data.py or download manually from Kaggle:\n"
            "  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )

    fake_df = pd.read_csv(FAKE_CSV)
    true_df = pd.read_csv(TRUE_CSV)

    fake_df["label"] = 0  # FAKE
    true_df["label"] = 1  # REAL

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Combine title + text into one rich feature
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    return df


# ─── Preprocessing Pipeline ───────────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply clean_text to the combined content column.
    Returns the dataframe with a new 'cleaned_content' column.
    """
    print("[*] Cleaning text (this may take a minute) …")
    df = df.copy()
    df["cleaned_content"] = df["content"].apply(clean_text)

    # Drop rows where cleaning results in empty string
    initial_len = len(df)
    df = df[df["cleaned_content"].str.strip().ne("")]
    dropped = initial_len - len(df)
    if dropped:
        print(f"    ↳ Dropped {dropped} empty rows after cleaning.")

    print(f"    ↳ Dataset ready: {len(df):,} samples "
          f"({df['label'].value_counts()[0]:,} fake / "
          f"{df['label'].value_counts()[1]:,} real)")
    return df


def split_data(df: pd.DataFrame):
    """
    Stratified train/test split.

    Returns
    -------
    X_train, X_test : pd.Series  (cleaned text)
    y_train, y_test : pd.Series  (0/1 labels)
    """
    X = df["cleaned_content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )
    print(f"[✓] Split → Train: {len(X_train):,} | Test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def save_combined(df: pd.DataFrame) -> None:
    """Persist the combined & cleaned dataset to data/combined_news.csv."""
    os.makedirs(os.path.dirname(COMBINED_CSV), exist_ok=True)
    df.to_csv(COMBINED_CSV, index=False)
    print(f"[✓] Combined CSV saved → {COMBINED_CSV}")


def load_preprocessed() -> pd.DataFrame:
    """Load the already-combined CSV if it exists."""
    if not os.path.exists(COMBINED_CSV):
        raise FileNotFoundError(
            f"Preprocessed CSV not found at {COMBINED_CSV}.\n"
            "Run the full pipeline first: python train.py"
        )
    return pd.read_csv(COMBINED_CSV)
