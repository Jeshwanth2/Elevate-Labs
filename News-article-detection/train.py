"""
train.py — End-to-end training pipeline.
Run: python train.py
"""
import os
import sys
import json
import time

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import (
    load_raw_data, preprocess_data, split_data, save_combined
)
from src.feature_engineering import (
    fit_transform_vectorizer, save_vectorizer
)
from src.model_training import (
    train_all_models, save_models, save_best_model
)
from src.evaluation import (
    evaluate_all_models, select_best_model,
    plot_confusion_matrix, plot_roc_curve, plot_model_comparison
)
from src.utils import save_metrics
from src.config import MODELS_DIR, METRICS_PATH


def main():
    start = time.time()
    print("=" * 60)
    print("  News Article Classification — Training Pipeline")
    print("=" * 60)

    # ── Step 1: Load ──────────────────────────────────────────────
    print("\n[1/6] Loading dataset …")
    df = load_raw_data()
    print(f"      Shape: {df.shape}")

    # ── Step 2: Preprocess ────────────────────────────────────────
    print("\n[2/6] Preprocessing text …")
    df = preprocess_data(df)
    save_combined(df)

    # ── Step 3: Split ─────────────────────────────────────────────
    print("\n[3/6] Splitting data …")
    X_train, X_test, y_train, y_test = split_data(df)

    # ── Step 4: TF-IDF ────────────────────────────────────────────
    print("\n[4/6] Vectorising with TF-IDF …")
    vectorizer, X_train_tfidf, X_test_tfidf = fit_transform_vectorizer(X_train, X_test)
    save_vectorizer(vectorizer)

    # ── Step 5: Train ─────────────────────────────────────────────
    print("\n[5/6] Training models …")
    models = train_all_models(X_train_tfidf, y_train)
    save_models(models)

    # ── Step 6: Evaluate ──────────────────────────────────────────
    print("\n[6/6] Evaluating models …")
    all_metrics = evaluate_all_models(models, X_test_tfidf, y_test)
    best_name, best_model = select_best_model(models, all_metrics)
    save_best_model(best_model, best_name)

    # Persist metrics for app consumption
    all_metrics["best_model"] = best_name
    save_metrics(all_metrics, METRICS_PATH)

    # Save plots
    plots_dir = os.path.join(MODELS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for name, model in models.items():
        y_pred = model.predict(X_test_tfidf)
        plot_confusion_matrix(
            y_test, y_pred, name,
            save_path=os.path.join(plots_dir, f"cm_{name.replace(' ', '_').lower()}.png")
        )

    plot_roc_curve(
        models, X_test_tfidf, y_test,
        save_path=os.path.join(plots_dir, "roc_curves.png")
    )
    plot_model_comparison(
        all_metrics,
        save_path=os.path.join(plots_dir, "model_comparison.png")
    )

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print(f"  Best Model    : {best_name}")
    print(f"  F1 (weighted) : {all_metrics[best_name]['f1_weighted']:.4f}%")
    print(f"  Accuracy      : {all_metrics[best_name]['accuracy']:.4f}%")
    print(f"  Models saved  → {MODELS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
