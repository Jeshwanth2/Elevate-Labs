"""
evaluation.py — Model evaluation: accuracy, F1, confusion matrix, comparison.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc
)


# ─── Core Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, model_name: str = "Model") -> dict:
    """
    Compute accuracy, F1 (macro & weighted), precision, recall.

    Returns a dict keyed by metric name.
    """
    metrics = {
        "model": model_name,
        "accuracy":           round(accuracy_score(y_true, y_pred) * 100, 4),
        "f1_macro":           round(f1_score(y_true, y_pred, average="macro") * 100, 4),
        "f1_weighted":        round(f1_score(y_true, y_pred, average="weighted") * 100, 4),
        "precision_macro":    round(precision_score(y_true, y_pred, average="macro") * 100, 4),
        "recall_macro":       round(recall_score(y_true, y_pred, average="macro") * 100, 4),
    }

    print(f"\n{'─'*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<22}: {v:.4f}%")
    print(f"\n{classification_report(y_true, y_pred, target_names=['FAKE', 'REAL'])}")
    return metrics


def evaluate_all_models(models: dict, X_test, y_test) -> dict:
    """
    Evaluate every model in the dict and return all metrics.

    Returns
    -------
    dict  {model_name: metrics_dict}
    """
    all_metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        all_metrics[name] = compute_metrics(y_test, y_pred, name)
    return all_metrics


def select_best_model(models: dict, metrics: dict, criterion: str = "f1_weighted"):
    """
    Return the model with the highest score on `criterion`.

    Returns
    -------
    (best_name, best_model)
    """
    best_name = max(metrics, key=lambda k: metrics[k][criterion])
    print(f"\n[★] Best model by {criterion}: {best_name} "
          f"({metrics[best_name][criterion]:.4f}%)")
    return best_name, models[best_name]


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str = None):
    """Plot and optionally save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["FAKE", "REAL"],
        yticklabels=["FAKE", "REAL"],
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Confusion matrix saved → {save_path}")
    return fig


def plot_roc_curve(models: dict, X_test, y_test, save_path: str = None):
    """Plot ROC curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for (name, model), color in zip(models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_prob  = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] ROC curve saved → {save_path}")
    return fig


def plot_model_comparison(metrics: dict, save_path: str = None):
    """Bar chart comparing accuracy & F1 across models."""
    names  = list(metrics.keys())
    acc    = [metrics[n]["accuracy"]    for n in names]
    f1_w   = [metrics[n]["f1_weighted"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, acc,  width, label="Accuracy (%)",    color="#4C72B0")
    bars2 = ax.bar(x + width/2, f1_w, width, label="F1 Weighted (%)", color="#DD8452")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim([90, 101])
    ax.legend()

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[✓] Model comparison chart saved → {save_path}")
    return fig
