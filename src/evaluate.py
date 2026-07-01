"""
evaluate.py
-----------
Model evaluation: accuracy, classification report, confusion matrix,
training curves, sample predictions, misclassification grid, and ROC curves.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from itertools import cycle

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from src.utils import ensure_dir

logger = logging.getLogger(__name__)


# ─── Prediction helpers ───────────────────────────────────────────────────────

def get_predictions(
    model:          tf.keras.Model,
    data_generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on every batch of *data_generator*.

    Returns
    -------
    y_true    : (N,)   integer true class indices
    y_pred    : (N,)   integer predicted class indices
    y_proba   : (N, C) softmax probabilities
    """
    data_generator.reset()
    y_proba = model.predict(data_generator, verbose=1)
    y_pred  = np.argmax(y_proba, axis=1)

    # ── Build ground-truth vector ────────────────────────────────────────────
    # flow_from_dataframe gives us one-hot labels via the generator
    n_batches = len(data_generator)
    y_true = []
    data_generator.reset()
    for i in range(n_batches):
        _, batch_labels = data_generator[i]
        y_true.extend(np.argmax(batch_labels, axis=1))

    y_true = np.array(y_true[: len(y_pred)])   # trim to exact length
    return y_true, y_pred, y_proba


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list,
) -> dict:
    """
    Compute accuracy, macro precision/recall/F1, and per-class report.

    Returns a dict with keys: accuracy, precision, recall, f1, report_str, report_df
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    report_str = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose().round(4)

    logger.info(
        f"\nEvaluation Metrics\n"
        f"  Accuracy  : {acc:.4f}\n"
        f"  Precision : {prec:.4f}\n"
        f"  Recall    : {rec:.4f}\n"
        f"  F1-score  : {f1:.4f}\n"
    )
    logger.info(f"\nClassification Report:\n{report_str}")

    return dict(
        accuracy=acc, precision=prec, recall=rec, f1=f1,
        report_str=report_str, report_df=report_df,
    )


# ─── Visualisation functions ──────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list,
    save_path:   str = None,
    normalize:   bool = True,
) -> None:
    """Plot (and optionally save) a heat-map confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig_size = max(10, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    sns.heatmap(
        cm,
        annot=n <= 30,           # skip cell text for large matrices
        fmt=".2f" if normalize else "d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Confusion Matrix" + (" (normalised)" if normalize else ""), fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    _save_or_show(fig, save_path, "confusion_matrix.png")


def plot_training_curves(
    history: dict,
    save_path: str = None,
) -> None:
    """Plot accuracy and loss training / validation curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history["accuracy"],     label="Train Accuracy",     color="#2196F3")
    axes[0].plot(history["val_accuracy"], label="Val Accuracy",       color="#FF5722", linestyle="--")
    axes[0].set_title("Model Accuracy", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history["loss"],     label="Train Loss",     color="#4CAF50")
    axes[1].plot(history["val_loss"], label="Val Loss",       color="#F44336", linestyle="--")
    axes[1].set_title("Model Loss", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path, "training_curves.png")


def plot_sample_predictions(
    model:       tf.keras.Model,
    image_paths: list,
    true_labels: list,
    class_names: list,
    n:           int = 16,
    save_path:   str = None,
) -> None:
    """
    Display a grid of *n* sample images with their true vs predicted labels.
    Green title = correct, red title = wrong.
    """
    import cv2

    indices = np.random.choice(len(image_paths), size=min(n, len(image_paths)), replace=False)
    cols = 4
    rows = (len(indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img_raw  = cv2.imread(image_paths[idx])
        img_raw  = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_disp = cv2.resize(img_raw, (224, 224))

        img_in   = img_disp.astype(np.float32) / 255.0
        img_in   = np.expand_dims(img_in, 0)

        proba    = model.predict(img_in, verbose=0)[0]
        pred_idx = np.argmax(proba)
        pred_cls = class_names[pred_idx]
        conf     = proba[pred_idx] * 100

        true_cls = true_labels[idx]
        color    = "green" if pred_cls == true_cls else "red"

        axes[i].imshow(img_disp)
        axes[i].set_title(
            f"True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)",
            color=color, fontsize=7,
        )
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path, "sample_predictions.png")


def plot_misclassified(
    model:       tf.keras.Model,
    image_paths: list,
    true_labels: list,
    class_names: list,
    n:           int = 12,
    save_path:   str = None,
) -> None:
    """
    Collect misclassified images and display them in a grid with
    true vs predicted labels.
    """
    import cv2

    misclassified = []
    for path, true_lbl in zip(image_paths, true_labels):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        inp = np.expand_dims(img.astype(np.float32) / 255.0, 0)
        proba    = model.predict(inp, verbose=0)[0]
        pred_idx = np.argmax(proba)
        pred_lbl = class_names[pred_idx]
        if pred_lbl != true_lbl:
            misclassified.append((img, true_lbl, pred_lbl, proba[pred_idx]))

        if len(misclassified) >= n:
            break

    if not misclassified:
        logger.info("No misclassified images found in the sample.")
        return

    cols = 4
    rows = (len(misclassified) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()

    for i, (img, true_lbl, pred_lbl, conf) in enumerate(misclassified):
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {true_lbl}\nPred: {pred_lbl} ({conf*100:.1f}%)",
            color="red", fontsize=7,
        )
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Misclassified Images", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path, "misclassified.png")


def plot_roc_curves(
    y_true:      np.ndarray,
    y_proba:     np.ndarray,
    class_names: list,
    save_path:   str = None,
    max_classes: int = 10,
) -> None:
    """
    Plot One-vs-Rest ROC curves for up to *max_classes* classes.
    Macro-average AUC is annotated in the legend.
    """
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=np.arange(n_classes))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i]         = auc(fpr[i], tpr[i])

    # Macro-average
    all_fpr  = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr   /= n_classes
    macro_auc   = auc(all_fpr, mean_tpr)

    # Plot top-N classes by AUC
    top_classes = sorted(range(n_classes), key=lambda i: roc_auc[i], reverse=True)[:max_classes]
    colors      = plt.cm.tab10(np.linspace(0, 1, len(top_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(all_fpr, mean_tpr, color="navy", lw=2.5, linestyle="--",
            label=f"Macro-avg ROC (AUC = {macro_auc:.3f})")

    for cls_idx, color in zip(top_classes, colors):
        ax.plot(fpr[cls_idx], tpr[cls_idx], lw=1.2, color=color,
                label=f"{class_names[cls_idx]} (AUC={roc_auc[cls_idx]:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves – One-vs-Rest (Top Classes)", fontsize=13)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    _save_or_show(fig, save_path, "roc_curves.png")


# ─── Full evaluation pipeline ─────────────────────────────────────────────────

def run_full_evaluation(
    model:           tf.keras.Model,
    test_generator,
    class_names:     list,
    history:         dict,
    image_paths:     list,
    true_labels:     list,
    reports_dir:     str = "reports",
    images_dir:      str = "images",
) -> dict:
    """
    End-to-end evaluation: compute metrics, save all plots, return results dict.
    """
    ensure_dir(reports_dir)
    ensure_dir(images_dir)

    # ── Predictions ──────────────────────────────────────────────────────────
    logger.info("Running inference on test set …")
    y_true, y_pred, y_proba = get_predictions(model, test_generator)

    # ── Metrics ──────────────────────────────────────────────────────────────
    results = compute_metrics(y_true, y_pred, class_names)

    # ── Save classification report ───────────────────────────────────────────
    report_csv = os.path.join(reports_dir, "classification_report.csv")
    results["report_df"].to_csv(report_csv)
    logger.info(f"Classification report saved to {report_csv}")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_training_curves(
        history,
        save_path=os.path.join(images_dir, "training_curves.png"),
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(images_dir, "confusion_matrix.png"),
    )
    plot_sample_predictions(
        model, image_paths, true_labels, class_names,
        save_path=os.path.join(images_dir, "sample_predictions.png"),
    )
    plot_misclassified(
        model, image_paths, true_labels, class_names,
        save_path=os.path.join(images_dir, "misclassified.png"),
    )
    plot_roc_curves(
        y_true, y_proba, class_names,
        save_path=os.path.join(images_dir, "roc_curves.png"),
    )

    logger.info("Evaluation complete. All plots saved.")
    return results


# ─── Internal helper ──────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: str, default_name: str) -> None:
    """Save figure to *save_path* if provided, otherwise display it."""
    if save_path:
        ensure_dir(str(Path(save_path).parent))
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)
