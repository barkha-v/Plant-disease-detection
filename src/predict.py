"""
predict.py
----------
Stand-alone prediction script.

Usage
-----
    python -m src.predict --image path/to/leaf.jpg \
                          --model saved_model/best_model.keras \
                          --classes saved_model/class_names.json

Or import `predict_single` / `predict_batch` from this module.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from src.utils import load_model, load_class_names, ensure_dir

logger = logging.getLogger(__name__)

# ─── Image loading ────────────────────────────────────────────────────────────

def preprocess_for_inference(
    image_path: str,
    target_size: tuple = (224, 224),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an image, resize, normalise, and return
    (display_image: HxWx3 uint8, model_input: 1xHxWx3 float32).

    Raises
    ------
    FileNotFoundError  if the image cannot be read.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"OpenCV could not decode: {image_path}")

    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_disp = cv2.resize(img_rgb, target_size)
    img_norm = img_disp.astype(np.float32) / 255.0
    img_in   = np.expand_dims(img_norm, axis=0)     # (1, H, W, 3)

    return img_disp, img_in


# ─── Single-image prediction ──────────────────────────────────────────────────

def predict_single(
    image_path:  str,
    model:       tf.keras.Model,
    class_names: list,
    top_k:       int = 5,
) -> dict:
    """
    Predict the disease class for a single leaf image.

    Parameters
    ----------
    image_path  : path to the leaf image
    model       : loaded Keras model
    class_names : ordered list of class name strings
    top_k       : number of top predictions to return

    Returns
    -------
    dict with keys:
        predicted_class  (str)
        confidence       (float, 0–1)
        top_k            (list of {"class": str, "confidence": float})
        probabilities    (np.ndarray, shape (n_classes,))
    """
    img_disp, img_in = preprocess_for_inference(image_path)

    proba    = model.predict(img_in, verbose=0)[0]   # (n_classes,)
    pred_idx = int(np.argmax(proba))
    pred_cls = class_names[pred_idx]
    conf     = float(proba[pred_idx])

    top_indices = np.argsort(proba)[::-1][:top_k]
    top_preds   = [
        {"class": class_names[i], "confidence": round(float(proba[i]), 4)}
        for i in top_indices
    ]

    result = dict(
        predicted_class = pred_cls,
        confidence      = round(conf, 4),
        top_k           = top_preds,
        probabilities   = proba,
    )

    logger.info(
        f"Image: {image_path}\n"
        f"  → Predicted: {pred_cls}  (confidence: {conf*100:.2f}%)\n"
        f"  → Top-{top_k} predictions:"
    )
    for rank, p in enumerate(top_preds, start=1):
        logger.info(f"       {rank}. {p['class']:40s}  {p['confidence']*100:.2f}%")

    return result


# ─── Batch prediction ─────────────────────────────────────────────────────────

def predict_batch(
    image_paths: list,
    model:       tf.keras.Model,
    class_names: list,
    batch_size:  int = 32,
) -> list[dict]:
    """
    Run inference on a list of images and return a list of result dicts.
    Each dict contains: image_path, predicted_class, confidence.
    """
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        inputs = []
        for p in batch_paths:
            try:
                _, img_in = preprocess_for_inference(p)
                inputs.append(img_in[0])
            except Exception as e:
                logger.warning(f"Skipping {p}: {e}")
                inputs.append(np.zeros((224, 224, 3), dtype=np.float32))

        batch_np = np.stack(inputs, axis=0)             # (B, H, W, 3)
        probas   = model.predict(batch_np, verbose=0)   # (B, n_classes)

        for path, proba in zip(batch_paths, probas):
            pred_idx = int(np.argmax(proba))
            results.append({
                "image_path":     path,
                "predicted_class": class_names[pred_idx],
                "confidence":      round(float(proba[pred_idx]), 4),
            })

    return results


# ─── Visualisation ────────────────────────────────────────────────────────────

def display_prediction(
    image_path:  str,
    result:      dict,
    save_path:   str = None,
) -> None:
    """
    Show the leaf image alongside a horizontal confidence bar chart.
    """
    img_disp, _ = preprocess_for_inference(image_path)
    top_preds    = result["top_k"]

    classes = [p["class"] for p in top_preds][::-1]
    confs   = [p["confidence"] * 100 for p in top_preds][::-1]
    colors  = ["#4CAF50" if c == result["predicted_class"] else "#90A4AE"
               for c in classes]

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.6])

    # Image
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img_disp)
    ax0.set_title(
        f"Predicted: {result['predicted_class']}\n"
        f"Confidence: {result['confidence']*100:.2f}%",
        fontsize=12, fontweight="bold",
        color="#4CAF50" if result["confidence"] > 0.7 else "#FF5722",
    )
    ax0.axis("off")

    # Bar chart
    ax1 = fig.add_subplot(gs[1])
    bars = ax1.barh(classes, confs, color=colors[::-1], edgecolor="white", height=0.6)
    ax1.set_xlim(0, 105)
    ax1.set_xlabel("Confidence (%)", fontsize=11)
    ax1.set_title("Top-K Predictions", fontsize=12, fontweight="bold")
    for bar, conf in zip(bars, confs[::-1]):
        ax1.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{conf:.1f}%", va="center", fontsize=9,
        )
    ax1.grid(axis="x", alpha=0.3)

    plt.suptitle("Plant Disease Detection", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        ensure_dir(str(Path(save_path).parent))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Prediction visualisation saved to {save_path}")

    plt.show()
    plt.close()


import matplotlib.gridspec as gridspec   # already imported above; no harm repeating


# ─── CLI entry-point ──────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Plant Disease Prediction")
    parser.add_argument("--image",   required=True,
                        help="Path to the leaf image")
    parser.add_argument("--model",   default="saved_model/best_model.keras",
                        help="Path to the saved Keras model")
    parser.add_argument("--classes", default="saved_model/class_names.json",
                        help="Path to class_names.json")
    parser.add_argument("--top_k",   type=int, default=5,
                        help="Number of top predictions to display")
    parser.add_argument("--save",    default=None,
                        help="Optional path to save the visualisation PNG")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args         = parse_args(argv)
    model        = load_model(args.model)
    class_names  = load_class_names(args.classes)

    result = predict_single(args.image, model, class_names, top_k=args.top_k)

    print("\n" + "=" * 55)
    print(f"  PREDICTION RESULT")
    print("=" * 55)
    print(f"  Image            : {args.image}")
    print(f"  Predicted class  : {result['predicted_class']}")
    print(f"  Confidence       : {result['confidence']*100:.2f}%")
    print("\n  Top-K Predictions:")
    for rank, p in enumerate(result["top_k"], start=1):
        bar = "█" * int(p["confidence"] * 30)
        print(f"    {rank}. {p['class']:<35} {p['confidence']*100:5.2f}%  {bar}")
    print("=" * 55)

    display_prediction(args.image, result, save_path=args.save)


if __name__ == "__main__":
    main()
