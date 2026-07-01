"""
main.py
-------
Orchestration script for the Plant Disease Detection project.

Workflow
--------
1. Load & preprocess the PlantVillage dataset
2. Build Keras data generators with augmentation
3. Train the custom CNN
4. Evaluate on the test set (metrics + plots)
5. Save model + class names for inference

Usage
-----
    python main.py                          # train from scratch
    python main.py --skip-train             # evaluate a saved model
    python main.py --epochs 30 --lr 0.0005 # custom hyper-parameters
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# ── Make the project root importable ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils      import setup_logging, set_seed, ensure_dir, save_class_names, load_class_names, load_model, load_history
from src.preprocess import load_dataset, build_dataframe, split_dataset, create_data_generators, visualize_samples
from src.train      import train_model
from src.evaluate   import run_full_evaluation

# ─── Configuration ────────────────────────────────────────────────────────────
CONFIG = dict(
    # Paths
    dataset_dir      = "archive/Train/Train",   # ← updated
    model_save_dir   = "saved_model",
    reports_dir      = "reports",
    images_dir       = "images",
    class_names_path = "saved_model/class_names.json",
    best_model_path  = "saved_model/best_model.keras",

    # Hyper-parameters
    image_size     = (224, 224),
    batch_size     = 32,
    epochs         = 50,
    learning_rate  = 1e-3,
    dropout_rate   = 0.40,
    l2_reg         = 1e-4,
    patience       = 10,

    # Splits
    test_size      = 0.15,
    val_size       = 0.15,

    # Misc
    seed           = 42,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plant Disease Detection – main script")
    p.add_argument("--dataset",    default=CONFIG["dataset_dir"],
                   help="Path to PlantVillage dataset root")
    p.add_argument("--epochs",     type=int,   default=CONFIG["epochs"])
    p.add_argument("--lr",         type=float, default=CONFIG["learning_rate"],
                   dest="learning_rate")
    p.add_argument("--batch-size", type=int,   default=CONFIG["batch_size"],
                   dest="batch_size")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training and evaluate a pre-saved model")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable  the data augmentation during the training")
    return p.parse_args()


def _dirs_exist():
    """Create all required output directories."""
    for d in [CONFIG["model_save_dir"], CONFIG["reports_dir"], CONFIG["images_dir"]]:
        ensure_dir(d)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    setup_logging(log_level="INFO", log_file="reports/run.log")
    logger = logging.getLogger(__name__)

    args = parse_args()
    set_seed(CONFIG["seed"])
    _dirs_exist()

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  PLANT DISEASE DETECTION – TRAINING PIPELINE")
    logger.info("=" * 60)

    dataset_dir = args.dataset
    if not Path(dataset_dir).exists():
        logger.error(
            f"Dataset not found at '{dataset_dir}'.\n"
            "Download PlantVillage via Kaggle:\n"
            "  kaggle datasets download -d abdallahalidev/plantvillage-dataset\n"
            "  unzip plantvillage-dataset.zip -d dataset/"
        )
        sys.exit(1)

    image_paths, labels, class_names = load_dataset(dataset_dir)

    # ── 2. Save class names ──────────────────────────────────────────────────
    save_class_names(class_names, CONFIG["class_names_path"])
    logger.info(f"Classes ({len(class_names)}): {class_names[:5]} … ")

    # ── 3. Visualise samples ─────────────────────────────────────────────────
    sample_path = os.path.join(CONFIG["images_dir"], "dataset_samples.png")
    try:
        visualize_samples(image_paths, labels, n=12, save_path=sample_path)
    except Exception as e:
        logger.warning(f"Could not visualise samples: {e}")

    # ── 4. Split ─────────────────────────────────────────────────────────────
    splits = split_dataset(
        image_paths, labels,
        test_size  = CONFIG["test_size"],
        val_size   = CONFIG["val_size"],
        random_state = CONFIG["seed"],
    )

    import pandas as pd
    train_df = pd.DataFrame({"image_path": splits["train_paths"],
                              "label":      splits["train_labels"]})
    val_df   = pd.DataFrame({"image_path": splits["val_paths"],
                              "label":      splits["val_labels"]})
    test_df  = pd.DataFrame({"image_path": splits["test_paths"],
                              "label":      splits["test_labels"]})

    # ── 5. Data generators ───────────────────────────────────────────────────
    train_gen, val_gen, test_gen = create_data_generators(
        train_df, val_df, test_df,
        image_size  = CONFIG["image_size"],
        batch_size  = args.batch_size,
        augment     = not args.no_augment,
    )

    num_classes = len(class_names)

    # ── 6. Train or load ─────────────────────────────────────────────────────
    if args.skip_train:
        logger.info("--skip-train set: loading pre-trained model …")
        if not Path(CONFIG["best_model_path"]).exists():
            logger.error(f"No saved model found at {CONFIG['best_model_path']}")
            sys.exit(1)
        model   = load_model(CONFIG["best_model_path"])
        history = load_history(os.path.join(CONFIG["reports_dir"],
                                            "training_history.json"))
    else:
        model, history = train_model(
            train_generator = train_gen,
            val_generator   = val_gen,
            num_classes     = num_classes,
            model_save_dir  = CONFIG["model_save_dir"],
            reports_dir     = CONFIG["reports_dir"],
            epochs          = args.epochs,
            learning_rate   = args.learning_rate,
            dropout_rate    = CONFIG["dropout_rate"],
            l2_reg          = CONFIG["l2_reg"],
            patience        = CONFIG["patience"],
        )

    # ── 7. Evaluate ──────────────────────────────────────────────────────────
    results = run_full_evaluation(
        model        = model,
        test_generator = test_gen,
        class_names  = class_names,
        history      = history,
        image_paths  = splits["test_paths"],
        true_labels  = splits["test_labels"],
        reports_dir  = CONFIG["reports_dir"],
        images_dir   = CONFIG["images_dir"],
    )

    # ── 8. Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Accuracy  : {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    logger.info(f"  Precision : {results['precision']:.4f}")
    logger.info(f"  Recall    : {results['recall']:.4f}")
    logger.info(f"  F1-score  : {results['f1']:.4f}")
    logger.info("=" * 60)
    logger.info("All plots and reports saved. Pipeline complete.")


if __name__ == "__main__":
    main()
