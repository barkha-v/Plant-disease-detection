"""
utils.py
--------
Shared utility helpers used across the project:
  - directory management
  - model loading / saving
  - class-name persistence
  - logging configuration
  - reproducibility seeding
"""

import os
import json
import random
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Global random seed set to {seed}.")


# ─── Directory helpers ────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """Create *path* (and parents) if it does not exist. Returns the path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def project_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parent.parent


# ─── Class-name persistence ───────────────────────────────────────────────────

def save_class_names(class_names: list, save_path: str) -> None:
    """Persist class names to a JSON file so they survive restarts."""
    ensure_dir(str(Path(save_path).parent))
    with open(save_path, "w") as f:
        json.dump(class_names, f, indent=2)
    logger.info(f"Class names saved to {save_path}")


def load_class_names(load_path: str) -> list:
    """Load class names from a JSON file created by `save_class_names`."""
    if not Path(load_path).exists():
        raise FileNotFoundError(f"Class-names file not found: {load_path}")
    with open(load_path) as f:
        class_names = json.load(f)
    logger.info(f"Loaded {len(class_names)} class names from {load_path}")
    return class_names


# ─── Model I/O ────────────────────────────────────────────────────────────────

def save_model(model: tf.keras.Model, save_path: str) -> None:
    """Save a Keras model (auto-detects .h5 vs SavedModel format)."""
    ensure_dir(str(Path(save_path).parent))
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")


def load_model(model_path: str) -> tf.keras.Model:
    """Load a saved Keras model from *model_path*."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


# ─── GPU configuration ────────────────────────────────────────────────────────

def configure_gpu(memory_limit_mb: Optional[int] = None) -> None:
    """
    Enable GPU memory growth (prevents TF from grabbing all VRAM).
    Optionally cap memory to *memory_limit_mb* MB.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("No GPU detected – running on CPU.")
        return

    try:
        for gpu in gpus:
            if memory_limit_mb:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                )
            else:
                tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Configured {len(gpus)} GPU(s).")
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")


# ─── Training history helpers ─────────────────────────────────────────────────

def save_history(history: dict, save_path: str) -> None:
    """Serialize a Keras History.history dict to JSON."""
    ensure_dir(str(Path(save_path).parent))
    # Convert numpy floats to plain Python floats for JSON serialisation
    clean = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(save_path, "w") as f:
        json.dump(clean, f, indent=2)
    logger.info(f"Training history saved to {save_path}")


def load_history(load_path: str) -> dict:
    """Load a training history JSON previously saved by `save_history`."""
    with open(load_path) as f:
        return json.load(f)


# ─── Logging setup ────────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure the root logger with a console handler (and optional file handler).
    """
    fmt     = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler()]
    if log_file:
        ensure_dir(str(Path(log_file).parent))
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )


# ─── Misc ─────────────────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    """Convert *seconds* to a human-readable h/m/s string."""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")


def count_parameters(model: tf.keras.Model) -> dict:
    """Return trainable and non-trainable parameter counts."""
    trainable     = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    logger.info(
        f"Parameters → trainable: {trainable:,}  non-trainable: {non_trainable:,}"
    )
    return {"trainable": trainable, "non_trainable": non_trainable}
