"""
preprocess.py
-------------
Data loading, preprocessing, augmentation, and splitting
for the PlantVillage plant-disease detection project.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)
CHANNELS   = 3
BATCH_SIZE = 32
SEED       = 42


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(dataset_dir: str) -> tuple[list, list, list]:
    """
    Walk *dataset_dir* and collect (image_path, class_name, class_index).

    Expected layout:
        dataset_dir/
            ClassName_A/
                img1.jpg  img2.jpg ...
            ClassName_B/
                ...

    Returns
    -------
    image_paths : list[str]
    labels      : list[str]   (human-readable class names)
    class_names : list[str]   (sorted, unique)
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No sub-directories (classes) found in {dataset_dir}")

    class_names  = [d.name for d in class_dirs]
    image_paths, labels = [], []
    valid_exts   = {".jpg", ".jpeg", ".png", ".bmp"}

    for cls_dir in class_dirs:
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in valid_exts]
        for f in files:
            image_paths.append(str(f))
            labels.append(cls_dir.name)

    logger.info(f"Found {len(image_paths)} images across {len(class_names)} classes.")
    return image_paths, labels, class_names


def build_dataframe(image_paths: list, labels: list) -> pd.DataFrame:
    """Return a DataFrame with columns ['image_path', 'label']."""
    df = pd.DataFrame({"image_path": image_paths, "label": labels})
    logger.info(f"Dataset distribution:\n{df['label'].value_counts()}")
    return df


# ─── Image helpers ────────────────────────────────────────────────────────────

def load_and_preprocess_image(
    image_path: str,
    target_size: tuple = IMAGE_SIZE,
    normalize: bool = True
) -> np.ndarray:
    """
    Load one image with OpenCV, resize, convert BGR→RGB, optionally normalize.

    Parameters
    ----------
    image_path  : str   – path to the image file
    target_size : tuple – (H, W) to resize to
    normalize   : bool  – divide pixel values by 255 when True

    Returns
    -------
    np.ndarray of shape (H, W, 3), dtype float32
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img = cv2.resize(img, target_size[::-1])          # cv2 wants (W, H)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    if normalize:
        img /= 255.0

    return img


def load_images_batch(
    image_paths: list,
    target_size: tuple = IMAGE_SIZE,
    normalize: bool = True
) -> np.ndarray:
    """
    Load and preprocess a list of images, returning a 4-D NumPy array.

    Returns
    -------
    np.ndarray of shape (N, H, W, 3)
    """
    images = []
    failed = 0
    for path in tqdm(image_paths, desc="Loading images"):
        try:
            images.append(load_and_preprocess_image(path, target_size, normalize))
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            failed += 1

    if failed:
        logger.warning(f"{failed} images could not be loaded.")

    return np.array(images, dtype=np.float32)


# ─── Label encoding ───────────────────────────────────────────────────────────

def encode_labels(labels: list, class_names: list) -> tuple[np.ndarray, LabelEncoder]:
    """
    Integer-encode string labels and return (encoded_array, fitted_encoder).
    """
    le = LabelEncoder()
    le.classes_ = np.array(class_names)
    encoded = le.transform(labels)
    return encoded, le


def one_hot_encode(encoded_labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot vectors."""
    return to_categorical(encoded_labels, num_classes=num_classes)


# ─── Train / Val / Test split ─────────────────────────────────────────────────

def split_dataset(
    image_paths: list,
    labels: list,
    test_size: float   = 0.15,
    val_size: float    = 0.15,
    random_state: int  = SEED
) -> dict:
    """
    Split paths + labels into train / validation / test sets (stratified).

    Returns
    -------
    dict with keys: train_paths, val_paths, test_paths,
                    train_labels, val_labels, test_labels
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train_val
    )

    logger.info(
        f"Split → train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}"
    )

    return {
        "train_paths":  X_train, "val_paths":  X_val, "test_paths":  X_test,
        "train_labels": y_train, "val_labels": y_val, "test_labels": y_test,
    }


# ─── Data generators ──────────────────────────────────────────────────────────

def create_data_generators(
    train_df:     pd.DataFrame,
    val_df:       pd.DataFrame,
    test_df:      pd.DataFrame,
    image_size:   tuple = IMAGE_SIZE,
    batch_size:   int   = BATCH_SIZE,
    augment:      bool  = True
) -> tuple:
    """
    Build Keras ImageDataGenerators for train / val / test splits.

    The DataFrames must have columns 'image_path' and 'label' (string).

    Returns
    -------
    (train_gen, val_gen, test_gen)
    """
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode="nearest",
            brightness_range=[0.8, 1.2],
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    h, w = image_size

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col="image_path", y_col="label",
        target_size=(h, w), batch_size=batch_size,
        class_mode="categorical", shuffle=True, seed=SEED,
    )
    val_gen = val_test_datagen.flow_from_dataframe(
        val_df, x_col="image_path", y_col="label",
        target_size=(h, w), batch_size=batch_size,
        class_mode="categorical", shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_dataframe(
        test_df, x_col="image_path", y_col="label",
        target_size=(h, w), batch_size=batch_size,
        class_mode="categorical", shuffle=False,
    )

    return train_gen, val_gen, test_gen


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_samples(
    image_paths: list,
    labels: list,
    n: int = 12,
    save_path: str = None
) -> None:
    """Plot a grid of sample images from the dataset."""
    indices = np.random.choice(len(image_paths), size=min(n, len(image_paths)), replace=False)
    cols = 4
    rows = (len(indices) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        img = load_and_preprocess_image(image_paths[idx], normalize=False)
        axes[i].imshow(img.astype(np.uint8))
        axes[i].set_title(labels[idx].replace("_", " "), fontsize=7)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Images from Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved sample grid to {save_path}")

    plt.show()
    plt.close()
