"""
train.py
--------
Build, compile, and train the custom CNN model for plant-disease detection.
"""

import os
import time
import logging
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger,
)

from src.utils import (
    ensure_dir,
    save_history,
    count_parameters,
    format_duration,
    configure_gpu,
)

logger = logging.getLogger(__name__)

# ─── Default hyper-parameters ─────────────────────────────────────────────────
DEFAULTS = dict(
    image_size   = (224, 224, 3),
    learning_rate= 1e-3,
    epochs       = 50,
    batch_size   = 32,
    dropout_rate = 0.4,
    l2_reg       = 1e-4,
)


# ─── Model architecture ───────────────────────────────────────────────────────

def build_cnn_model(
    num_classes:   int,
    input_shape:   tuple  = (224, 224, 3),
    dropout_rate:  float  = 0.4,
    l2_reg:        float  = 1e-4,
) -> tf.keras.Model:
    """
    Custom CNN for plant-disease image classification.

    Architecture
    ────────────
    3 × (Conv2D → BN → Conv2D → BN → MaxPool → Dropout)
    followed by a GlobalAveragePool head with two Dense layers.

    Parameters
    ----------
    num_classes  : number of disease categories (output units)
    input_shape  : (H, W, C)
    dropout_rate : fraction of units to drop after each pool
    l2_reg       : L2 kernel-regularisation strength

    Returns
    -------
    Compiled tf.keras.Model (not yet fitted)
    """
    reg = regularizers.l2(l2_reg)

    inputs = layers.Input(shape=input_shape, name="input_layer")

    # ── Block 1 ─────────────────────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)

    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop1")(x)

    # ── Block 2 ─────────────────────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)

    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(dropout_rate * 0.75, name="drop2")(x)

    # ── Block 3 ─────────────────────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)

    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Dropout(dropout_rate, name="drop3")(x)

    # ── Block 4 ─────────────────────────────────────────────────────────────
    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)

    x = layers.Conv2D(256, (3, 3), padding="same", kernel_regularizer=reg,
                      name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.Activation("relu", name="relu4_2")(x)

    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = layers.Dropout(dropout_rate, name="drop4")(x)

    # ── Classification head ──────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    x = layers.Dense(512, kernel_regularizer=reg, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.Activation("relu", name="relu_dense1")(x)
    x = layers.Dropout(dropout_rate, name="drop_dense1")(x)

    x = layers.Dense(256, kernel_regularizer=reg, name="dense2")(x)
    x = layers.BatchNormalization(name="bn_dense2")(x)
    x = layers.Activation("relu", name="relu_dense2")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="drop_dense2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs, outputs, name="PlantDiseaseNet")
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Compile *model* with Adam + categorical cross-entropy + accuracy.
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(f"Model compiled. LR={learning_rate}")
    return model


# ─── Callbacks ────────────────────────────────────────────────────────────────

def build_callbacks(
    checkpoint_path: str,
    log_dir:         str,
    csv_log_path:    str,
    patience:        int = 10,
) -> list:
    """
    Return a list of Keras callbacks for training:
      - EarlyStopping
      - ModelCheckpoint (saves best model only)
      - ReduceLROnPlateau
      - TensorBoard
      - CSVLogger
    """
    ensure_dir(str(Path(checkpoint_path).parent))
    ensure_dir(log_dir)
    ensure_dir(str(Path(csv_log_path).parent))

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )

    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
    )

    csv_logger = CSVLogger(csv_log_path, append=False)

    return [early_stop, checkpoint, reduce_lr, tensorboard, csv_logger]


# ─── Training entry-point ─────────────────────────────────────────────────────

def train_model(
    train_generator,
    val_generator,
    num_classes:    int,
    model_save_dir: str = "saved_model",
    reports_dir:    str = "reports",
    epochs:         int = 50,
    learning_rate:  float = 1e-3,
    dropout_rate:   float = 0.4,
    l2_reg:         float = 1e-4,
    patience:       int   = 10,
) -> tuple[tf.keras.Model, dict]:
    """
    Build, compile, and train the CNN.

    Parameters
    ----------
    train_generator : Keras data generator for training
    val_generator   : Keras data generator for validation
    num_classes     : total number of disease classes
    model_save_dir  : directory for .keras checkpoint
    reports_dir     : directory for CSV log
    epochs          : maximum training epochs
    learning_rate   : initial Adam learning rate
    dropout_rate    : Dropout fraction
    l2_reg          : L2 regularisation weight
    patience        : EarlyStopping patience

    Returns
    -------
    (trained_model, history_dict)
    """
    configure_gpu()

    # ── Build ────────────────────────────────────────────────────────────────
    model = build_cnn_model(
        num_classes  = num_classes,
        dropout_rate = dropout_rate,
        l2_reg       = l2_reg,
    )
    model = compile_model(model, learning_rate=learning_rate)

    logger.info(model.summary())
    count_parameters(model)

    # ── Paths ────────────────────────────────────────────────────────────────
    checkpoint_path = os.path.join(model_save_dir, "best_model.keras")
    tensorboard_dir = os.path.join(reports_dir, "tensorboard_logs")
    csv_log_path    = os.path.join(reports_dir, "training_log.csv")
    history_path    = os.path.join(reports_dir, "training_history.json")

    callbacks = build_callbacks(
        checkpoint_path = checkpoint_path,
        log_dir         = tensorboard_dir,
        csv_log_path    = csv_log_path,
        patience        = patience,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info(f"Training for up to {epochs} epochs …")
    t0 = time.time()

    history = model.fit(
        train_generator,
        validation_data = val_generator,
        epochs          = epochs,
        callbacks       = callbacks,
        verbose         = 1,
    )

    elapsed = time.time() - t0
    logger.info(f"Training finished in {format_duration(elapsed)}.")

    # ── Persist history ──────────────────────────────────────────────────────
    save_history(history.history, history_path)

    # ── Save final model ─────────────────────────────────────────────────────
    final_path = os.path.join(model_save_dir, "final_model.keras")
    model.save(final_path)
    logger.info(f"Final model saved to {final_path}")

    return model, history.history
