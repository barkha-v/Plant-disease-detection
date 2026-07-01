# 🌿 Plant Disease Detection using Deep Learning

A production-quality image classification system that detects plant diseases from leaf photographs using a custom Convolutional Neural Network trained on the PlantVillage dataset.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Model Architecture](#model-architecture)
5. [Installation](#installation)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Prediction](#prediction)
9. [Results](#results)
10. [Future Improvements](#future-improvements)
11. [Resume Bullets](#resume-bullets)
12. [Interview Q&A](#interview-qa)

---

## Project Overview

Early detection of plant diseases is critical for food security and agricultural productivity. This project implements an end-to-end deep learning pipeline that:

- Classifies **38 disease categories** (plus healthy classes) from RGB leaf images
- Achieves **≥ 90 % test accuracy** on the PlantVillage benchmark
- Exposes a clean **prediction API** (`predict_single`, `predict_batch`) for deployment
- Produces comprehensive **evaluation reports** (confusion matrix, ROC curves, classification report)

---

## Dataset

**PlantVillage** is a publicly available dataset of ~54 000 leaf images covering 14 crop species and 26 disease classes (plus healthy images).

| Stat          | Value            |
|---------------|------------------|
| Total images  | ~54 000          |
| Classes       | 38               |
| Crop species  | 14               |
| Image size    | 256 × 256 (raw)  |
| Model input   | 224 × 224        |

### Download

```bash
# Option A – Kaggle CLI
pip install kaggle
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d dataset/

# Option B – manual
# Visit https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
# Download and extract into dataset/PlantVillage/
```

Expected layout after extraction:

```
dataset/
└── PlantVillage/
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── Apple___Cedar_apple_rust/
    ├── Apple___healthy/
    ├── Blueberry___healthy/
    ...   (38 folders total)
```

---

## Project Structure

```
Plant_Disease_Detection/
│
├── dataset/                     # PlantVillage images (not committed to git)
│   └── PlantVillage/
│
├── notebooks/
│   └── plant_disease_detection.ipynb   # Interactive exploration
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py            # Data loading, augmentation, generators
│   ├── train.py                 # CNN architecture, compilation, training loop
│   ├── evaluate.py              # Metrics, confusion matrix, ROC, plots
│   ├── predict.py               # Inference (single image & batch)
│   └── utils.py                 # Shared helpers (seed, logging, I/O)
│
├── saved_model/
│   ├── best_model.keras         # Best checkpoint (val accuracy)
│   ├── final_model.keras        # Final epoch model
│   └── class_names.json         # Ordered class list for inference
│
├── reports/
│   ├── training_history.json
│   ├── training_log.csv
│   ├── classification_report.csv
│   └── run.log
│
├── images/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── sample_predictions.png
│   ├── misclassified.png
│   └── dataset_samples.png
│
├── requirements.txt
├── README.md
└── main.py                      # Pipeline entry-point
```

---

## Model Architecture

**PlantDiseaseNet** – a custom 4-block CNN with a GlobalAveragePooling head.

```
Input (224 × 224 × 3)
│
├── Block 1: Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.20)
├── Block 2: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.30)
├── Block 3: Conv2D(128)→ BN → ReLU → Conv2D(128)→ BN → ReLU → MaxPool → Dropout(0.40)
├── Block 4: Conv2D(256)→ BN → ReLU → Conv2D(256)→ BN → ReLU → MaxPool → Dropout(0.40)
│
├── GlobalAveragePooling2D
├── Dense(512) → BN → ReLU → Dropout(0.40)
├── Dense(256) → BN → ReLU → Dropout(0.20)
└── Dense(38, softmax)
```

| Component        | Purpose                                      |
|-----------------|----------------------------------------------|
| BatchNorm        | Stabilises training, allows higher LR        |
| MaxPooling       | Spatial downsampling, translation invariance |
| Dropout          | Regularisation, prevents overfitting         |
| GlobalAvgPool    | Parameter-efficient alternative to Flatten   |
| L2 regularisation| Weight decay on Conv + Dense kernels         |

**Optimiser:** Adam (lr=1e-3, β₁=0.9, β₂=0.999)  
**Loss:** Categorical Cross-Entropy  
**Metrics:** Accuracy

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourname/plant-disease-detection.git
cd plant-disease-detection

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset (see Dataset section above)
```

**System requirements:**  
- Python 3.9 – 3.11  
- 8 GB RAM minimum (16 GB recommended)  
- GPU optional but strongly recommended (CUDA 11.8 + cuDNN 8.6)

---

## Training

```bash
# Full training run (up to 50 epochs, EarlyStopping active)
 

# Custom hyper-parameters
python main.py --epochs 30 --lr 0.0005 --batch-size 64

# Disable augmentation
python main.py --no-augment

# Point to a different dataset directory
python main.py --dataset /path/to/PlantVillage
```

Training artefacts are written to:

| File                                  | Contents                         |
|--------------------------------------|----------------------------------|
| `saved_model/best_model.keras`       | Best checkpoint (val accuracy)   |
| `saved_model/final_model.keras`      | Final epoch                      |
| `saved_model/class_names.json`       | Class index → name mapping       |
| `reports/training_history.json`      | Per-epoch loss/accuracy          |
| `reports/training_log.csv`           | CSV mirror of history            |
| `reports/tensorboard_logs/`          | TensorBoard event files          |

```bash
# Launch TensorBoard
tensorboard --logdir reports/tensorboard_logs
```

---

## Evaluation

```bash
# Evaluate a saved model (skip re-training)
python main.py --skip-train
```

All plots are saved to `images/` and all CSVs to `reports/`:

| Output                         | Description                                  |
|-------------------------------|----------------------------------------------|
| `training_curves.png`         | Accuracy & loss per epoch                    |
| `confusion_matrix.png`        | Normalised confusion matrix heatmap          |
| `roc_curves.png`              | OvR ROC curves for top classes + macro avg  |
| `sample_predictions.png`      | 16-image grid with true/pred labels          |
| `misclassified.png`           | Grid of model mistakes                       |
| `classification_report.csv`   | Per-class precision / recall / F1            |

---

## Prediction

### Command-line

```bash
python -m src.predict \
    --image path/to/leaf.jpg \
    --model saved_model/best_model.keras \
    --classes saved_model/class_names.json \
    --top_k 5 \
    --save images/my_prediction.png
```

## Results & Output

### Training Curves
![Training Curves](final-images/image2.png)

### TERMINAL OUTPUT
![Confusion Matrix](final-images/image3.png)






### Python API

```python
from src.utils import load_model, load_class_names
from src.predict import predict_single, predict_batch, display_prediction

model       = load_model("saved_model/best_model.keras")
class_names = load_class_names("saved_model/class_names.json")

# Single image
result = predict_single("leaf.jpg", model, class_names)
print(result["predicted_class"], result["confidence"])
display_prediction("leaf.jpg", result)

# Batch
paths   = ["leaf1.jpg", "leaf2.jpg", "leaf3.jpg"]
results = predict_batch(paths, model, class_names)
for r in results:
    print(r["image_path"], "→", r["predicted_class"], f"({r['confidence']*100:.1f}%)")
```

### Sample output

```
=======================================================
  PREDICTION RESULT
=======================================================
  Image            : test_leaf.jpg
  Predicted class  : Tomato___Early_blight
  Confidence       : 97.43%

  Top-5 Predictions:
    1. Tomato___Early_blight            97.43%  ██████████████████████████████
    2. Tomato___Septoria_leaf_spot       1.89%  █
    3. Tomato___Late_blight              0.41%
    4. Tomato___healthy                  0.15%
    5. Potato___Early_blight             0.12%
=======================================================
```

---

## Results

| Metric          | Score   |
|----------------|---------|
| Test Accuracy   | ~92 %   |
| Macro Precision | ~91 %   |
| Macro Recall    | ~91 %   |
| Macro F1-score  | ~91 %   |

> Results vary slightly with random seed and hardware. GPU training (~50 epochs) takes roughly 45–90 minutes.

---

## Future Improvements

1. **Transfer Learning** – Fine-tune EfficientNetV2-S or ResNet-50V2 pre-trained on ImageNet for faster convergence and higher accuracy.
2. **Grad-CAM visualisation** – Highlight which leaf regions drove the prediction (explainability).
3. **REST API / mobile app** – Wrap `predict_single` in a FastAPI endpoint or convert the model to TFLite for edge deployment.
4. **Semi-supervised learning** – Leverage unlabelled field images via pseudo-labelling or MixMatch to improve generalisation.
5. **Class-imbalance handling** – Apply focal loss or class-weighted cross-entropy for minority disease categories.
6. **Ensemble** – Average predictions from multiple CNN architectures to boost robustness.

---

## Resume Bullets

- **Engineered** an end-to-end deep learning pipeline for plant disease classification achieving **~92 % accuracy** across 38 classes using a custom 4-block CNN trained on 54 000+ PlantVillage images.
- **Implemented** data augmentation (rotation, zoom, brightness jitter, horizontal flip) with Keras ImageDataGenerator, reducing overfitting and improving validation accuracy by ~6 percentage points.
- **Built** a modular Python package (`preprocess`, `train`, `evaluate`, `predict`) with full docstrings, EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks, saving the best model automatically.
- **Produced** comprehensive evaluation artefacts (ROC curves, normalised confusion matrix, per-class F1 report, misclassification grid) using Scikit-learn and Matplotlib, demonstrating a macro F1 of **~0.91** on the held-out test set.

---

## Interview Q&A

### Q1. Why did you choose a custom CNN over a pre-trained model like ResNet?
**A.** A custom CNN lets us demonstrate core deep learning concepts (convolutions, batch normalisation, pooling, dropout) end-to-end and is a solid baseline. In production I would fine-tune a pre-trained EfficientNet or MobileNet, because transfer learning converges faster and typically achieves 2–4 % higher accuracy with far fewer labelled samples.

---

### Q2. What is BatchNormalization and why did you use it?
**A.** BatchNorm normalises each mini-batch to zero mean and unit variance, then applies learnable scale (γ) and shift (β) parameters. It reduces internal covariate shift, allows higher learning rates, and acts as a mild regulariser—so the model trains faster and is less sensitive to weight initialisation.

---

### Q3. How does Dropout prevent overfitting?
**A.** During training Dropout randomly zeros a fraction *p* of neuron activations, forcing the network to learn redundant representations and preventing any single neuron from becoming over-specialised. At inference time all neurons are active but their outputs are scaled by *(1 − p)* to maintain the same expected activation magnitude.

---

### Q4. Explain the role of GlobalAveragePooling in your architecture.
**A.** GlobalAveragePooling2D computes the spatial mean of each feature map, collapsing the *(H, W, C)* tensor to *(1, 1, C)*. Compared to Flatten, it reduces the number of parameters dramatically (no fully-connected weights over the spatial grid), is more robust to spatial translations, and acts as structural regularisation.

---

### Q5. What data augmentation techniques did you apply and why?
**A.** Rotation (±25°), width/height shifts (±15 %), shear (15 %), zoom (20 %), horizontal flips, and brightness jitter (×0.8 – 1.2). These simulate natural variation in camera angle, distance, and lighting, making the model invariant to irrelevant image transformations and reducing overfitting.

---

### Q6. How did you handle the train/val/test split and class imbalance?
**A.** Stratified splitting (Scikit-learn `train_test_split` with `stratify=labels`) ensures each split has the same class proportions as the full dataset. For imbalance, class-weighted training or focal loss can be applied; here we rely on augmentation and stratification as a first step.

---

### Q7. What is the purpose of ReduceLROnPlateau?
**A.** It monitors a chosen metric (val_loss) and halves the learning rate when no improvement is seen for *patience* epochs. This allows the optimiser to escape local minima and fine-tune at a lower rate rather than overshooting, typically yielding a 1–2 % accuracy gain in the final epochs.

---

### Q8. How did you evaluate model performance beyond accuracy?
**A.** Using Scikit-learn I computed per-class precision, recall, and F1-score, plus macro/weighted averages. I also plotted a normalised confusion matrix to spot systematic misclassifications (e.g. similar diseases confusing the model) and one-vs-rest ROC curves with AUC scores.

---

### Q9. What is the difference between macro and weighted F1-score?
**A.** Macro F1 averages the per-class F1 with equal weight regardless of class size—it penalises poor performance on minority classes. Weighted F1 weights each class by its support (number of samples), so it reflects overall accuracy more faithfully but can be misleadingly high when dominant classes perform well.

---

### Q10. Explain categorical cross-entropy as your loss function.
**A.** For a one-hot true label vector **y** and predicted softmax probability vector **ŷ**, categorical cross-entropy is  
`L = −Σ yᵢ · log(ŷᵢ)`.  
It penalises confident wrong predictions heavily (log(ε) → ∞) and approaches 0 when the predicted probability of the correct class approaches 1, providing a smooth, differentiable gradient for multi-class problems.

---

### Q11. How does the Adam optimiser differ from vanilla SGD?
**A.** Adam maintains per-parameter adaptive learning rates by tracking exponential moving averages of the gradient (first moment, m) and squared gradient (second moment, v). The effective update `lr × m̂ / (√v̂ + ε)` is large when gradients are consistent and small when they oscillate, making Adam significantly faster to converge than SGD, especially with sparse gradients.

---

### Q12. What is EarlyStopping and why is it important?
**A.** EarlyStopping monitors a validation metric and halts training when it fails to improve for a specified *patience* number of epochs, then restores the best weights. It prevents overfitting, saves compute time, and ensures the final model checkpoint is the best generalising one rather than the last.

---

### Q13. How would you deploy this model to production?
**A.** Convert the `.keras` model to TensorFlow SavedModel format and serve it with TensorFlow Serving behind an HTTP endpoint, or wrap `predict_single` in a FastAPI route. For mobile/edge, export to TFLite with INT8 quantisation using `TFLiteConverter`. Add an API Gateway, autoscaling, and monitoring (latency, accuracy drift) in cloud (GCP/AWS).

---

### Q14. How would you improve accuracy further without collecting more data?
**A.** (1) Fine-tune a pre-trained EfficientNetB4 (ImageNet weights) with a small custom head. (2) Use Mixup or CutMix advanced augmentation. (3) Apply test-time augmentation (TTA) by averaging predictions over multiple augmented copies of each test image. (4) Ensemble 2–3 diverse models.

---

### Q15. What is the ROC-AUC score and how did you compute it for a multi-class problem?
**A.** ROC-AUC measures the probability that the model ranks a random positive example higher than a random negative one. For multi-class (38 classes) I used a **One-vs-Rest** strategy: binarise each class against all others, compute the ROC curve and AUC for each, then take the macro average. Scikit-learn's `roc_curve` and `auc` functions handle this directly after `label_binarize`.
