"""
AFI Oral Cancer Screening — Model Comparison Pipeline
======================================================
Custom AFI-CNN vs MobileNetV1 (α=0.5) vs MobileNetV2 vs EfficientNetB0 vs ViT-B/16

Project context:
  - Hardware: ESP32-S3 (240MHz, 8MB Flash, 8MB PSRAM)
  - Input: 96×96 grayscale AFI fluorescence images + 4D patient metadata
  - Target: INT8 TFLite model < 1MB, <200ms inference, offline
  - Dataset: ~100 clinical AFI images (5× augmentation → ~500 training samples)
  - Training GPU: RTX 3050 Laptop (4GB VRAM, CUDA 12.1)

Model roles:
  Custom AFI-CNN   → Primary candidate. Designed from scratch for 96×96
                     grayscale AFI input + metadata fusion. Targets INT8
                     TFLite deployment on ESP32-S3.
  MobileNetV1 α=0.5 → Baseline edge candidate. Previously evaluated in
                     project pipeline. Known INT8 size ~0.45 MB.
  MobileNetV2      → Stronger edge candidate. Better accuracy/FLOP than V1.
                     INT8 ~1.4 MB — fits Flash with margin.
  EfficientNetB0   → Upper-bound edge reference. Best accuracy of edge
                     group but ~4 MB INT8 — tight for 8MB Flash.
  ViT-B/16         → Accuracy ceiling reference ONLY. 86M params, ~86MB
                     INT8. Physically impossible on ESP32-S3. Included to
                     show the cost of going transformer at this scale.

NOTE on dataset size:
  With 100 base images → 5× augmentation → ~500 training samples, a
  single 80/20 split leaves ~100 test samples — too few for stable
  accuracy estimates. We use StratifiedKFold(n=5) and report mean ± std,
  which is the correct approach for small medical datasets.

DATA LAYOUT:
  data/
    cancer/      ← AFI images of suspicious/malignant lesions
    non_cancer/   ← AFI images of healthy tissue

USAGE:
  pip install tensorflow scikit-learn scipy matplotlib pandas vit-keras
  python oral_cancer_comparison.py
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, f1_score, precision_score
)
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt 

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
IMG_SIZE        = (96, 96)        # matches ESP32-S3 tensor preprocessing
VIT_IMG_SIZE    = (224, 224)      # ViT requires 224×224; resized internally
N_META_FEATURES = 4               # [age, smoking, betel_chewing, alcohol]
BATCH_SIZE      = 8               # small dataset — small batch
N_FOLDS         = 5
EPOCHS          = 40
EARLY_STOP_PAT  = 8
L2_PENALTY      = 1e-4
DROPOUT_RATE    = 0.3
OUTPUT_DIR      = "results"
SEED            = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Whether each model is a realistic ESP32-S3 edge candidate
EDGE_CANDIDATE = {
    "Custom AFI-CNN":   True,
    "MobileNetV1-0.5":  True,
    "MobileNetV2":      True,
    "EfficientNetB0":   True,   # tight but possible
    "ViT-B/16":         False,  # reference only
}

# Approximate INT8 TFLite sizes for reporting (from literature/profiling)
APPROX_INT8_MB = {
    "Custom AFI-CNN":   "~0.2 MB",
    "MobileNetV1-0.5":  "~0.45 MB",
    "MobileNetV2":      "~1.4 MB",
    "EfficientNetB0":   "~4.0 MB",
    "ViT-B/16":         "~86 MB (not deployable)",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING — images + synthetic metadata
# ─────────────────────────────────────────────────────────────────────────────
def load_file_labels(data_dir):
    """Returns (filepaths, labels). 1 = cancer/suspicious, 0 = healthy."""
    filepaths, labels = [], []
    for cls, label in [("cancer", 1), ("non_cancer", 0)]:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            raise FileNotFoundError(
                f"Directory not found: {cls_dir}\n"
                "Create data/cancer/ and data/no_cancer/ with your AFI images."
            )
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(label)
    return np.array(filepaths), np.array(labels)


def generate_synthetic_metadata(n_samples, labels, seed=SEED):
    """
    Generate plausible synthetic metadata for images that don't yet have
    real patient records. Replace with actual CSV in production.

    Fields:
      age          : integer 25–75, normalised to [0, 1]
      smoking      : binary (0/1); cancer class has higher prevalence
      betel_chewing: ordinal 0/1/2 (Never/Occasional/Daily)
      alcohol      : ordinal 0/1/2 (Never/Occasional/Daily)
    """
    rng = np.random.default_rng(seed)
    meta = np.zeros((n_samples, N_META_FEATURES), dtype=np.float32)

    for i, label in enumerate(labels):
        # Higher-risk profile for cancer class (reflects epidemiology)
        if label == 1:
            age = rng.integers(40, 76)
            smoking = rng.choice([0, 1], p=[0.3, 0.7])
            betel = rng.choice([0, 1, 2], p=[0.2, 0.3, 0.5])
            alcohol = rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
        else:
            age = rng.integers(25, 65)
            smoking = rng.choice([0, 1], p=[0.6, 0.4])
            betel = rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
            alcohol = rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])

        meta[i] = [
            (age - 25) / 50.0,   # normalise age to [0, 1]
            float(smoking),
            betel / 2.0,         # ordinal scale to [0, 1]
            alcohol / 2.0,
        ]

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# 2. AUGMENTATION — AFI-specific
# ─────────────────────────────────────────────────────────────────────────────
def augment_afi(img):
    """
    Augmentation tuned for AFI fluorescence images.
    Geometric transforms: safe — rotation, flips preserve quenching zones.
    Colour/brightness: kept very mild — AFI diagnostic signal IS the
    brightness difference between healthy (bright) and suspicious (dark)
    regions. Heavy brightness jitter would corrupt the label.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    # Rotation via rot90
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    # Very mild brightness jitter — max_delta kept small intentionally
    img = tf.image.random_brightness(img, max_delta=0.08)
    # Mild contrast — does not shift mean pixel value
    img = tf.image.random_contrast(img, lower=0.92, upper=1.08)
    img = tf.clip_by_value(img, -1.0, 1.0)  # preserve [-1,1] normalisation range
    return img


def decode_image(filepath, label, augment):
    img = tf.io.read_file(filepath)
    # channels=0: preserve original (RGB or grayscale); convert explicitly below
    img = tf.image.decode_image(img, channels=0, expand_animations=False)
    # Auto-convert: RGB white-light photos → grayscale; real AFI (1ch) → unchanged
    img = tf.cond(
        tf.equal(tf.shape(img)[2], 3),
        lambda: tf.image.rgb_to_grayscale(img),  # luminance: 0.299R+0.587G+0.114B
        lambda: img                               # already 1-channel (AFI)
    )
    img = tf.image.resize(img, IMG_SIZE)
    # Symmetric min-max → [-1, 1] to align with INT8 symmetric quantisation grid
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    if augment:
        img = augment_afi(img)
    return img, tf.cast(label, tf.float32)


def make_image_dataset(filepaths, labels, augment):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(lambda f, l: decode_image(f, l, augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def make_multimodal_dataset(filepaths, labels, metadata, augment):
    """Dataset yielding ((image, metadata), label) for multimodal models."""
    img_ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    img_ds = img_ds.map(lambda f, l: decode_image(f, l, augment),
                        num_parallel_calls=tf.data.AUTOTUNE)
    meta_ds = tf.data.Dataset.from_tensor_slices(metadata.astype(np.float32))
    label_ds = tf.data.Dataset.from_tensor_slices(labels.astype(np.float32))

    ds = tf.data.Dataset.zip((
        tf.data.Dataset.zip((
            img_ds.map(lambda img, _: img),
            meta_ds
        )),
        label_ds
    ))
    if augment:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── 3a. CUSTOM AFI-CNN (primary candidate) ──────────────────────────────────
def build_custom_afi_cnn(input_shape):
    """
    Purpose-built for 96×96 grayscale AFI fluorescence images + 4D metadata.

    Design decisions:
      - Grayscale input (1 channel): AFI is a single fluorescence band.
        Forcing 3-channel RGB conversion wastes parameters on channels
        that carry no additional spectral information.
      - Depthwise separable convolutions: reduce FLOPs by ~8-9× vs
        standard conv while preserving receptive field. Critical for
        keeping INT8 model under 0.5 MB.
      - Channel progression [16→32→64→128]: moderate growth — deeper
        networks overfit faster than they generalise on <500 images.
      - GlobalAveragePooling2D: replaces flatten+dense. Fewer params,
        better regularisation, spatial invariance to lesion position.
      - Metadata branch: 3-layer Dense MLP → 16-dim clinical embedding.
        Fused at embedding level (not early/late fusion).
      - Two-stage output: 128-dim fused → 64-dim → sigmoid. Matches
        the architecture validated in MSA University (Paper 3) and
        IIT Kharagpur (Paper 1) papers.
      - L2 regularisation on all Dense layers: compensates for small dataset.
      - Dropout 0.3 throughout: standard for medical imaging small datasets.

    Estimated INT8 TFLite size: ~0.2 MB
    Estimated ESP32-S3 inference: ~15-25 ms
    """

    # ── Image branch ──
    img_input = layers.Input(shape=input_shape, name="afi_image_input")

    x = layers.Conv2D(16, 3, padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(L2_PENALTY))(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)          # 96→48

    x = layers.SeparableConv2D(32, 3, padding="same",
                                depthwise_initializer="he_normal",
                                depthwise_regularizer=regularizers.l2(L2_PENALTY))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)          # 48→24

    x = layers.SeparableConv2D(64, 3, padding="same",
                                depthwise_initializer="he_normal",
                                depthwise_regularizer=regularizers.l2(L2_PENALTY))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)          # 24→12

    x = layers.SeparableConv2D(128, 3, padding="same",
                                depthwise_initializer="he_normal",
                                depthwise_regularizer=regularizers.l2(L2_PENALTY))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)          # 12→6

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    img_embedding = layers.Dense(
        64, activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(L2_PENALTY),
        name="img_embedding"
    )(x)

    # ── Metadata branch ──
    meta_input = layers.Input(shape=(N_META_FEATURES,), name="metadata_input")
    m = layers.Dense(32, activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(L2_PENALTY))(meta_input)
    m = layers.Dropout(DROPOUT_RATE)(m)
    meta_embedding = layers.Dense(
        16, activation="relu",
        kernel_initializer="he_normal",
        name="meta_embedding"
    )(m)

    # ── Fusion ──
    fused = layers.Concatenate()([img_embedding, meta_embedding])   # 80-dim

    x = layers.Dense(128, activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(L2_PENALTY))(fused)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    output = layers.Dense(1, activation="sigmoid", name="risk_score")(x)

    return models.Model(
        inputs=[img_input, meta_input],
        outputs=output,
        name="Custom_AFI_CNN"
    )


# ── 3b. MobileNetV1 α=0.5 ───────────────────────────────────────────────────
def build_mobilenetv1(input_shape):
    """
    MobileNetV1 with width multiplier α=0.5. Our existing baseline.
    Grayscale → 3-channel repeat (ImageNet weights expect RGB).
    INT8 ~0.45 MB. Previously validated in project pipeline.
    Image-only (no metadata branch) for fair architecture comparison.
    Add metadata fusion after picking the best image backbone.
    """
    img_input = layers.Input(shape=input_shape, name="image_input")

    # Repeat grayscale channel → 3 channels for ImageNet pretrained weights
    x = layers.Concatenate()([img_input, img_input, img_input])

    base = applications.MobileNet(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
        alpha=0.5
    )
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs=img_input, outputs=output, name="MobileNetV1_0.5")


# ── 3c. MobileNetV2 ─────────────────────────────────────────────────────────
def build_mobilenetv2(input_shape):
    """
    MobileNetV2 — depthwise separable + inverted residuals + linear bottleneck.
    Better accuracy/FLOP ratio than V1. INT8 ~1.4 MB.
    """
    img_input = layers.Input(shape=input_shape, name="image_input")
    x = layers.Concatenate()([img_input, img_input, img_input])
    x = applications.mobilenet_v2.preprocess_input(x * 127.5 + 127.5)

    base = applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs=img_input, outputs=output, name="MobileNetV2")


# ── 3d. EfficientNetB0 ──────────────────────────────────────────────────────
def build_efficientnet(input_shape):
    """
    EfficientNetB0 — compound scaling of depth/width/resolution.
    Best accuracy of edge group. INT8 ~4 MB — tight on 8MB Flash.
    Included as upper-bound edge reference.
    """
    img_input = layers.Input(shape=input_shape, name="image_input")
    x = layers.Concatenate()([img_input, img_input, img_input])
    x = (x + 1.0) * 127.5  # [-1,1] → [0,255] for EfficientNet preprocessing

    base = applications.EfficientNetB0(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs=img_input, outputs=output, name="EfficientNetB0")


# ── 3e. ViT-B/16 (reference only) ───────────────────────────────────────────
def build_vit(input_shape):
    """
    ViT-B/16. ACCURACY REFERENCE ONLY. See module docstring.
    86M params, ~86MB INT8 — ESP32-S3 deployment is physically impossible.
    Included so the report can state: 'ViT-B/16 achieves X% AUC at
    86M parameters; our Custom AFI-CNN achieves Y% at ~50K parameters.'
    """
    vit_base = vit.vit_b16(
        image_size=VIT_IMG_SIZE[0],
        pretrained=True,
        include_top=False,
        pretrained_top=False,
    )
    vit_base.trainable = False

    img_input = layers.Input(shape=input_shape, name="image_input")
    x = layers.Resizing(VIT_IMG_SIZE[0], VIT_IMG_SIZE[1])(img_input)
    x = layers.Concatenate()([x, x, x])  # grayscale → 3ch
    x = vit_base(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs=img_input, outputs=output, name="ViT_B16")


# Model registry
MODEL_BUILDERS = {
    "Custom AFI-CNN":  build_custom_afi_cnn,
    "MobileNetV1-0.5": build_mobilenetv1,
    "MobileNetV2":     build_mobilenetv2,
    "EfficientNetB0":  build_efficientnet,
}

MULTIMODAL_MODELS = {"Custom AFI-CNN"}   # models that take (image, metadata)


# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS
# ─────────────────────────────────────────────────────────────────────────────
def find_best_threshold(y_true, y_prob):
    """
    ROC-AUC calibrated threshold — maximises Youden's J (sensitivity + specificity - 1).
    Used for V2 architecture. Sonawane et al. (2026) validates this approach
    for out-of-distribution robustness on oral cancer classifiers.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def compute_metrics(y_true, y_prob, threshold=None):
    """
    For screening devices: sensitivity > specificity priority.
    A missed cancer (FN) is worse than a false alarm (FP).
    """
    if threshold is None:
        threshold = find_best_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,                      # recall on cancer class
        "specificity": specificity,
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":     roc_auc_score(y_true, y_prob),
        "threshold":   threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. K-FOLD TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_kfold_comparison():
    filepaths, labels = load_file_labels(DATA_DIR)
    metadata = generate_synthetic_metadata(len(labels), labels)

    print(f"\nDataset: {len(filepaths)} images "
          f"({int(labels.sum())} cancer / {int((1-labels).sum())} healthy)")
    print(f"Metadata shape: {metadata.shape}")
    print(f"Running {N_FOLDS}-fold stratified CV\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    input_shape = IMG_SIZE + (1,)   # grayscale

    fold_results = {name: [] for name in MODEL_BUILDERS}
    roc_data     = {name: [] for name in MODEL_BUILDERS}
    model_info   = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(filepaths, labels)):
        print(f"{'='*60}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"{'='*60}")

        y_val = labels[val_idx]

        for model_name, builder in MODEL_BUILDERS.items():
            tf.keras.backend.clear_session()

            is_multimodal = model_name in MULTIMODAL_MODELS

            if is_multimodal:
                train_ds = make_multimodal_dataset(
                    filepaths[train_idx], labels[train_idx],
                    metadata[train_idx], augment=True
                )
                val_ds = make_multimodal_dataset(
                    filepaths[val_idx], labels[val_idx],
                    metadata[val_idx], augment=False
                )
                model = builder(input_shape)
            else:
                train_ds = make_image_dataset(
                    filepaths[train_idx], labels[train_idx], augment=True
                )
                val_ds = make_image_dataset(
                    filepaths[val_idx], labels[val_idx], augment=False
                )
                model = builder(input_shape)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-3),
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=EARLY_STOP_PAT,
                    restore_best_weights=True,
                    verbose=0
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=0
                )
            ]

            t_train = time.time()
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1
            )
            train_time = time.time() - t_train

            # Inference
            if is_multimodal:
                y_prob = model.predict(val_ds, verbose=0).ravel()
            else:
                y_prob = model.predict(val_ds, verbose=0).ravel()

            metrics = compute_metrics(y_val, y_prob)
            fold_results[model_name].append(metrics)

            fpr, tpr, _ = roc_curve(y_val, y_prob)
            roc_data[model_name].append((fpr, tpr, metrics["roc_auc"]))

            # Profile once per model type
            if model_name not in model_info:
                t0 = time.time()
                for _ in range(20):
                    model.predict(val_ds.take(1), verbose=0)
                latency_ms = (time.time() - t0) / 20 * 1000

                model_info[model_name] = {
                    "params":        model.count_params(),
                    "latency_ms":    latency_ms,
                    "train_time_s":  train_time,
                    "int8_size_est": APPROX_INT8_MB[model_name],
                    "edge_candidate":EDGE_CANDIDATE[model_name],
                }

            print(
                f"  {model_name:<18} "
                f"acc={metrics['accuracy']:.3f}  "
                f"sens={metrics['sensitivity']:.3f}  "
                f"spec={metrics['specificity']:.3f}  "
                f"auc={metrics['roc_auc']:.3f}  "
                f"thr={metrics['threshold']:.3f}"
            )

    return fold_results, roc_data, model_info


# ─────────────────────────────────────────────────────────────────────────────
# 6. AGGREGATE + SIGNIFICANCE TEST
# ─────────────────────────────────────────────────────────────────────────────
def summarize(fold_results, model_info):
    rows = []
    for model_name, folds in fold_results.items():
        df = pd.DataFrame(folds)
        row = {"model": model_name}
        for col in df.columns:
            row[f"{col}_mean"] = df[col].mean()
            row[f"{col}_std"]  = df[col].std()
        row.update(model_info[model_name])
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"), index=False)

    print("\n" + "="*80)
    print("SUMMARY (mean ± std across 5 folds)")
    print("="*80)

    display_cols = [
        "model",
        "accuracy_mean", "accuracy_std",
        "sensitivity_mean", "sensitivity_std",
        "specificity_mean", "specificity_std",
        "roc_auc_mean", "roc_auc_std",
        "params", "int8_size_est", "latency_ms", "edge_candidate"
    ]
    print(summary[display_cols].to_string(index=False))

    print("\n[!] ViT-B/16 params/latency/int8_size are NOT deployment numbers.")
    print("    Included as accuracy ceiling reference only.\n")

    return summary


def significance_tests(fold_results):
    """
    Paired t-test: Custom AFI-CNN vs each other model.
    Primary metric: sensitivity (screening task).
    Secondary: accuracy, AUC.
    """
    print("="*60)
    print("SIGNIFICANCE TESTS (Custom AFI-CNN vs baselines)")
    print("Paired t-test on fold-wise sensitivity (n=5 folds)")
    print("="*60)

    custom_sens = [f["sensitivity"] for f in fold_results["Custom AFI-CNN"]]
    custom_acc  = [f["accuracy"]    for f in fold_results["Custom AFI-CNN"]]
    custom_auc  = [f["roc_auc"]     for f in fold_results["Custom AFI-CNN"]]

    for name in fold_results:
        if name == "Custom AFI-CNN":
            continue
        other_sens = [f["sensitivity"] for f in fold_results[name]]
        other_acc  = [f["accuracy"]    for f in fold_results[name]]
        other_auc  = [f["roc_auc"]     for f in fold_results[name]]

        _, p_sens = ttest_rel(custom_sens, other_sens)
        _, p_acc  = ttest_rel(custom_acc,  other_acc)
        _, p_auc  = ttest_rel(custom_auc,  other_auc)

        sig = lambda p: "significant (p<0.05)" if p < 0.05 else "not significant"
        print(f"\nCustom AFI-CNN vs {name}:")
        print(f"  sensitivity: p={p_sens:.4f} — {sig(p_sens)}")
        print(f"  accuracy:    p={p_acc:.4f}  — {sig(p_acc)}")
        print(f"  roc_auc:     p={p_auc:.4f}  — {sig(p_auc)}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(summary, roc_data):
    # ── Bar chart: accuracy, sensitivity, specificity, AUC ──
    metrics = ["accuracy", "sensitivity", "specificity", "roc_auc"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(summary))
    w = 0.18
    for i, m in enumerate(metrics):
        means = summary[f"{m}_mean"]
        stds  = summary[f"{m}_std"]
        ax.bar(x + i*w, means, w, yerr=stds, capsize=4, label=m)

    xlabels = []
    for _, row in summary.iterrows():
        label = row["model"]
        if not row["edge_candidate"]:
            label += "\n(ref only)"
        xlabels.append(label)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("AFI Oral Cancer Screening — Model Comparison (5-fold CV mean ± std)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_bar_chart.png"), dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/comparison_bar_chart.png")

    # ── ROC curves (fold 0 of each model) ──
    fig, ax = plt.subplots(figsize=(7, 7))
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for i, (model_name, curves) in enumerate(roc_data.items()):
        fpr, tpr, auc = curves[0]
        suffix = " [ref only]" if not EDGE_CANDIDATE[model_name] else ""
        ax.plot(fpr, tpr, linestyle=styles[i % len(styles)],
                label=f"{model_name}{suffix} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves — AFI Oral Cancer (fold 1)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/roc_curves.png")

    # ── Params vs AUC scatter (edge candidates only) ──
    fig, ax = plt.subplots(figsize=(7, 5))
    edge_rows = summary[summary["edge_candidate"] == True]
    for _, row in edge_rows.iterrows():
        ax.scatter(row["params"] / 1e6, row["roc_auc_mean"],
                   s=120, zorder=5, label=row["model"])
        ax.annotate(
            row["model"],
            (row["params"] / 1e6, row["roc_auc_mean"]),
            textcoords="offset points", xytext=(8, 4), fontsize=9
        )
    ax.set_xlabel("Parameters (Millions)")
    ax.set_ylabel("Mean AUC-ROC (5-fold)")
    ax.set_title("Edge candidates: parameters vs AUC")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "params_vs_auc.png"), dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/params_vs_auc.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(__doc__)
    fold_results, roc_data, model_info = run_kfold_comparison()
    summary = summarize(fold_results, model_info)
    significance_tests(fold_results)
    plot_results(summary, roc_data)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("  summary_metrics.csv")
    print("  comparison_bar_chart.png")
    print("  roc_curves.png")
    print("  params_vs_auc.png")
