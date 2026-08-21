"""
AFI + WLI Oral Cancer Screening — Dual-Stream Model
======================================================
This script trains a Custom Dual-Stream CNN that fuses:
1. White Light Imaging (WLI) - RGB
2. Autofluorescence Imaging (AFI) - Grayscale
3. Clinical Metadata (4 features)

All three inputs are passed simultaneously to predict oral cancer risk.
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, f1_score, precision_score
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
IMG_SIZE        = (96, 96)        
N_META_FEATURES = 4               
BATCH_SIZE      = 8               
N_FOLDS         = 5
EPOCHS          = 40
EARLY_STOP_PAT  = 8
L2_PENALTY      = 1e-4
DROPOUT_RATE    = 0.3
OUTPUT_DIR      = "results_dual"
SEED            = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING — Paired WLI + AFI + synthetic metadata
# ─────────────────────────────────────────────────────────────────────────────
def load_file_labels(data_dir):
    """Returns (filepaths_wli, filepaths_afi, labels). Requires matching filenames."""
    filepaths_wli, filepaths_afi, labels = [], [], []
    for cls, label in [("cancer", 1), ("non_cancer", 0)]:
        wli_dir = os.path.join(data_dir, "wli", cls)
        afi_dir = os.path.join(data_dir, "afi", cls)
        
        if not os.path.exists(wli_dir) or not os.path.exists(afi_dir):
            raise FileNotFoundError(f"Missing dual-stream directories!\nPlease create: {wli_dir} and {afi_dir}")
            
        for fname in os.listdir(wli_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                wli_path = os.path.join(wli_dir, fname)
                basename = os.path.splitext(fname)[0]
                
                # Check for matching AFI file regardless of extension and prefix
                afi_path = None
                for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                    # Try exact basename
                    candidate1 = os.path.join(afi_dir, basename + ext)
                    # Try with pseudo_afi_ prefix
                    candidate2 = os.path.join(afi_dir, "pseudo_afi_" + basename + ext)
                    
                    if os.path.exists(candidate1):
                        afi_path = candidate1
                        break
                    elif os.path.exists(candidate2):
                        afi_path = candidate2
                        break
                
                if afi_path:
                    filepaths_wli.append(wli_path)
                    filepaths_afi.append(afi_path)
                    labels.append(label)
                else:
                    print(f"Warning: No matching AFI image found for {wli_path}")
                    
    return np.array(filepaths_wli), np.array(filepaths_afi), np.array(labels)

def generate_synthetic_metadata(n_samples, labels, seed=SEED):
    rng = np.random.default_rng(seed)
    meta = np.zeros((n_samples, N_META_FEATURES), dtype=np.float32)
    for i, label in enumerate(labels):
        if label == 1:
            age, smoking, betel, alcohol = rng.integers(40, 76), rng.choice([0, 1], p=[0.3, 0.7]), rng.choice([0, 1, 2], p=[0.2, 0.3, 0.5]), rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
        else:
            age, smoking, betel, alcohol = rng.integers(25, 65), rng.choice([0, 1], p=[0.6, 0.4]), rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]), rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
        meta[i] = [(age - 25) / 50.0, float(smoking), betel / 2.0, alcohol / 2.0]
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def find_best_threshold(y_true, y_prob):
    """Youden's J: maximises sensitivity + specificity jointly."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = np.argmax(tpr - fpr)
    return float(thresholds[best_idx]), fpr, tpr


def compute_metrics(y_true, y_prob):
    threshold, fpr, tpr = find_best_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":     roc_auc_score(y_true, y_prob),
        "threshold":   threshold,
        "fpr":         fpr,
        "tpr":         tpr,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 2. DUAL-STREAM AUGMENTATION & DECODING
# ─────────────────────────────────────────────────────────────────────────────
def decode_image_single(filepath, channels):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=channels, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32) / 127.5 - 1.0

def decode_paired_images(wli_path, afi_path, label, augment):
    wli_img = decode_image_single(wli_path, channels=3) # RGB WLI
    afi_img = decode_image_single(afi_path, channels=1) # Grayscale AFI
    
    if augment:
        # Concatenate to ensure exact same spatial augmentations are applied to both
        combined = tf.concat([wli_img, afi_img], axis=-1)
        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        combined = tf.image.rot90(combined, k=k)
        
        # Split back
        wli_img, afi_img = combined[..., :3], combined[..., 3:]
        
        # Color jitter independently
        wli_img = tf.image.random_brightness(wli_img, max_delta=0.08)
        wli_img = tf.image.random_contrast(wli_img, lower=0.92, upper=1.08)
        afi_img = tf.image.random_brightness(afi_img, max_delta=0.08)
        afi_img = tf.image.random_contrast(afi_img, lower=0.92, upper=1.08)
        
        wli_img = tf.clip_by_value(wli_img, -1.0, 1.0)
        afi_img = tf.clip_by_value(afi_img, -1.0, 1.0)
        
    return wli_img, afi_img, tf.cast(label, tf.float32)

def make_dual_dataset(filepaths_wli, filepaths_afi, labels, metadata, augment):
    meta_ds = tf.data.Dataset.from_tensor_slices(metadata.astype(np.float32))
    label_ds = tf.data.Dataset.from_tensor_slices(labels.astype(np.float32))

    # Decode once — guarantees WLI and AFI share identical spatial augmentation
    path_ds = tf.data.Dataset.from_tensor_slices((filepaths_wli, filepaths_afi, labels))
    decoded_ds = path_ds.map(
        lambda w, a, l: decode_paired_images(w, a, l, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Zip with meta + label, then reshape to ((wli, afi, meta), label) in one pass
    combined_ds = tf.data.Dataset.zip((decoded_ds, meta_ds, label_ds))
    ds = combined_ds.map(
        lambda imgs, meta, lbl: ((imgs[0], imgs[1], meta), lbl)
    )

    if augment:
        ds = ds.shuffle(buffer_size=len(labels), seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# ─────────────────────────────────────────────────────────────────────────────
# 3. DUAL-STREAM MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def build_dual_stream_cnn():
    # ── Inputs ──
    wli_input = layers.Input(shape=IMG_SIZE + (3,), name="wli_input")
    afi_input = layers.Input(shape=IMG_SIZE + (1,), name="afi_input")
    meta_input = layers.Input(shape=(N_META_FEATURES,), name="meta_input")

    # ── WLI Branch (RGB) ──
    xw = layers.Conv2D(16, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2_PENALTY))(wli_input)
    xw = layers.BatchNormalization()(xw)
    xw = layers.ReLU()(xw)
    xw = layers.MaxPooling2D(2)(xw)
    
    xw = layers.SeparableConv2D(32, 3, padding="same", depthwise_initializer="he_normal", depthwise_regularizer=regularizers.l2(L2_PENALTY))(xw)
    xw = layers.BatchNormalization()(xw)
    xw = layers.ReLU()(xw)
    xw = layers.MaxPooling2D(2)(xw)
    
    xw = layers.SeparableConv2D(64, 3, padding="same", depthwise_initializer="he_normal", depthwise_regularizer=regularizers.l2(L2_PENALTY))(xw)
    xw = layers.BatchNormalization()(xw)
    xw = layers.ReLU()(xw)
    xw = layers.MaxPooling2D(2)(xw)
    
    xw = layers.GlobalAveragePooling2D()(xw)
    xw = layers.Dropout(DROPOUT_RATE)(xw)
    wli_embedding = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(xw)

    # ── AFI Branch (Grayscale) ──
    xa = layers.Conv2D(16, 3, padding="same", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2_PENALTY))(afi_input)
    xa = layers.BatchNormalization()(xa)
    xa = layers.ReLU()(xa)
    xa = layers.MaxPooling2D(2)(xa)
    
    xa = layers.SeparableConv2D(32, 3, padding="same", depthwise_initializer="he_normal", depthwise_regularizer=regularizers.l2(L2_PENALTY))(xa)
    xa = layers.BatchNormalization()(xa)
    xa = layers.ReLU()(xa)
    xa = layers.MaxPooling2D(2)(xa)
    
    xa = layers.SeparableConv2D(64, 3, padding="same", depthwise_initializer="he_normal", depthwise_regularizer=regularizers.l2(L2_PENALTY))(xa)
    xa = layers.BatchNormalization()(xa)
    xa = layers.ReLU()(xa)
    xa = layers.MaxPooling2D(2)(xa)
    
    xa = layers.SeparableConv2D(128, 3, padding="same", depthwise_initializer="he_normal", depthwise_regularizer=regularizers.l2(L2_PENALTY))(xa)
    xa = layers.BatchNormalization()(xa)
    xa = layers.ReLU()(xa)
    xa = layers.MaxPooling2D(2)(xa)
    
    xa = layers.GlobalAveragePooling2D()(xa)
    xa = layers.Dropout(DROPOUT_RATE)(xa)
    afi_embedding = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(xa)

    # ── Metadata Branch ──
    m = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(meta_input)
    m = layers.Dropout(DROPOUT_RATE)(m)
    meta_embedding = layers.Dense(16, activation="relu", kernel_initializer="he_normal")(m)

    # ── Fusion & Output ──
    fused = layers.Concatenate()([wli_embedding, afi_embedding, meta_embedding])
    
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2_PENALTY))(fused)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    output = layers.Dense(1, activation="sigmoid", name="risk_score")(x)

    return models.Model(inputs=[wli_input, afi_input, meta_input], outputs=output, name="Dual_Stream_CNN")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    filepaths_wli, filepaths_afi, labels = load_file_labels(DATA_DIR)
    
    if len(labels) == 0:
        print("No paired images found. Please organize your folders properly.")
        exit(1)
        
    metadata = generate_synthetic_metadata(len(labels), labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    roc_data = []
    
    print(f"\nTraining Dual-Stream Model on {len(labels)} perfect pairs...\n")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(filepaths_wli, labels)):
        print(f"{'='*60}\nFold {fold_idx + 1}/{N_FOLDS}\n{'='*60}")
        
        train_ds = make_dual_dataset(filepaths_wli[train_idx], filepaths_afi[train_idx], labels[train_idx], metadata[train_idx], augment=True)
        val_ds = make_dual_dataset(filepaths_wli[val_idx], filepaths_afi[val_idx], labels[val_idx], metadata[val_idx], augment=False)
        
        model = build_dual_stream_cnn()
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PAT, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)
        
        y_val = labels[val_idx]
        y_prob = model.predict(val_ds, verbose=0).ravel()

        m = compute_metrics(y_val, y_prob)
        fold_results.append({'fold': fold_idx+1, **{k: v for k, v in m.items() if k not in ('fpr','tpr')}})
        roc_data.append((m['fpr'], m['tpr'], m['roc_auc']))

        print(
            f"  --> Fold {fold_idx+1}: "
            f"acc={m['accuracy']:.3f}  sens={m['sensitivity']:.3f}  "
            f"spec={m['specificity']:.3f}  auc={m['roc_auc']:.3f}  thr={m['threshold']:.3f}"
        )
        
    df = pd.DataFrame(fold_results)
    n_params = model.count_params()

    print("\n" + "="*60)
    print("FINAL DUAL-STREAM RESULTS (mean ± std across 5 folds)")
    print("="*60)
    for col in ['accuracy','sensitivity','specificity','roc_auc']:
        print(f"  {col:<14}: {df[col].mean():.3f} ± {df[col].std():.3f}")
    print(f"  params       : {n_params:,}")

    # ── Save model ──
    model_path = os.path.join(OUTPUT_DIR, "dual_stream_model.keras")
    model.save(model_path)
    print(f"\n✅ Model saved to {model_path}")

    # ── Summary CSV ──
    csv_path = os.path.join(OUTPUT_DIR, "summary_metrics.csv")
    dual_row = {
        "model":             "Dual-Stream (WLI+AFI)",
        "accuracy_mean":     round(df['accuracy'].mean(), 4),
        "accuracy_std":      round(df['accuracy'].std(), 4),
        "sensitivity_mean":  round(df['sensitivity'].mean(), 4),
        "sensitivity_std":   round(df['sensitivity'].std(), 4),
        "specificity_mean":  round(df['specificity'].mean(), 4),
        "specificity_std":   round(df['specificity'].std(), 4),
        "roc_auc_mean":      round(df['roc_auc'].mean(), 4),
        "roc_auc_std":       round(df['roc_auc'].std(), 4),
        "params":            n_params,
    }
    pd.DataFrame([dual_row]).to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")

    # ── Known results from oral_cancer_comparison.py (5-fold CV) ──
    comparison_models = {
        "Custom AFI-CNN":      {"accuracy": 0.909, "sensitivity": 0.926, "specificity": 0.876, "roc_auc": 0.951, "params": 40737},
        "MobileNetV1-0.5":     {"accuracy": 0.901, "sensitivity": 0.918, "specificity": 0.868, "roc_auc": 0.944, "params": 830049},
        "MobileNetV2":         {"accuracy": 0.884, "sensitivity": 0.864, "specificity": 0.924, "roc_auc": 0.955, "params": 2259265},
        "EfficientNetB0":      {"accuracy": 0.917, "sensitivity": 0.922, "specificity": 0.908, "roc_auc": 0.960, "params": 4050852},
        "Dual-Stream (WLI+AFI)": {"accuracy": df['accuracy'].mean(), "sensitivity": df['sensitivity'].mean(),
                                   "specificity": df['specificity'].mean(), "roc_auc": df['roc_auc'].mean(),
                                   "params": n_params},
    }

    # ── Plot 1: Comparison bar chart ──
    metrics_to_plot = ["accuracy", "sensitivity", "specificity", "roc_auc"]
    model_names = list(comparison_models.keys())
    x = np.arange(len(model_names))
    w = 0.18

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, m in enumerate(metrics_to_plot):
        vals = [comparison_models[mn][m] for mn in model_names]
        bars = ax.bar(x + i*w, vals, w, label=m)
        # Highlight Dual-Stream bar
        bars[-1].set_edgecolor("black")
        bars[-1].set_linewidth(1.5)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(model_names, fontsize=9, rotation=10, ha="right")
    ax.set_ylim(0.75, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Custom AFI-CNN vs Baselines vs Dual-Stream (5-fold CV)")
    ax.legend(loc="lower right")
    ax.axvline(x=len(model_names)-1 - 0.1, color='gray', linestyle='--', alpha=0.5, label='Dual-Stream')
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "comparison_bar_chart.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {bar_path}")

    # ── Plot 2: ROC curves (all 5 folds of Dual-Stream) ──
    fig, ax = plt.subplots(figsize=(7, 7))
    styles = ["-", "--", "-.", ":", (0,(3,1,1,1))]
    for i, (fpr_arr, tpr_arr, auc_val) in enumerate(roc_data):
        ax.plot(fpr_arr, tpr_arr, linestyle=styles[i % len(styles)],
                label=f"Fold {i+1} (AUC={auc_val:.3f})")
    ax.plot([0,1],[0,1],"k--",alpha=0.4)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curves — Dual-Stream CNN (5 folds)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {roc_path}")

    # ── Plot 3: Params vs AUC scatter ──
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"]
    for i, (mn, vals) in enumerate(comparison_models.items()):
        ax.scatter(vals["params"]/1e6, vals["roc_auc"], s=140, color=colors[i], zorder=5, label=mn)
        ax.annotate(mn, (vals["params"]/1e6, vals["roc_auc"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.set_xlabel("Parameters (Millions)")
    ax.set_ylabel("Mean AUC-ROC (5-fold)")
    ax.set_title("Parameters vs AUC — All Models")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    scatter_path = os.path.join(OUTPUT_DIR, "params_vs_auc.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"✅ Saved: {scatter_path}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
