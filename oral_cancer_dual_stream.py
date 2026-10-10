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
BATCH_SIZE      = 32               
N_FOLDS         = 5
EPOCHS          = 100
EARLY_STOP_PAT  = 12
L2_PENALTY      = 1e-4
DROPOUT_RATE    = 0.3
OUTPUT_DIR      = "results_dual_v6"
SEED            = 42

# Data is now manually oversampled to 1:1 (8878 cancer / 8878 non_cancer).
# Training is balanced — class weights are neutral.
# Logit offset -1.225 is STILL applied post-training to recalibrate to the
# real-world 3.4:1 prior (balanced training ≠ balanced real world).
CLINICAL_CLASS_WEIGHTS = {
    0: 1.0,   # Healthy   (balanced — no penalty needed)
    1: 1.0,   # Malignant (balanced — no penalty needed)
}
LOGIT_OFFSET    = 1.225   # ln(3.4) — real-world prior correction, unchanged

# Real patient metadata CSV
METADATA_CSV    = os.path.join(DATA_DIR, "image_metadata_mapping.csv")

# Oversampling no longer needed — data is already 1:1 balanced
USE_OVERSAMPLING = True

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
            if fname.lower().startswith("cancer_aug_"):
                continue  # Prevent fold leakage from augmented copies
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
    """Fallback: generate synthetic metadata to simulate clinical priors."""
    rng = np.random.default_rng(seed)
    meta = np.zeros((n_samples, N_META_FEATURES), dtype=np.float32)
    for i, label in enumerate(labels):
        if label == 1:
            age, smoking, betel, alcohol = rng.integers(40, 76), rng.choice([0, 1], p=[0.3, 0.7]), rng.choice([0, 1, 2], p=[0.2, 0.3, 0.5]), rng.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
        else:
            age, smoking, betel, alcohol = rng.integers(25, 65), rng.choice([0, 1], p=[0.6, 0.4]), rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15]), rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
        meta[i] = [(age - 25) / 50.0, float(smoking), betel / 2.0, alcohol / 2.0]
    return meta


def load_real_metadata(filepaths_wli, csv_path):
    """
    Load real patient metadata from image_metadata_mapping.csv.

    CSV columns used: WLI_Filename, Age, Smoking, Chewing_Betel_Quid, Alcohol
    Encoded to N_META_FEATURES=4 floats:
      [0] age_norm  = (Age - 18) / 82  -> [0, 1]
      [1] smoking   = 1 if Yes else 0
      [2] betel     = 1 if Yes else 0
      [3] alcohol   = 1 if Yes else 0

    Imputation strategy:
      - Exact CSV match          -> real values
      - cancer_aug_* files       -> cancer-patient population medians from CSV
      - non_cancer / unmatched   -> [median_age, 0.5, 0.5, 0.5]
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["WLI_Filename"] = df["WLI_Filename"].str.strip()

    def yn(val):
        return 1.0 if str(val).strip().lower() == "yes" else 0.0

    # Encode all CSV rows
    df["age_norm"] = pd.to_numeric(df["Age"], errors="coerce")
    df["age_norm"] = ((df["age_norm"] - 18.0) / 82.0).clip(0.0, 1.0)
    df["smoking_enc"] = df["Smoking"].apply(yn)
    df["betel_enc"]   = df["Chewing_Betel_Quid"].apply(yn)
    df["alcohol_enc"] = df["Alcohol"].apply(yn)

    # Cancer-patient population medians (for augmented copies)
    cancer_medians = [
        float(df["age_norm"].median()),
        float(df["smoking_enc"].mean()),    # mean of 0/1 = rate
        float(df["betel_enc"].mean()),
        float(df["alcohol_enc"].mean()),
    ]
    # General population median (for non-cancer unmatched)
    general_impute = [float(df["age_norm"].median()), 0.5, 0.5, 0.5]

    lookup = df.set_index("WLI_Filename")
    meta   = np.zeros((len(filepaths_wli), N_META_FEATURES), dtype=np.float32)

    matched = aug_imputed = general_imputed = 0

    for i, fp in enumerate(filepaths_wli):
        bname = os.path.basename(fp)

        if bname in lookup.index:
            # Exact match in CSV
            row = lookup.loc[bname]
            try:
                age_norm = float(np.clip((float(row["Age"]) - 18.0) / 82.0, 0.0, 1.0))
            except (ValueError, TypeError):
                age_norm = cancer_medians[0]
            meta[i] = [age_norm, yn(row["Smoking"]),
                       yn(row["Chewing_Betel_Quid"]), yn(row["Alcohol"])]
            matched += 1

        elif bname.startswith("cancer_aug_"):
            # Oversampled cancer copy — use cancer population medians
            meta[i] = cancer_medians
            aug_imputed += 1

        else:
            # Non-cancer or unmatched original — maximum uncertainty
            meta[i] = general_impute
            general_imputed += 1

    print(f"  Metadata loaded:")
    print(f"    Exact CSV match   : {matched:>5}")
    print(f"    cancer_aug_* fill : {aug_imputed:>5}  (cancer population medians)")
    print(f"    General imputation: {general_imputed:>5}  (median age, 0.5/0.5/0.5)")
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
        ds = ds.shuffle(buffer_size=min(2000, len(labels)), seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def make_balanced_dataset(filepaths_wli, filepaths_afi, labels, metadata):
    """
    Balances a 3.4:1 class-imbalanced paired dataset using tf.data pipeline oversampling.

    Strategy:
      - Split into two class-specific datasets (cancer / non_cancer)
      - Repeat the minority class infinitely with augmentation ON
        so every repeat gets a different random flip/rotation/contrast
      - sample_from_datasets draws 50% from each class per batch
      - steps_per_epoch = (2 * majority_count) // BATCH_SIZE caps the epoch
        so training doesn't run forever

    Returns: (balanced_ds, steps_per_epoch)
    """
    cancer_idx     = np.where(labels == 1)[0]
    non_cancer_idx = np.where(labels == 0)[0]

    def _make_class_ds(idx, augment, repeat=False):
        wli_f  = filepaths_wli[idx]
        afi_f  = filepaths_afi[idx]
        lbl    = labels[idx]
        meta   = metadata[idx]

        path_ds = tf.data.Dataset.from_tensor_slices((wli_f, afi_f, lbl))
        decoded = path_ds.map(
            lambda w, a, l: decode_paired_images(w, a, l, augment),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        meta_ds = tf.data.Dataset.from_tensor_slices(meta.astype(np.float32))
        lbl_ds  = tf.data.Dataset.from_tensor_slices(lbl.astype(np.float32))

        ds = tf.data.Dataset.zip((decoded, meta_ds, lbl_ds))
        ds = ds.map(lambda imgs, m, l: ((imgs[0], imgs[1], m), l))
        if repeat:
            ds = ds.shuffle(buffer_size=len(idx), seed=SEED, reshuffle_each_iteration=True)
            ds = ds.repeat()   # infinite — fresh shuffle + augmentation every repeat
        return ds

    cancer_ds     = _make_class_ds(cancer_idx,     augment=True,  repeat=True)
    non_cancer_ds = _make_class_ds(non_cancer_idx, augment=True,  repeat=True)

    # Draw 50% from each class every batch
    balanced_ds = tf.data.Dataset.sample_from_datasets(
        [cancer_ds, non_cancer_ds],
        weights=[0.5, 0.5],
        seed=SEED,
        stop_on_empty_dataset=False
    )
    balanced_ds = balanced_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Cap each epoch: 2 × majority so both classes are seen roughly once
    majority_count = max(len(cancer_idx), len(non_cancer_idx))
    steps_per_epoch = (2 * majority_count) // BATCH_SIZE

    print(f"   Oversampling active:")
    print(f"     cancer pairs     : {len(cancer_idx)}")
    print(f"     non-cancer pairs : {len(non_cancer_idx)} (repeated w/ fresh augmentation)")
    print(f"     steps/epoch      : {steps_per_epoch}")

    return balanced_ds, steps_per_epoch

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
    
    # ── UNDERSAMPLING ──────────────────────────────────────────────────────────
    # Reduce healthy (0) count to exactly match cancer (1) count
    cancer_idx = np.where(labels == 1)[0]
    non_cancer_idx = np.where(labels == 0)[0]
    if len(non_cancer_idx) > len(cancer_idx):
        print(f"Undersampling healthy images from {len(non_cancer_idx)} to {len(cancer_idx)}")
        np.random.shuffle(non_cancer_idx)
        non_cancer_idx = non_cancer_idx[:len(cancer_idx)]
    
    keep_idx = np.concatenate([cancer_idx, non_cancer_idx])
    np.random.shuffle(keep_idx)
    filepaths_wli = filepaths_wli[keep_idx]
    filepaths_afi = filepaths_afi[keep_idx]
    labels = labels[keep_idx]
    # ──────────────────────────────────────────────────────────────────────────
    if len(labels) == 0:
        print("No paired images found. Please organize your folders properly.")
        exit(1)

    # ── Corrupt-file validation (with cache) ────────────────────────────────
    # First run: fully decodes every image with PIL (slow, ~3-5 min for 8K pairs).
    # Saves valid pair paths to a cache file.
    # Subsequent runs: loads cache instantly — no rescan needed.
    from PIL import Image, UnidentifiedImageError
    import warnings, json
    from tqdm import tqdm

    CACHE_FILE = os.path.join(OUTPUT_DIR, "valid_pairs_cache.json")

    def _is_valid(path):
        # TF decode_image supports only these formats
        TF_OK = {"JPEG", "PNG", "GIF", "BMP", "MPO"}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with Image.open(path) as im:
                    im.load()
                    if im.format not in TF_OK:
                        return False   # e.g. WebP, TIFF, HEIC — TF will crash
            return True
        except Exception:
            return False


    # Check if cache exists and covers the same set of WLI files
    cache_hit = False
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        cached_wli = set(cache.get("wli", []))
        current_wli = set(filepaths_wli.tolist())
        if cached_wli == current_wli:
            valid_wli = np.array(cache["valid_wli"])
            valid_afi = np.array(cache["valid_afi"])
            valid_lbl = np.array(cache["valid_labels"])
            print(f"  Loaded valid-pairs cache ({len(valid_wli)} pairs). Delete {CACHE_FILE} to rescan.")
            cache_hit = True

    if not cache_hit:
        print(f"  Scanning {len(filepaths_wli)} pairs for corrupt images (runs once, then cached)...")
        valid_wli_list, valid_afi_list, valid_lbl_list = [], [], []
        removed = 0
        for wli_p, afi_p, lbl in tqdm(zip(filepaths_wli, filepaths_afi, labels),
                                       total=len(labels), desc="Validating"):
            if _is_valid(wli_p) and _is_valid(afi_p):
                valid_wli_list.append(wli_p)
                valid_afi_list.append(afi_p)
                valid_lbl_list.append(int(lbl))
            else:
                removed += 1
        valid_wli = np.array(valid_wli_list)
        valid_afi = np.array(valid_afi_list)
        valid_lbl = np.array(valid_lbl_list)
        print(f"  {removed} corrupt pair(s) removed. {len(valid_wli)} clean pairs remaining.")
        # Save cache
        with open(CACHE_FILE, "w") as f:
            json.dump({"wli": filepaths_wli.tolist(),
                       "valid_wli": valid_wli.tolist(),
                       "valid_afi": valid_afi.tolist(),
                       "valid_labels": valid_lbl.tolist()}, f)
        print(f"  Cache saved to {CACHE_FILE}\n")

    filepaths_wli = valid_wli
    filepaths_afi = valid_afi
    labels        = valid_lbl
    # ────────────────────────────────────────────────────────────────────────


    # Load real patient metadata from CSV (imputes missing entries)
    if os.path.exists(METADATA_CSV):
        metadata = load_real_metadata(filepaths_wli, METADATA_CSV)
    else:
        print(f"  Warning: {METADATA_CSV} not found — using synthetic metadata")
        metadata = generate_synthetic_metadata(len(labels), labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    roc_data = []
    
    print(f"\nTraining Dual-Stream Model on {len(labels)} perfect pairs...\n")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(filepaths_wli, labels)):
        print(f"{'='*60}\nFold {fold_idx + 1}/{N_FOLDS}\n{'='*60}")
        
        if USE_OVERSAMPLING:
            train_ds, steps_per_epoch = make_balanced_dataset(
                filepaths_wli[train_idx], filepaths_afi[train_idx],
                labels[train_idx], metadata[train_idx]
            )
            # Class weights neutral — balance is handled by pipeline sampling
            fold_class_weights = {0: 1.0, 1: 1.0}
        else:
            train_ds = make_dual_dataset(
                filepaths_wli[train_idx], filepaths_afi[train_idx],
                labels[train_idx], metadata[train_idx], augment=True
            )
            steps_per_epoch  = None   # let Keras infer from dataset length
            fold_class_weights = CLINICAL_CLASS_WEIGHTS

        val_ds = make_dual_dataset(
            filepaths_wli[val_idx], filepaths_afi[val_idx],
            labels[val_idx], metadata[val_idx], augment=False
        )
        
        model = build_dual_stream_cnn()

        # ── Cosine Decay LR: starts at 1e-3, decays smoothly to 1e-6 ──
        # steps_per_epoch may be None (non-oversampling path); fall back to
        # a rough estimate based on dataset size and batch size.
        _steps = steps_per_epoch or max(1, len(train_idx) // BATCH_SIZE)
        cosine_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=2e-4,
            decay_steps=EPOCHS * _steps,
            alpha=1e-6   # minimum LR at end of schedule
        )
        # ── Label Smoothing: prevents overconfident hard-0/hard-1 outputs ──
        smoothed_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(cosine_lr),
            loss=smoothed_loss,
            metrics=["accuracy"]
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=EARLY_STOP_PAT, restore_best_weights=True
            ),
        ]
        
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            class_weight=fold_class_weights,
            verbose=1
        )
        
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

    # ── Save uncalibrated model (pre-recalibration checkpoint) ──
    model_path = os.path.join(OUTPUT_DIR, "dual_stream_model.keras")
    model.save(model_path)
    print(f"\n✅ Uncalibrated model saved to {model_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP A: Elkan prior-correction — recalibrate final Dense(1) bias
    # Rationale: class_weight training calibrates p(y=1) for a balanced prior.
    # Subtracting ln(w1/w0) = ln(3.4) = 1.225 from the bias restores calibration
    # to the true clinical 3.4:1 prior WITHOUT retraining.
    # This is baked in before TFLite export so ESP32-S3 needs zero extra math.
    # ─────────────────────────────────────────────────────────────────────────
    output_layer = model.get_layer("risk_score")  # named Dense(1, sigmoid)
    weights, biases = output_layer.get_weights()
    old_bias = float(biases[0])
    biases[0] = biases[0] - LOGIT_OFFSET
    output_layer.set_weights([weights, biases])
    print(f"\n🔧 Logit recalibration applied:")
    print(f"   bias before : {old_bias:.6f}")
    print(f"   offset      : -{LOGIT_OFFSET}  [ln({CLINICAL_CLASS_WEIGHTS[1]}/{CLINICAL_CLASS_WEIGHTS[0]:.4f}) = ln(3.4)]")
    print(f"   bias after  : {float(biases[0]):.6f}")

    # Save calibrated model separately
    calibrated_path = os.path.join(OUTPUT_DIR, "dual_stream_calibrated.keras")
    model.save(calibrated_path)
    print(f"✅ Calibrated model saved to {calibrated_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP B: INT8 TFLite export (from calibrated model)
    # representative_data_gen feeds real training images from the last fold
    # so the quantiser can compute per-layer activation ranges.
    # ─────────────────────────────────────────────────────────────────────────
    # Build a small representative dataset from last fold training indices
    _rep_wli  = filepaths_wli[train_idx[:30]]
    _rep_afi  = filepaths_afi[train_idx[:30]]
    _rep_meta = metadata[train_idx[:30]]

    def representative_data_gen():
        """Yields 30 real samples for INT8 activation range calibration."""
        for i in range(len(_rep_wli)):
            # Decode WLI (RGB)
            wli_raw = tf.io.read_file(_rep_wli[i])
            wli_img = tf.image.decode_image(wli_raw, channels=3, expand_animations=False)
            wli_img = tf.image.resize(wli_img, IMG_SIZE)
            wli_img = tf.cast(wli_img, tf.float32) / 127.5 - 1.0
            wli_img = tf.expand_dims(wli_img, 0)   # (1, 96, 96, 3)

            # Decode AFI (grayscale)
            afi_raw = tf.io.read_file(_rep_afi[i])
            afi_img = tf.image.decode_image(afi_raw, channels=1, expand_animations=False)
            afi_img = tf.image.resize(afi_img, IMG_SIZE)
            afi_img = tf.cast(afi_img, tf.float32) / 127.5 - 1.0
            afi_img = tf.expand_dims(afi_img, 0)   # (1, 96, 96, 1)

            meta = tf.expand_dims(_rep_meta[i], 0).numpy()  # (1, 4)

            # Yield as list matching model input order: [wli, afi, meta]
            yield [wli_img.numpy(), afi_img.numpy(), meta]

    print("\n⚙️  Converting to INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = os.path.join(OUTPUT_DIR, "oral_sense_v2_calibrated.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    tflite_kb = len(tflite_model) / 1024
    print(f"✅ INT8 TFLite saved: {tflite_path}")
    print(f"   Size: {tflite_kb:.1f} KB  ({tflite_kb/1024:.3f} MB)")
    print(f"   ESP32-S3 Flash budget: 8192 KB  →  {8192 - tflite_kb:.0f} KB remaining")

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
