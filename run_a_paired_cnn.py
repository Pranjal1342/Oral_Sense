import os
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

tf.config.experimental.enable_op_determinism()

IMG_SIZE = (96, 96)
BATCH_SIZE = 8
N_FOLDS = 5
EPOCHS = 100
EARLY_STOP_PAT = 12
L2_PENALTY = 1e-4
DROPOUT_RATE = 0.3
SEED = 42

def hash_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def build_dual_stream_cnn():
    wli_input = layers.Input(shape=IMG_SIZE + (3,), name="wli_input")
    afi_input = layers.Input(shape=IMG_SIZE + (1,), name="afi_input")

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

    fused = layers.Concatenate()([wli_embedding, afi_embedding])
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(L2_PENALTY))(fused)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    output = layers.Dense(1, activation="sigmoid", name="risk_score")(x)

    return models.Model(inputs=[wli_input, afi_input], outputs=output)

def augment_fn(wli_img, afi_img, label):
    combined = tf.concat([wli_img, afi_img], axis=-1)
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    combined = tf.image.rot90(combined, k=k)
    
    wli_img, afi_img = combined[..., :3], combined[..., 3:]
    
    wli_img = tf.image.random_brightness(wli_img, max_delta=0.08)
    wli_img = tf.image.random_contrast(wli_img, lower=0.92, upper=1.08)
    afi_img = tf.image.random_brightness(afi_img, max_delta=0.08)
    afi_img = tf.image.random_contrast(afi_img, lower=0.92, upper=1.08)
    
    wli_img = tf.clip_by_value(wli_img, -1.0, 1.0)
    afi_img = tf.clip_by_value(afi_img, -1.0, 1.0)
    return (wli_img, afi_img), label

def load_cache_to_ram(cache_dir, filenames, labels):
    N = len(filenames)
    wli_all = np.zeros((N, 96, 96, 3), dtype=np.float32)
    afi_all = np.zeros((N, 96, 96, 1), dtype=np.float32)
    
    for i, (f, lbl) in enumerate(zip(filenames, labels)):
        l_str = 'cancer' if lbl == 1 else 'non_cancer'
        base = os.path.splitext(f)[0]
        p1 = os.path.join(cache_dir, '96x96', l_str, f"{base}.npz")
        p2 = os.path.join(cache_dir, '96x96', l_str, f"{f}.npz")
        if os.path.exists(p1): path = p1
        elif os.path.exists(p2): path = p2
        else: raise FileNotFoundError(f"Missing {p1} or {p2}")
            
        with np.load(path) as data:
            w = data['wli'].astype(np.float32) / 127.5 - 1.0
            a = data['afi'].astype(np.float32) / 127.5 - 1.0
            if a.ndim == 2: a = np.expand_dims(a, -1)
            wli_all[i] = w
            afi_all[i] = a
    return wli_all, afi_all

def run_cache_training(cache_dir, cache_name):
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    
    print(f"\n{'='*60}\nRunning on {cache_name} (Dir: {cache_dir})\n{'='*60}")
    
    manifest = pd.read_pickle('cache_manifest.pkl')
    manifest = manifest.sort_values('file').reset_index(drop=True)
    
    filenames = manifest['file'].values
    labels = manifest['class'].values
    cluster_ids = manifest['cluster_id'].values
    fallback_flags = manifest['fallback'].values
    
    if 'phash_group' in manifest.columns:
        phash_groups = manifest['phash_group'].values
    else:
        phash_groups = np.array(['NONE']*len(filenames))
    
    print(f"Loading {len(filenames)} images into RAM from {cache_dir}...")
    wli_all, afi_all = load_cache_to_ram(cache_dir, filenames, labels)
    
    print(f"Hashing 20 non-fallback files from {cache_dir} (RAM check)...")
    non_fallback_idx = np.where(~fallback_flags)[0]
    for idx in non_fallback_idx[:20]:
        h = hashlib.sha256(wli_all[idx].tobytes() + afi_all[idx].tobytes()).hexdigest()
        print(f"File {filenames[idx]}: {h}")
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    oof_preds = np.zeros(len(filenames))
    oof_folds = np.zeros(len(filenames), dtype=int)
    
    scored_count = 0
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(filenames, labels)):
        tf.random.set_seed(SEED + fold_idx)
        np.random.seed(SEED + fold_idx)
        
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + fold_idx)
        for inner_train_idx_rel, early_stop_idx_rel in inner_skf.split(train_idx, labels[train_idx]):
            inner_train_idx = train_idx[inner_train_idx_rel]
            early_stop_idx = train_idx[early_stop_idx_rel]
            break
            
        tc = inner_train_idx[labels[inner_train_idx] == 1]
        th = inner_train_idx[labels[inner_train_idx] == 0]
        np.random.shuffle(th)
        th_kept = th[:len(tc)]
        final_train_idx = np.concatenate([tc, th_kept])
        np.random.shuffle(final_train_idx)
        
        esc = early_stop_idx[labels[early_stop_idx] == 1]
        esh = early_stop_idx[labels[early_stop_idx] == 0]
        np.random.shuffle(esh)
        esh_kept = esh[:len(esc)]
        final_es_idx = np.concatenate([esc, esh_kept])
        np.random.shuffle(final_es_idx)
        
        final_test_idx = test_idx
        oof_folds[final_test_idx] = fold_idx + 1
        scored_count += len(final_test_idx)
        
        print(f"\nFold {fold_idx+1} [Cache: {cache_dir}]")
        print(f"  Inner Train : {len(final_train_idx)} images")
        print(f"  Early Stop  : {len(final_es_idx)} images")
        print(f"  OOF Test    : {len(final_test_idx)} images")
        
        train_ds = tf.data.Dataset.from_tensor_slices((wli_all[final_train_idx], afi_all[final_train_idx], labels[final_train_idx]))
        train_ds = train_ds.shuffle(len(final_train_idx)).map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        es_ds = tf.data.Dataset.from_tensor_slices((wli_all[final_es_idx], afi_all[final_es_idx], labels[final_es_idx]))
        es_ds = es_ds.map(lambda w,a,l: ((w,a), l)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices((wli_all[final_test_idx], afi_all[final_test_idx], labels[final_test_idx]))
        test_ds = test_ds.map(lambda w,a,l: ((w,a), l)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        model = build_dual_stream_cnn()
        
        cosine_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=2e-4,
            decay_steps=EPOCHS * (len(final_train_idx) // BATCH_SIZE),
            alpha=1e-6
        )
        smoothed_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(cosine_lr), loss=smoothed_loss, metrics=['accuracy'])
        
        model.fit(
            train_ds,
            validation_data=es_ds,
            epochs=EPOCHS,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PAT, restore_best_weights=True)],
            verbose=1
        )
        
        preds = model.predict(test_ds, verbose=0).ravel()
        oof_preds[final_test_idx] = preds
        
    auc = roc_auc_score(labels, oof_preds)
    print(f"\n>>> {cache_name} Overall OOF AUC: {auc:.4f}")
    print(f"Total manifest files: {len(filenames)}. Total images scored: {scored_count}.")
    
    out_df = pd.DataFrame({
        'path': filenames,
        'fold': oof_folds,
        'prob': oof_preds,
        'label': labels,
        'cluster_id': cluster_ids,
        'phash_group': phash_groups,
        'fallback': fallback_flags
    })
    out_df.to_csv(f"oof_{cache_name}.csv", index=False)
    return out_df

if __name__ == "__main__":
    print("Starting Run A with strict validation sets...")
    df_roi = run_cache_training("data_cached_roi", "ROI")
    df_center = run_cache_training("data_cached_center", "CENTER")

    print("\nRun A completed! Please use compute_drop_table.py to analyze the results.")
