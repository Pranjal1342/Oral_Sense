# AFI & WLI Oral Cancer Screening Models

This repository contains a machine learning pipeline for screening oral cancer using Autofluorescence Imaging (AFI), White Light Imaging (WLI), and clinical metadata. The models are specifically designed and constrained for edge deployment on low-resource microcontrollers like the **ESP32-S3**.

## Recent Updates: Unbiased Validation Pipeline
We recently overhauled the pipeline to strictly validate whether the Dual-Stream CNN is learning genuine biology or accidentally relying on camera/crop cues (a known issue in unconstrained medical datasets).

### The Evaluation Scripts
1. **`build_cache.py`**:
   * Reads raw RGB images and dynamically generates "pseudo-AFI" using CLAHE and contrast ratios.
   * Extracts two strictly controlled image crops per file: a **Region of Interest (ROI)** crop based on Otsu saturation, and a fallback **CENTER** crop.
   * Builds un-augmented `96x96` caches mapped by camera EXIF and quantization tables to ensure strict cluster-level data separation.
2. **`run_a_paired_cnn.py` (Run A)**:
   * The definitive, unbiased training script.
   * **No Metadata Leakage**: Clinical metadata has been completely stripped to evaluate pure image biology.
   * **Perfect Class Balance**: Forces a 1:1 balance in every single training fold by dynamically subsampling the Healthy class to match the Cancer class.
   * **Dynamic Augmentation**: Applies real-time GPU-based random flips, rotations, and contrast jittering to the 11,480 base images, preventing data leakage across folds.
3. **`compute_drop_table.py`**:
   * Evaluates the statistical difference between the ROI cache predictions and the CENTER cache predictions.
   * Uses a rigorous **Paired Group-Bootstrap** (N=1000 resamples over 3 random seeds) to compute the 95% Confidence Interval. 
   * Definitively tests if the ROI provides statistically significant biological signal over a dumb center crop.

*Note: `oral_cancer_dual_stream.py` is preserved in the repository as a historical reference of the original V2 architecture.*

## Project Context
- **Hardware Target**: ESP32-S3 (240MHz, 8MB Flash, 8MB PSRAM)
- **Input**: 96×96 images 
- **Target Constraints**: INT8 TFLite model < 1MB, <200ms inference, offline execution
- **Evaluation**: 5-Fold Stratified Cross-Validation (suitable for small medical datasets)

## Data Layout Requirements

To build the caches, you must organize your original, un-augmented base WLI images into the following folder structure:

```
data/
  wli/
    cancer/        ← Raw WLI images of suspicious/malignant lesions
    non_cancer/    ← Raw WLI images of healthy tissue
```

*Note: `build_cache.py` will automatically filter out any pre-augmented offline images (e.g. `cancer_aug_*.jpg`) to prevent data leakage during cross-validation.*

## Setup & Usage

### 1. Create a Conda Environment
Using [Conda](https://docs.conda.io/en/latest/) is the recommended way to manage dependencies.

```bash
# Create a new conda environment
conda create -n afi-screening python=3.10 -y

# Activate the environment
conda activate afi-screening

# Install the required Python dependencies
pip install -r requirements.txt
```

### 2. Build the ROI and Center Caches
Generate the `96x96` caches and the `cache_manifest.csv` cluster mapping:
```bash
python build_cache.py
```

### 3. Run the Dual-Stream CNN
Train the 5-fold Stratified CNN on both caches natively on your GPU:
```bash
python run_a_paired_cnn.py
```
*Outputs: `oof_ROI.csv` and `oof_CENTER.csv`.*

### 4. Evaluate the Decision Rule
Run the paired bootstrap script to test statistical significance:
```bash
python compute_drop_table.py
```

## Clinical Metrics
The pipeline optimizes for and reports on metrics critical for medical screening:
- **Sensitivity (Recall)**: Prioritized over specificity. Missing a true cancer case (False Negative) is heavily penalized.
- **Specificity**
- **ROC-AUC (Area Under Curve)**: Inherently immune to class imbalance during OOF test evaluation.
