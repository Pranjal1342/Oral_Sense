# AFI & WLI Oral Cancer Screening Models

This repository contains a machine learning pipeline for screening oral cancer using Autofluorescence Imaging (AFI), White Light Imaging (WLI), and clinical metadata. The models are specifically designed and constrained for edge deployment on low-resource microcontrollers like the **ESP32-S3**.

## Project Context
- **Hardware Target**: ESP32-S3 (240MHz, 8MB Flash, 8MB PSRAM)
- **Input**: 96×96 images + 4D patient metadata (Age, Smoking, Betel Chewing, Alcohol)
- **Target Constraints**: INT8 TFLite model < 1MB, <200ms inference, offline execution
- **Evaluation**: 5-Fold Stratified Cross-Validation (suitable for small medical datasets)

## Repository Structure

- `oral_cancer_comparison.py`: **Version 1 (V1)** — Single-stream pipeline comparing a custom lightweight AFI-CNN against standard baselines (MobileNetV1, MobileNetV2, EfficientNetB0, and ViT-B/16). It uses grayscale AFI images and clinical metadata.
- `oral_cancer_dual_stream.py`: **Version 2 (V2)** — Our second version of comparison. It introduces a dual-stream architecture that fuses White Light Imaging (RGB), Autofluorescence Imaging (Grayscale), and clinical metadata.
- `requirements.txt`: Required Python dependencies.

## Model Architectures

### 1. Custom AFI-CNN (Primary Edge Candidate)
Designed from scratch for 96×96 grayscale AFI input and metadata fusion. It uses depthwise separable convolutions to drastically reduce FLOPs, keeping the INT8 quantized model size around **~0.2 MB**.

### 2. Dual-Stream CNN (Version 2 Comparison)
As our second version of model comparison, this architecture fuses three modalities:
- WLI Branch (RGB Image)
- AFI Branch (Grayscale Image)
- Metadata Branch (Clinical Features)
All three embeddings are concatenated and passed through dense layers to predict oral cancer risk.

### 3. Baselines Evaluated
- **MobileNetV1 (α=0.5)**: Baseline edge candidate (~0.45 MB INT8).
- **MobileNetV2**: Stronger edge candidate (~1.4 MB INT8).
- **EfficientNetB0**: Upper-bound edge reference (~4.0 MB INT8).
- **ViT-B/16**: Accuracy ceiling reference only. At ~86 MB INT8, it is physically impossible to deploy on the ESP32-S3.

## Data Layout Requirements

To train the models, you must organize your image data into the following folder structure:

```
data/
  wli/
    cancer/        ← WLI images of suspicious/malignant lesions
    non_cancer/    ← WLI images of healthy tissue
  afi/
    cancer/        ← AFI images of suspicious/malignant lesions
    non_cancer/    ← AFI images of healthy tissue
```

*Note: For the single-stream `oral_cancer_comparison.py`, you can place your images directly into `data/cancer/` and `data/non_cancer/`.*

## Setup & Usage

### 1. Create a Conda Environment & Install Dependencies
It is highly recommended to use Python 3.9 - 3.11. TensorFlow 2.13.1 is required to remain compatible with Keras 2 (needed for the ViT implementation).

Using [Conda](https://docs.conda.io/en/latest/) is the recommended way to manage dependencies. Run the following commands in your terminal to set up and activate the environment:

```bash
# Create a new conda environment named 'afi-screening' with Python 3.10
conda create -n afi-screening python=3.10 -y

# Activate the environment
conda activate afi-screening

# Install the required Python dependencies
pip install -r requirements.txt
```

### 2. Run the Comparison Pipeline
Trains and compares the single-stream Custom AFI-CNN against baseline architectures.
```bash
python oral_cancer_comparison.py
```
*Outputs (saved to `results/`): Summary metrics CSV, comparison bar charts, ROC curves, and Parameters vs AUC scatter plot.*

### 3. Run the Dual-Stream Pipeline
Trains the dual-stream model on paired WLI and AFI images.
```bash
python oral_cancer_dual_stream.py
```
*Outputs (saved to `results_dual/`): The trained `.keras` model, summary metrics, and performance plots.*

## Clinical Metrics
Both scripts optimize for and report on metrics critical for medical screening:
- **Sensitivity (Recall)**: Prioritized over specificity. Missing a true cancer case (False Negative) is heavily penalized.
- **Specificity**
- **ROC-AUC**
- **Youden's J Statistic**: Used to automatically calibrate the optimal classification threshold.
