# MRI Router: Cascade Brain MRI Diagnosis System

A comprehensive deep learning diagnostic system for brain MRI slice analysis. The router implements a clinically motivated **Two-Stage Cascade Architecture**:
1. **Stage 1 (Primary - Dementia Severity Assessment)**: Evaluates brain scans using **EfficientNetV2-S** for dataset-defined dementia categories (*Non Demented*, *Very mild Dementia*, *Mild Dementia*, *Moderate Dementia*).
2. **Stage 2 (Reflex - Brain Tumor Screening)**: Triggered **ONLY** when Stage 1 indicates a normal brain (*Non Demented*). If dementia pathology is identified, tumor screening is bypassed as per cascade protocol. Includes **5-layer gradient-fused Grad-CAM** localization.

---

## Architecture Overview

```
                      [Input Brain MRI Slice]
                                 │
                                 ▼
                 Stage 1: Dementia Model (EfficientNetV2-S)
                                 │
               ┌─────────────────┴─────────────────┐
               ▼                                   ▼
      [Dementia Detected]                 [Non-Demented (Normal)]
               │                                   │
               │ (Primary diagnosis confirmed)     ▼
               ▼                         Stage 2: Brain Tumor CNN
        Stage 2 Bypassed                           │
        (Screening Skipped)                ┌───────┴───────┐
                                           ▼               ▼
                                    [Tumor Positive] [Tumor Negative]
                                     (Multi-Layer CAM) (Normal Brain)
```

---

## Quick Start

### 1. Launch the Interactive GUI Window (Recommended)
You can directly launch the graphical user interface by running:

```bash
python predict.py
```
*(or `python gui_predict.py`)*

In the GUI, simply click **Browse Image...** or paste your image path and click **Run Cascade Diagnosis**.

### 2. Run via Command Line (Terminal Mode)
```bash
python predict.py --cli --image "path/to/mri.jpg"
```

---

## Directory Structure

```
MRI_Router/
├── predict.py                  # Global entrypoint (launches GUI or CLI)
├── gui_predict.py              # Interactive desktop GUI application
├── with dementia.jpeg          # Sample MRI scan (Dementia positive)
├── with out dementia.jpeg      # Sample MRI scan (Dementia negative)
│
├── dementia_mri_project/       # Stage 1: Dementia classification system
│   ├── checkpoints/            # Trained model weights (best_model.pth)
│   ├── src/                    # Data pipeline, training & inference code
│   │   ├── gui.py              # Cascade GUI implementation
│   │   ├── inference.py        # Stage 1 inference engine
│   │   ├── tumor_detector.py   # Stage 2 PyTorch reflex engine
│   │   └── ...
│   └── predictions/            # Generated Grad-CAM heatmaps & JSON outputs
│
└── brain-_tumor_detection/     # Stage 2: Brain Tumor detection components
    ├── models/                 # Pretrained weights (.h5 / .model)
    ├── gradcam.py              # Multi-layer Grad-CAM algorithm
    └── ...
```

---

## Explainable AI (XAI)
- **Stage 1**: Multi-layer Grad-CAM for cortical atrophy and dementia patterns.
- **Stage 2**: 5-layer gradient-fused Grad-CAM (`conv0`, `bn0`, `activation`, `max_pool0`, `max_pool1`) with pre-sigmoid logit backpropagation and anatomical skull masking.
