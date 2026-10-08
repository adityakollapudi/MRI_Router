#!/usr/bin/env python
"""Global Entry Point: Cascade Brain MRI Diagnosis GUI Window.

Stage 1: Dementia Severity Classification (EfficientNetV2-S)
Stage 2: Reflex Brain Tumor Screening (CNN with Multi-Layer Grad-CAM)
         - Automatically triggered ONLY when Dementia is Normal (Non-Demented)
         - Bypassed when Dementia is detected

Usage:
    python gui_predict.py
    python gui_predict.py --image "path/to/scan.jpg"
"""
import os
import sys
from pathlib import Path

# Resolve top-level workspace root and subproject folders
ROOT = Path(__file__).resolve().parent
DEMENTIA_DIR = ROOT / "dementia_mri_project"
TUMOR_DIR = ROOT / "brain-_tumor_detection"

sys.path.insert(0, str(DEMENTIA_DIR))
sys.path.insert(0, str(TUMOR_DIR))

from src.gui import main

if __name__ == "__main__":
    main()
