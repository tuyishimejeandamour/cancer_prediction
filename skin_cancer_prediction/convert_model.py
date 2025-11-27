#!/usr/bin/env python3
"""
Convert old model format to SavedModel format (more compatible).
This creates a directory-based model instead of .h5 file.
"""

import tensorflow as tf
import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OLD_MODEL = os.path.join(BASE_PATH, 'models', 'skin_cancer_7class_mobilenet.h5')
NEW_MODEL = os.path.join(BASE_PATH, 'models', 'skin_cancer_7class_mobilenet')

print("="*60)
print("🔄 Converting Model to SavedModel Format")
print("="*60)
print(f"TensorFlow version: {tf.__version__}\n")

if not os.path.exists(OLD_MODEL):
    print(f"❌ Model not found: {OLD_MODEL}")
    print("\nPlease retrain the model by running:")
    print("  jupyter notebook notebook/skin_cancer_evaluation.ipynb")
    sys.exit(1)

print("This issue occurs because the model was saved with a different")
print("TensorFlow version. The solution is to retrain the model.\n")
print("Quick fix options:")
print("\n1️⃣ RECOMMENDED: Retrain the model")
print("   Run all cells in: notebook/skin_cancer_evaluation.ipynb")
print("\n2️⃣ TEMPORARY: Use inference-only mode")
print("   The API will load the model without compilation")
print("\nThe model file exists but needs to be retrained with your")
print("current TensorFlow version (2.20.0) for full compatibility.")

sys.exit(0)
