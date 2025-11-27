#!/usr/bin/env python3
"""
Test if the custom preprocessing layer is properly registered.
Run this AFTER training to verify the model loads correctly.
"""

import tensorflow as tf
import os
import sys
from keras import saving

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our custom classes
from prediction import FocalLoss, MobileNetPreprocessing

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'models', 'skin_cancer_7class_mobilenet.h5')

print("="*60)
print("🧪 Testing Model Loading")
print("="*60)
print(f"TensorFlow: {tf.__version__}\n")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    print("\nPlease train the model first:")
    print("  jupyter notebook notebook/skin_cancer_evaluation.ipynb")
    sys.exit(1)

try:
    print(f"📂 Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'FocalLoss': FocalLoss,
            'MobileNetPreprocessing': MobileNetPreprocessing
        },
        compile=False
    )
    print("✅ Model loaded successfully!\n")
    
    # Test prediction
    print("🧪 Testing prediction...")
    import numpy as np
    test_input = np.random.rand(1, 96, 96, 3).astype(np.float32) * 255
    pred = model.predict(test_input, verbose=0)
    print(f"✅ Prediction works! Output shape: {pred.shape}")
    print(f"   Probabilities sum: {pred.sum():.4f} (should be ~1.0)")
    
    print("\n" + "="*60)
    print("🎉 SUCCESS! Model is ready for API use!")
    print("="*60)
    print("\nYou can now start the API:")
    print("  uvicorn src.api:app --reload --host 0.0.0.0 --port 8000")
    
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nThis means the model needs to be retrained.")
    print("Run all cells in: notebook/skin_cancer_evaluation.ipynb")
    sys.exit(1)
