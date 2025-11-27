#!/usr/bin/env python3
"""
Fix model compatibility by reloading and resaving in current TensorFlow version.
Run this if you get "Unknown layer: 'TrueDivide'" errors.
"""

import tensorflow as tf
import os
import sys

# Custom Focal Loss
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        
    def call(self, y_true, y_pred):
        NUM_CLASSES = 7
        y_true = tf.cast(y_true, tf.int32)
        y_true_smooth = tf.one_hot(y_true, NUM_CLASSES)
        y_true_smooth = y_true_smooth * (1 - self.label_smoothing) + self.label_smoothing / NUM_CLASSES
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true_smooth * tf.math.log(y_pred)
        weight = self.alpha * y_true_smooth * tf.pow(1 - y_pred, self.gamma)
        focal_loss = weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


def fix_model(model_path):
    """Load and resave model in current TensorFlow version."""
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run the notebook first to train the model.")
        return False
    
    print(f"Loading model from: {model_path}")
    print(f"TensorFlow version: {tf.__version__}")
    
    try:
        # Try loading with compile=False (most compatible)
        print("\n1️⃣ Attempting to load with compile=False...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'FocalLoss': FocalLoss},
            compile=False
        )
        print("✅ Model loaded successfully!")
        
        # Create backup
        backup_path = model_path.replace('.h5', '_backup.h5')
        if not os.path.exists(backup_path):
            print(f"\n2️⃣ Creating backup: {backup_path}")
            import shutil
            shutil.copy2(model_path, backup_path)
            print("✅ Backup created!")
        
        # Resave in current format
        print(f"\n3️⃣ Resaving model in current TensorFlow format...")
        model.save(model_path)
        print("✅ Model resaved successfully!")
        
        # Test loading
        print("\n4️⃣ Testing reload...")
        test_model = tf.keras.models.load_model(
            model_path,
            custom_objects={'FocalLoss': FocalLoss},
            compile=False
        )
        print("✅ Model loads correctly!")
        
        print("\n" + "="*60)
        print("✨ Model fixed successfully!")
        print("="*60)
        print(f"Original saved to: {backup_path}")
        print(f"Fixed model at: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're using TensorFlow 2.13+")
        print("2. Retrain the model by running the notebook")
        print("3. Check if the model file is corrupted")
        return False


if __name__ == "__main__":
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_PATH, 'models', 'skin_cancer_7class_mobilenet.h5')
    
    print("="*60)
    print("🔧 Model Compatibility Fixer")
    print("="*60)
    
    success = fix_model(MODEL_PATH)
    sys.exit(0 if success else 1)
