import tensorflow as tf
import numpy as np
import os
from keras import saving
from preprocessing import (
    preprocess_image, 
    IMG_HEIGHT, IMG_WIDTH,
    CLASS_NAMES, CLASS_FULL_NAMES, MALIGNANT_CLASSES, NUM_CLASSES
)


# Custom Preprocessing Layer (must match notebook)
@saving.register_keras_serializable(package="SkinCancer")
class MobileNetPreprocessing(tf.keras.layers.Layer):
    """Custom preprocessing layer for MobileNetV2 that serializes properly"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        # MobileNetV2 preprocessing: scale to [-1, 1]
        x = tf.cast(inputs, tf.float32)
        x = x / 127.5 - 1.0
        return x
    
    def get_config(self):
        return super().get_config()


# Custom Focal Loss for loading the model
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for handling class imbalance"""
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_smooth = tf.one_hot(y_true, NUM_CLASSES)
        y_true_smooth = y_true_smooth * (1 - self.label_smoothing) + self.label_smoothing / NUM_CLASSES
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce = -y_true_smooth * tf.math.log(y_pred)
        weight = self.alpha * y_true_smooth * tf.pow(1 - y_pred, self.gamma)
        focal_loss = weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))


class Predictor:
    """
    7-Class Skin Lesion Predictor with Out-of-Distribution Detection
    """
    def __init__(self, model_path, confidence_threshold=0.60, entropy_threshold=1.3):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            confidence_threshold: Minimum confidence to accept prediction (default 60%)
            entropy_threshold: Maximum entropy to accept prediction (default 1.3)
        """
        # Load with compile=False to avoid compatibility issues
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'FocalLoss': FocalLoss,
                'MobileNetPreprocessing': MobileNetPreprocessing
            },
            compile=False
        )
        # Recompile if needed (optional for inference)
        # self.model.compile()
        self.class_names = CLASS_NAMES
        self.class_full_names = CLASS_FULL_NAMES
        self.malignant_classes = MALIGNANT_CLASSES
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold

    def predict(self, image_array):
        """
        Make a prediction with OOD detection.
        
        Args:
            image_array: numpy array or tensor of the image
            
        Returns:
            dict with prediction results including OOD detection
        """
        processed_image = preprocess_image(image_array)
        predictions = self.model.predict(processed_image, verbose=0)
        probs = predictions[0]
        
        # Calculate metrics
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = np.log(NUM_CLASSES)
        
        # OOD Detection
        is_valid = True
        warning = None
        
        if confidence < self.confidence_threshold:
            is_valid = False
            warning = f"Low confidence ({confidence*100:.1f}%)"
        
        if entropy > self.entropy_threshold:
            is_valid = False
            warning = f"High uncertainty (entropy={entropy:.2f})"
        
        if np.std(probs) < 0.1:
            is_valid = False
            warning = "Uniform probability - likely not a skin lesion"
        
        pred_class = self.class_names[pred_idx]
        is_malignant = pred_class in self.malignant_classes
        
        result = {
            'status': 'success' if is_valid else 'rejected',
            'prediction': pred_class if is_valid else 'NOT_SKIN_LESION',
            'predicted_class': pred_class,
            'diagnosis': self.class_full_names[pred_class],
            'confidence': confidence,
            'confidence_percent': confidence * 100,
            'entropy': entropy,
            'normalized_entropy': entropy / max_entropy,
            'is_valid': is_valid,
            'is_malignant': is_malignant if is_valid else None,
            'risk_level': 'HIGH' if is_malignant else 'LOW' if is_valid else 'UNKNOWN',
            'warning': warning,
            'recommendation': self._get_recommendation(pred_class, is_valid, is_malignant),
            'all_probabilities': {
                self.class_names[i]: float(probs[i]) * 100 
                for i in range(NUM_CLASSES)
            }
        }
        
        return result
    
    def _get_recommendation(self, pred_class, is_valid, is_malignant):
        """Generate appropriate recommendation based on prediction."""
        if not is_valid:
            return "Please upload a clear dermoscopy image of a skin lesion"
        
        if is_malignant:
            if pred_class == 'mel':
                return "⚠️ URGENT: Possible melanoma detected. Consult a dermatologist immediately!"
            elif pred_class == 'bcc':
                return "⚠️ Possible basal cell carcinoma. Schedule a dermatologist appointment soon."
            else:  # akiec
                return "⚠️ Possible pre-cancerous lesion. Consult a dermatologist for evaluation."
        else:
            return "Appears benign, but monitor for any changes in size, shape, or color."
    
    def predict_simple(self, image_array):
        """
        Simplified prediction returning just class and confidence.
        For backward compatibility with old code.
        """
        result = self.predict(image_array)
        return result['predicted_class'], result['confidence_percent']


def predict_single(image_path, model_path):
    """
    Predict a single image from file path.
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
        
    Returns:
        dict with full prediction results
    """
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    
    predictor = Predictor(model_path)
    return predictor.predict(img_array)
