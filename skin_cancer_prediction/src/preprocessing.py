import tensorflow as tf
import numpy as np
import os

# Updated for 7-class MobileNetV2 model (CPU-optimized)
IMG_HEIGHT = 96
IMG_WIDTH = 96
BATCH_SIZE = 16

# 7-class configuration
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_FULL_NAMES = {
    'akiec': 'Actinic Keratoses (Pre-cancerous)',
    'bcc': 'Basal Cell Carcinoma (Skin Cancer)',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma (Benign)',
    'mel': 'Melanoma (Skin Cancer)',
    'nv': 'Melanocytic Nevi (Mole - Benign)',
    'vasc': 'Vascular Lesion (Benign)'
}
MALIGNANT_CLASSES = ['mel', 'bcc', 'akiec']
NUM_CLASSES = 7


def load_data(data_dir, use_7class=True):
    """
    Loads data from directory and returns tf.data.Dataset objects
    
    Args:
        data_dir: Base data directory
        use_7class: If True, uses 7-class directories, else binary
    """
    if use_7class:
        train_dir = os.path.join(data_dir, 'train_7class')
        test_dir = os.path.join(data_dir, 'test_7class')
    else:
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)
        
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    # Configure for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds


def preprocess_image(image):
    """
    Preprocesses a single image for prediction
    """
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.expand_dims(image, 0)  # Create a batch
    return image
