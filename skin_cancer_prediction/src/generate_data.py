import os
import numpy as np
from PIL import Image
import random

def create_dummy_image(path, label):
    # Create a 64x64 image
    img_data = np.random.rand(64, 64, 3) * 255
    
    if label == 'malignant':
        # Add red tint
        img_data[:, :, 0] += 50
    else:
        # Add green tint
        img_data[:, :, 1] += 50
        
    img_data = np.clip(img_data, 0, 255).astype('uint8')
    img = Image.fromarray(img_data)
    img.save(path)

def generate_dataset(base_path, num_train=50, num_test=10):
    categories = ['benign', 'malignant']
    
    for category in categories:
        # Train data
        train_path = os.path.join(base_path, 'data', 'train', category)
        os.makedirs(train_path, exist_ok=True)
        for i in range(num_train):
            create_dummy_image(os.path.join(train_path, f'img_{i}.png'), category)
            
        # Test data
        test_path = os.path.join(base_path, 'data', 'test', category)
        os.makedirs(test_path, exist_ok=True)
        for i in range(num_test):
            create_dummy_image(os.path.join(test_path, f'img_{i}.png'), category)
            
    print(f"Generated {num_train} training images and {num_test} test images per class.")

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_dataset(base_path)
