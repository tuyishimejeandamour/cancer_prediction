import os
import tensorflow as tf
from preprocessing import load_data
from model import create_model

def train_model(base_path):
    data_dir = os.path.join(base_path, 'data')
    models_dir = os.path.join(base_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print("Loading data...")
    train_ds, val_ds, test_ds = load_data(data_dir)
    
    print("Creating model...")
    model = create_model()
    
    print("Training model...")
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    
    print("Evaluating model...")
    loss, acc = model.evaluate(test_ds)
    print(f"Test accuracy: {acc}")
    
    model_path = os.path.join(models_dir, 'skin_cancer_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return history, acc

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_model(base_path)
