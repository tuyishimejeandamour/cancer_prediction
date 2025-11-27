from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import sys
import numpy as np
from PIL import Image
import io
import tensorflow as tf

# Add the src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prediction import Predictor, predict_single, FocalLoss
from preprocessing import CLASS_NAMES, CLASS_FULL_NAMES, MALIGNANT_CLASSES

app = FastAPI(
    title="Skin Cancer Prediction API",
    description="7-Class Skin Lesion Classification with Out-of-Distribution Detection",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Try new 7-class model first, fall back to old binary model
MODEL_PATH_7CLASS = os.path.join(BASE_PATH, 'models', 'skin_cancer_7class_mobilenet.h5')
MODEL_PATH_BINARY = os.path.join(BASE_PATH, 'models', 'skin_cancer_model.h5')

# Use 7-class model if available
if os.path.exists(MODEL_PATH_7CLASS):
    MODEL_PATH = MODEL_PATH_7CLASS
    MODEL_TYPE = "7-class"
else:
    MODEL_PATH = MODEL_PATH_BINARY
    MODEL_TYPE = "binary"

# Initialize predictor
predictor = None


def get_predictor():
    """Lazy load the predictor with enhanced error handling."""
    global predictor
    if predictor is None and os.path.exists(MODEL_PATH):
        try:
            predictor = Predictor(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Error loading model: {error_msg}")
            
            # Check if it's the TrueDivide layer issue
            if "TrueDivide" in error_msg or "Unknown layer" in error_msg:
                print("\n" + "="*60)
                print("🔧 MODEL COMPATIBILITY ISSUE DETECTED")
                print("="*60)
                print("\nThe model was trained with a different TensorFlow version.")
                print("To fix this, please retrain the model:\n")
                print("1. Open: notebook/skin_cancer_evaluation.ipynb")
                print("2. Run all cells (this will take ~45-60 minutes)")
                print("3. Restart this API server\n")
                print("="*60 + "\n")
            
            predictor = None
    return predictor


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Skin Cancer Prediction API",
        "version": "2.0.0",
        "model_type": MODEL_TYPE,
        "classes": CLASS_NAMES if MODEL_TYPE == "7-class" else ["benign", "malignant"],
        "features": [
            "7-class classification",
            "Out-of-Distribution detection",
            "Confidence calibration",
            "Risk assessment"
        ]
    }


@app.get("/classes")
def get_classes():
    """Return information about the diagnostic classes."""
    return {
        "num_classes": len(CLASS_NAMES),
        "classes": [
            {
                "code": cls,
                "name": CLASS_FULL_NAMES[cls],
                "is_malignant": cls in MALIGNANT_CLASSES
            }
            for cls in CLASS_NAMES
        ]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict skin lesion class from uploaded image.
    
    Returns:
        - prediction: Predicted class code
        - diagnosis: Full diagnosis name
        - confidence: Prediction confidence (0-100%)
        - is_valid: Whether the image appears to be a valid skin lesion
        - is_malignant: Whether the prediction indicates malignancy
        - risk_level: HIGH, LOW, or UNKNOWN
        - recommendation: Medical recommendation
        - all_probabilities: Probabilities for all classes
    """
    try:
        pred = get_predictor()
        if pred is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Model not available or incompatible",
                    "message": "Please retrain the model using the Jupyter notebook",
                    "solution": "Run all cells in: notebook/skin_cancer_evaluation.ipynb"
                }
            )
        
        contents = await file.read()
        
        # Save temporarily to process
        temp_file = "temp_image.jpg"
        with open(temp_file, "wb") as f:
            f.write(contents)
        
        try:
            result = predict_single(temp_file, MODEL_PATH)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Add filename to result
        result['filename'] = file.filename
        
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={
                "error": str(e),
                "type": type(e).__name__
            }
        )


@app.post("/predict_simple")
async def predict_simple(file: UploadFile = File(...)):
    """
    Simplified prediction endpoint for backward compatibility.
    Returns just the class and confidence.
    """
    try:
        pred = get_predictor()
        if pred is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Model not available"}
            )
        
        contents = await file.read()
        temp_file = "temp_image.jpg"
        with open(temp_file, "wb") as f:
            f.write(contents)
        
        try:
            result = predict_single(temp_file, MODEL_PATH)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return {
            "filename": file.filename,
            "prediction": result['predicted_class'],
            "confidence": result['confidence_percent']
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...), label: str = "nv"):
    """
    Upload training data with 7-class labels.
    
    Valid labels: akiec, bcc, bkl, df, mel, nv, vasc
    """
    if label not in CLASS_NAMES:
        return JSONResponse(
            status_code=400, 
            content={
                "error": f"Invalid label '{label}'. Must be one of: {CLASS_NAMES}"
            }
        )
    
    try:
        # Save to 7-class training data
        save_dir = os.path.join(BASE_PATH, 'data', 'train_7class', label)
        os.makedirs(save_dir, exist_ok=True)
        
        file_path = os.path.join(save_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "message": f"File saved to {label} training data",
            "class_name": CLASS_FULL_NAMES[label],
            "is_malignant": label in MALIGNANT_CLASSES
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    }


@app.get("/data_stats")
def get_data_stats():
    """Get statistics about the training data."""
    train_dir = os.path.join(BASE_PATH, 'data', 'train_7class')
    
    if not os.path.exists(train_dir):
        return {
            "total": 0,
            "classes": {cls: 0 for cls in CLASS_NAMES}
        }
    
    class_counts = {}
    total = 0
    
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(train_dir, cls)
        if os.path.exists(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[cls] = count
            total += count
        else:
            class_counts[cls] = 0
    
    return {
        "total": total,
        "classes": class_counts
    }


@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Trigger model retraining.
    Note: In production, this would start actual training.
    Currently returns a simulation response.
    """
    return {
        "status": "started",
        "message": "Model retraining initiated. This is a simulation - actual training requires running the notebook.",
        "estimated_time": "20-30 minutes"
    }


@app.get("/training_status")
def get_training_status():
    """Get the status of ongoing training."""
    # In production, this would check actual training status
    return {
        "status": "idle",
        "message": "No training in progress"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
