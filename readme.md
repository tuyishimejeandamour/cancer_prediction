# Skin Cancer Detection System

AI-powered skin lesion classification using deep learning on the HAM10000 dataset. Classifies skin lesions into 7 diagnostic categories with confidence scoring and risk assessment.

## Quick Links

| Resource              | Link                                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| **Live Demo**         | [https://cancerpredictionprivate.vercel.app/](https://cancerpredictionprivate.vercel.app/)           |
| **Video Demo**        | [YouTube Demo](https://youtu.be/UzNylPFbsOE)                                                         |
| **API Documentation** | [https://cancer-prediction-7rdt.onrender.com/docs](https://cancer-prediction-7rdt.onrender.com/docs) |

## What This Does

- **Classifies** skin lesion images into 7 medical categories
- **Detects** malignant vs benign lesions (melanoma, carcinoma, etc.)
- **Rejects** non-skin images (out-of-distribution detection)
- **Provides** confidence scores and medical recommendations
- **Runs** on CPU (no GPU required)

## Quick Start (3 Steps)

### 1. Train the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook notebook/skin_cancer_evaluation.ipynb
```

**Run all cells** to:

- Organize the HAM10000 dataset
- Train the MobileNetV2 model
- Evaluate performance
- Save model to `models/skin_cancer_7class_mobilenet.h5`

### 2. Start the Application

Using Docker (recommended):

```bash
docker compose up --build
```

Or manually:

```bash
# Terminal 1 - Backend API
pip install -r requirements.txt
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm install
npm start
```

### 3. Use the Application

- **Frontend UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## How the Data Works

### Dataset: HAM10000 (10,015 images)

The project uses the HAM10000 dataset with 7 diagnostic categories:

| Class     | Full Name                | Type          | Count |
| --------- | ------------------------ | ------------- | ----- |
| **mel**   | Melanoma                 | Malignant     | 1,113 |
| **bcc**   | Basal Cell Carcinoma     | Malignant     | 514   |
| **akiec** | Actinic Keratoses        | Pre-cancerous | 327   |
| **bkl**   | Benign Keratosis         | Benign        | 1,099 |
| **nv**    | Melanocytic Nevi (moles) | Benign        | 6,705 |
| **df**    | Dermatofibroma           | Benign        | 115   |
| **vasc**  | Vascular Lesions         | Benign        | 142   |

### Data Organization

The notebook automatically organizes images into:

```
data/
├── HAM10000_metadata.csv         # Original metadata
├── HAM10000_images_part_1/       # Source images
├── HAM10000_images_part_2/       # Source images
├── train_7class/                 # 80% for training
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
└── test_7class/                  # 20% for testing
    ├── akiec/
    ├── bcc/
    └── ...
```

### Data Processing Pipeline

1. **Organize**: `organize_ham10000_7class()` splits data into train/test (80/20)
2. **Load**: `load_data_7class()` creates TensorFlow datasets with caching
3. **Augment**: Random flips, rotations, zoom, and contrast adjustments
4. **Preprocess**: Resize to 96x96, normalize for MobileNetV2
5. **Balance**: Class weights handle the imbalanced dataset (nv: 6705 vs df: 115)

## Model Architecture

**MobileNetV2** - Lightweight CNN optimized for CPU inference

- **Input**: 96×96 RGB images
- **Backbone**: Pre-trained MobileNetV2 (ImageNet weights)
- **Head**: Custom classifier with dropout and regularization
- **Output**: 7-class softmax with confidence scores
- **Loss**: Focal Loss (handles class imbalance) + Label Smoothing
- **Training**: Two-phase (frozen backbone → fine-tuning)
- **Inference**: ~50-200ms per image on CPU

## API Endpoints

| Endpoint          | Method | Description                           |
| ----------------- | ------ | ------------------------------------- |
| `/`               | GET    | API info and version                  |
| `/predict`        | POST   | Full prediction with OOD detection    |
| `/predict_simple` | POST   | Quick prediction (class + confidence) |
| `/upload_data`    | POST   | Upload labeled training images        |
| `/classes`        | GET    | Get all diagnostic classes            |
| `/health`         | GET    | System health check                   |

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict image
curl -X POST "http://localhost:8000/predict" \
  -F "file=@lesion.jpg"

# Response
{
  "status": "success",
  "prediction": "mel",
  "diagnosis": "Melanoma (Skin Cancer)",
  "confidence_percent": 87.5,
  "is_malignant": true,
  "risk_level": "HIGH",
  "recommendation": "URGENT: Possible melanoma detected..."
}
```

## Project Structure

```
skin_cancer_prediction/
├── notebook/
│   └── skin_cancer_evaluation.ipynb    # Main training notebook
├── src/
│   ├── api.py                          # FastAPI backend
│   ├── preprocessing.py                # Data loading & processing
│   ├── prediction.py                   # Inference logic
│   ├── model.py                        # Model architecture
│   └── train.py                        # Training script
├── frontend/
│   └── src/
│       ├── App.js                      # React main component
│       └── components/
│           ├── Prediction.js           # Upload & predict UI
│           ├── DataManagement.js       # Training data upload
│           └── Monitoring.js           # Metrics dashboard
├── data/
│   ├── HAM10000_metadata.csv          # Original labels
│   ├── HAM10000_images_part_1/        # Source images (1)
│   ├── HAM10000_images_part_2/        # Source images (2)
│   ├── train_7class/                  # Training data (7 folders)
│   └── test_7class/                   # Test data (7 folders)
├── models/
│   └── skin_cancer_7class_mobilenet.h5  # Trained model
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Features

### 1. **Smart Prediction**

- Upload any image
- Get 7-class diagnosis
- See confidence scores for all classes
- Out-of-distribution rejection (detects non-skin images)
- Risk assessment and medical recommendations

### 2. **Data Management**

- Upload new labeled images
- Supports all 7 diagnostic classes
- Automatic organization into training folders
- Dataset statistics and distribution

### 3. **Monitoring Dashboard**

- Model performance metrics
- Class distribution visualizations
- Prediction history
- System health status

## Model Performance

| Metric          | Value            |
| --------------- | ---------------- |
| Test Accuracy   | ~75-85%          |
| Training Time   | ~45-60 min (CPU) |
| Inference Speed | ~50-200ms (CPU)  |
| Model Size      | ~15 MB           |
| Memory Usage    | ~500 MB RAM      |

### Per-Class Performance

The model performs best on:

- **nv** (melanocytic nevi) - 6,705 samples
- **mel** (melanoma) - 1,113 samples
- **bkl** (benign keratosis) - 1,099 samples

Lower performance on rare classes:

- **df** (dermatofibroma) - 115 samples
- **vasc** (vascular lesions) - 142 samples

## Docker Deployment

The project includes complete containerization:

```bash
# Build and start all services
docker-compose up --build

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

Services:

- **api**: FastAPI backend on port 8000
- **frontend**: React app on port 3000

## Configuration

### Key Settings

**Image Processing** (`src/preprocessing.py`):

```python
IMG_HEIGHT = 96
IMG_WIDTH = 96
BATCH_SIZE = 16
```

**OOD Detection** (`src/prediction.py`):

```python
confidence_threshold = 0.50  # 50% minimum
entropy_threshold = 1.5       # Max uncertainty
```

**Model Path** (`src/api.py`):

```python
MODEL_PATH = 'models/skin_cancer_7class_mobilenet.h5'
```

## Requirements

- **Python**: 3.9+
- **Node.js**: 18+
- **TensorFlow**: 2.13+
- **Docker** (optional): For containerized deployment

### Key Dependencies

**Python**:

- tensorflow
- fastapi
- uvicorn
- pillow
- numpy
- scikit-learn
- pandas
- matplotlib
- seaborn

**JavaScript**:

- react
- react-dom
- axios
- recharts

## License

Educational project for African Leadership University - Machine Learning Pipeline Course
