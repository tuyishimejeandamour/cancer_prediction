AFRICAN LEADERSHIP UNIVERSITY
[BSE]
[MACHINE LEARNING PIPELINE

[SUMMATIVE ASSIGNMENT]
========== OUR MODEL IS ABOUT PREDICTING SKIN CANCER ==========

Machine Learning Cycle

Objective:
You will demonstrate the end-to-end Machine Learning process.

Overview
As a Machine Learning engineer, you have been tasked with creating an ML Pipeline and scaling and monitoring it on a cloud platform of your choice. This project focuses on Skin Cancer prediction using the HAM10000 dataset (or synthetic data for testing).

Tasks
Creating a Machine Learning Classification model offline and deploying it.

Evaluate the model(s) using all the metrics required on a Jupyter Notebook and demonstrate how good the model(s) are.

The Breakdown:

Create the following processes:
Data acquisition (HAM10000 or Synthetic)
Data processing
Model Creation
Model testing
Model Retraining
API creation with Python.

## Setup Instructions

1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Data Setup:**

   - **Option A: Use Synthetic Data (Fastest)**
     Run the generation script to create dummy images for testing the pipeline.
     ```bash
     python src/generate_data.py
     ```
   - **Option B: Use HAM10000 Dataset (Real Data)**
     1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
     2. Extract the files.
     3. Update the paths in `src/organize_data.py` to point to your downloaded files.
     4. Run `python src/organize_data.py` to sort images into `data/train` and `data/test` folders.

4. **Train the Model:**

   ```bash
   python src/train.py
   ```

5. **Run the API:**

   ```bash
   uvicorn src.api:app --reload
   ```

   The API will be available at `http://localhost:8000`.

6. **Run the Dashboard (UI):**

   ```bash
   streamlit run src/app.py
   ```

   The dashboard will be available at `http://localhost:8501`.

7. **Load Testing (Locust):**
   Make sure the API is running, then:

   ```bash
   locust -f locustfile.py
   ```

   Open `http://localhost:8089` to start the simulation.

8. **Docker Deployment (Recommended):**
   ```bash
   docker-compose up --build
   ```
   This will start:
   - Backend API at `http://localhost:8000`
   - Frontend UI at `http://localhost:3000`

## Project Structure

- `src/`: Source code for preprocessing, model, training, API, and UI.
- `data/`: Directory for dataset (train/test).
- `models/`: Saved model files.
- `notebook/`: Jupyter notebooks for experimentation.
- `locustfile.py`: Load testing script.
- `Dockerfile`: Container configuration.

## Features

- **Prediction:** Upload an image to classify as Benign or Malignant.
- **Retraining:** Trigger model retraining from the UI.
- **Data Management:** Upload new training data via the UI.
- **Monitoring:** View dataset distribution and model metrics.

Create a UI to cover the following
Model up-time
Data Visualizations
Access to train and retrain functionalities

Deploy the processes in task 1 on a cloud platform. Demonstrate the evaluation process of the model in production.

Simulate a flood of requests (using software Locust - Click here ) send them to the model, and show how the model responds to these requests. Record and show the latency and response time of the requests with different numbers of Docker containers.

Demonstrate how a user uploads values/features and the model predicts

A User should be able to upload new data and trigger retraining.
The final solution MUST have the following functionalities:

Model prediction - Allow a user to predict one datapoint from an image or sound
Visualizations - Create visualizations that make sense of different features in your dataset. Create interpretations of at least 3 features in your dataset. What story does it tell?
Upload Data - Bulk data that will be used to retrain (multiple images, multiple .wav files for sound)
Trigger retraining based on the uploaded data - Have a feature where one can press a button that can trigger a retraining process
Github Repo Directory Structure
Project_name/
│
├── README.md
│
├── notebook/
│ ├──project_name.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── model.py
│ └── prediction.py
│
├── data/
│ ├──train/
│ └── test/
└── models/
├── \_model_name.pkl or \_model_name.tf

Requirements
A link to the GitHub repo.
The README.md should have clear instructions on:
A video Demo - YouTube Link
URL where applicable
The project description
And clear steps on how to set it up
Results from Flood Request Simulation
Notebook
Should contain detailed preprocessing steps
Model Training
Model Test / Prediction Functions
The model file
Pickle (.pkl), tensorflow (.tf) or .h5 file
Submission Instructions
You will have two attempts during submission Make sure to submit the following in each attempt respectively

The first attempt will be a Zip File of the GitHub Code Repository
The second attempt will be a GitHub Repository URL

what is needed: =========CRETERIAL TO FOLLOW====================

1. Retraining Process The script + Model file for retraining MUST be present to evaluate this criteria

Clear Demonstration of the following triggers 1. Data file Uploading + Saving to Database (for purposes of retraining) 2. Data Preprocessing of the uploaded data 3. Retraining - The student uses a custom model created as a pre-trained model

2. Prediction Process The script + model file for prediction MUST be present to evaluate this criterion

Clear Demonstration of the following: 1. Inserting a data point for prediction (Could be an image or inputs of a CSV row, or an audio file) 2. Displays the CORRECT prediction based on the label/class of the data point

3. Evaluation of Models The notebook used to create the model MUST be present to evaluate this criteria

Clear Preprocessing steps are present with clear use of optimization techniques (Regularization, Optimizers, early stopping,use of a pretrained model or hyper parameter tuning) and Uses At least 4 Evaluation metrics used (Accuracy, loss, F1 score, Precision, Recall e.t.c)

4. Deployment Package

Showcases a UI using mobile app or Web app (Dockerized or public URL ) Contains some data insights based on the dataset
