import streamlit as st
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_page_config(page_title="Skin Cancer Prediction Dashboard", layout="wide")

API_URL = "http://localhost:8000"

st.title("Skin Cancer Prediction Dashboard")

# Sidebar for navigation
page = st.sidebar.selectbox("Navigate", ["Prediction", "Data Management", "Model Monitoring"])

if page == "Prediction":
    st.header("Make a Prediction")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Predict"):
            with st.spinner('Predicting...'):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Prediction: {result['prediction']}")
                        st.info(f"Confidence: {result['confidence']:.2f}%")
                    else:
                        st.error("Error making prediction")
                except Exception as e:
                    st.error(f"Connection error: {e}. Is the API running?")

elif page == "Data Management":
    st.header("Data Management")
    
    st.subheader("Upload New Training Data")
    upload_label = st.selectbox("Label", ["benign", "malignant"])
    new_data = st.file_uploader("Upload Image for Training", type=["jpg", "png", "jpeg"])
    
    if st.button("Upload and Save"):
        if new_data is not None:
            try:
                files = {"file": new_data.getvalue()}
                params = {"label": upload_label}
                response = requests.post(f"{API_URL}/upload_data", files=files, params=params)
                
                if response.status_code == 200:
                    st.success("Data uploaded successfully!")
                else:
                    st.error("Failed to upload data")
            except Exception as e:
                st.error(f"Connection error: {e}")
    
    st.subheader("Retrain Model")
    if st.button("Trigger Retraining"):
        try:
            response = requests.post(f"{API_URL}/retrain")
            if response.status_code == 200:
                st.success("Retraining started in the background!")
            else:
                st.error("Failed to start retraining")
        except Exception as e:
            st.error(f"Connection error: {e}")

elif page == "Model Monitoring":
    st.header("Model Monitoring & Visualizations")
    
    # Visualize Dataset Distribution
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, 'data', 'train')
    
    if os.path.exists(data_dir):
        classes = ['benign', 'malignant']
        counts = []
        for c in classes:
            path = os.path.join(data_dir, c)
            if os.path.exists(path):
                counts.append(len(os.listdir(path)))
            else:
                counts.append(0)
        
        df = pd.DataFrame({'Class': classes, 'Count': counts})
        
        st.subheader("Training Data Distribution")
        fig, ax = plt.subplots()
        sns.barplot(x='Class', y='Count', data=df, ax=ax)
        st.pyplot(fig)
        
        st.metric("Total Training Images", sum(counts))
    else:
        st.warning("Data directory not found.")

    # Simulated Model Metrics (In a real app, fetch from a database or log file)
    st.subheader("Model Performance History")
    # Dummy data for visualization
    history_df = pd.DataFrame({
        'Epoch': [1, 2, 3, 4, 5],
        'Accuracy': [0.6, 0.7, 0.75, 0.8, 0.85],
        'Loss': [0.8, 0.6, 0.5, 0.4, 0.3]
    })
    
    st.line_chart(history_df.set_index('Epoch'))
