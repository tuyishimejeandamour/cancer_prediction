import React, { useState } from 'react';
import API from '../api';

function Prediction() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await API.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPrediction(response.data);
    } catch (err) {
      setError('Failed to make prediction. Please ensure the API is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Make a Prediction</h2>
      <p style={{ color: '#666', marginBottom: '2rem' }}>
        Upload a skin lesion image to get an AI-powered prediction
      </p>

      <div style={{ textAlign: 'center' }}>
        <input
          type="file"
          id="file-upload"
          className="file-input"
          accept="image/*"
          onChange={handleFileSelect}
        />
        <label htmlFor="file-upload" className="file-label">
          Choose Image
        </label>

        {preview && (
          <div className="image-preview">
            <img src={preview} alt="Preview" />
          </div>
        )}

        {selectedFile && !loading && (
          <div style={{ marginTop: '1.5rem' }}>
            <button className="button-primary" onClick={handlePredict}>
              Analyze Image
            </button>
          </div>
        )}

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Analyzing image...</p>
          </div>
        )}

        {error && (
          <div className="alert alert-error">
            {error}
          </div>
        )}

        {prediction && (
          <div className={`result-card ${prediction.is_valid === false ? 'result-invalid' : prediction.is_malignant ? 'result-danger' : 'result-safe'}`}>
            <h3>Prediction Result</h3>
            
            {/* Check if image was rejected as non-skin */}
            {prediction.is_valid === false ? (
              <>
                <div className="confidence rejected">REJECTED</div>
                <p style={{ fontSize: '1.2rem', marginTop: '1rem', color: '#e74c3c' }}>
                  <strong>Not a valid skin lesion image</strong>
                </p>
                <p style={{ fontSize: '0.9rem', marginTop: '0.5rem', color: '#666' }}>
                  Confidence: {prediction.confidence_percent?.toFixed(1) || (prediction.confidence * 100).toFixed(1)}%
                  {prediction.warning && ` - ${prediction.warning}`}
                </p>
                <p style={{ fontSize: '0.9rem', marginTop: '1rem', opacity: 0.9 }}>
                  {prediction.recommendation || 'Please upload a clear dermoscopy image of a skin lesion'}
                </p>
              </>
            ) : (
              <>
                <div className="confidence">
                  {prediction.confidence_percent?.toFixed(1) || (prediction.confidence * 100).toFixed(1)}%
                </div>
                <p style={{ fontSize: '1.3rem', marginTop: '1rem' }}>
                  Classification: <strong>{prediction.prediction?.toUpperCase()}</strong>
                </p>
                {prediction.diagnosis && (
                  <p style={{ fontSize: '1rem', marginTop: '0.5rem', color: '#666' }}>
                    {prediction.diagnosis}
                  </p>
                )}
                <p style={{ 
                  fontSize: '1rem', 
                  marginTop: '1rem', 
                  padding: '0.75rem',
                  borderRadius: '8px',
                  backgroundColor: prediction.is_malignant ? '#fff5f5' : '#f0fff4',
                  color: prediction.is_malignant ? '#c53030' : '#276749'
                }}>
                  <strong>Risk Level: {prediction.risk_level || (prediction.is_malignant ? 'HIGH' : 'LOW')}</strong>
                </p>
                <p style={{ fontSize: '0.9rem', marginTop: '1rem', opacity: 0.9 }}>
                  {prediction.recommendation || (prediction.is_malignant 
                    ? 'This lesion shows characteristics of malignancy. Please consult a dermatologist immediately.'
                    : 'This lesion appears benign, but regular monitoring is recommended.')}
                </p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default Prediction;
