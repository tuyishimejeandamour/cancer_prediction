import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import API from '../api';

const CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'];
const MALIGNANT_CLASSES = ['mel', 'bcc', 'akiec'];

// Monochrome/neutral color palette
const COLORS = {
  akiec: '#1a1a2e',
  bcc: '#16213e',
  bkl: '#0f3460',
  df: '#533483',
  mel: '#2d3748',
  nv: '#4a5568',
  vasc: '#718096'
};

function Monitoring() {
  const [apiStatus, setApiStatus] = useState({ status: 'checking', modelLoaded: false });
  const [healthLoading, setHealthLoading] = useState(true);

  // Performance metrics
  const performanceData = [
    { metric: 'Accuracy', value: 76.19 },
    { metric: 'Precision', value: 71.2 },
    { metric: 'Recall', value: 68.5 },
    { metric: 'F1-Score', value: 69.8 }
  ];

  // Per-class performance
  const classPerformance = [
    { class: 'NV', precision: 87, recall: 92, f1: 89, count: 1341 },
    { class: 'MEL', precision: 55, recall: 43, f1: 48, count: 223 },
    { class: 'BKL', precision: 46, recall: 46, f1: 46, count: 220 },
    { class: 'BCC', precision: 69, recall: 45, f1: 54, count: 103 },
    { class: 'AKIEC', precision: 49, recall: 49, f1: 49, count: 65 },
    { class: 'DF', precision: 26, recall: 43, f1: 33, count: 23 },
    { class: 'VASC', precision: 58, recall: 68, f1: 62, count: 28 }
  ];

  // Dataset distribution
  const datasetDistribution = CLASS_NAMES.map(cls => ({
    name: cls.toUpperCase(),
    value: cls === 'nv' ? 5368 : cls === 'mel' ? 890 : cls === 'bkl' ? 879 : 
           cls === 'bcc' ? 411 : cls === 'akiec' ? 262 : cls === 'vasc' ? 114 : 92,
    color: COLORS[cls],
    isMalignant: MALIGNANT_CLASSES.includes(cls)
  }));

  // Training history
  const trainingHistory = [
    { epoch: 1, trainAcc: 62, valAcc: 71, trainLoss: 0.57, valLoss: 0.42 },
    { epoch: 5, trainAcc: 70, valAcc: 73, trainLoss: 0.20, valLoss: 0.16 },
    { epoch: 10, trainAcc: 70, valAcc: 72, trainLoss: 0.16, valLoss: 0.14 },
    { epoch: 15, trainAcc: 71, valAcc: 73, trainLoss: 0.15, valLoss: 0.13 },
    { epoch: 20, trainAcc: 76, valAcc: 74, trainLoss: 0.10, valLoss: 0.12 },
    { epoch: 25, trainAcc: 80, valAcc: 75, trainLoss: 0.08, valLoss: 0.11 },
    { epoch: 30, trainAcc: 84, valAcc: 76, trainLoss: 0.06, valLoss: 0.10 }
  ];

  useEffect(() => {
    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkApiHealth = async () => {
    setHealthLoading(true);
    try {
      const response = await API.get('/health');
      setApiStatus({
        status: 'online',
        modelLoaded: response.data.model_loaded,
        modelPath: response.data.model_path,
        modelExists: response.data.model_exists
      });
    } catch (err) {
      setApiStatus({
        status: 'offline',
        modelLoaded: false,
        error: err.message
      });
    } finally {
      setHealthLoading(false);
    }
  };

  const totalImages = datasetDistribution.reduce((sum, d) => sum + d.value, 0);
  const malignantCount = datasetDistribution
    .filter(d => d.isMalignant)
    .reduce((sum, d) => sum + d.value, 0);

  // Status indicator component
  const StatusDot = ({ active }) => (
    <span style={{
      display: 'inline-block',
      width: '8px',
      height: '8px',
      borderRadius: '50%',
      backgroundColor: active ? '#10b981' : '#ef4444',
      marginRight: '8px'
    }} />
  );

  return (
    <div>
      {/* System Status */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0 }}>System Status</h2>
          <button 
            onClick={checkApiHealth}
            disabled={healthLoading}
            style={{
              padding: '0.5rem 1rem',
              background: '#fff',
              border: '1px solid #e2e8f0',
              borderRadius: '6px',
              cursor: healthLoading ? 'wait' : 'pointer',
              fontSize: '0.875rem',
              color: '#4a5568'
            }}
          >
            {healthLoading ? 'Checking...' : 'Refresh'}
          </button>
        </div>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
          <div style={{ padding: '1.25rem', background: '#fafafa', borderRadius: '8px', borderLeft: '3px solid ' + (apiStatus.status === 'online' ? '#10b981' : '#ef4444') }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>API Status</div>
            <div style={{ fontSize: '1.25rem', fontWeight: '600', color: '#1a1a2e', display: 'flex', alignItems: 'center' }}>
              <StatusDot active={apiStatus.status === 'online'} />
              {apiStatus.status === 'online' ? 'Online' : apiStatus.status === 'checking' ? 'Checking' : 'Offline'}
            </div>
          </div>
          
          <div style={{ padding: '1.25rem', background: '#fafafa', borderRadius: '8px', borderLeft: '3px solid ' + (apiStatus.modelLoaded ? '#10b981' : '#ef4444') }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>Model Status</div>
            <div style={{ fontSize: '1.25rem', fontWeight: '600', color: '#1a1a2e', display: 'flex', alignItems: 'center' }}>
              <StatusDot active={apiStatus.modelLoaded} />
              {apiStatus.modelLoaded ? 'Loaded' : 'Not Loaded'}
            </div>
          </div>
          
          <div style={{ padding: '1.25rem', background: '#fafafa', borderRadius: '8px', borderLeft: '3px solid #1a1a2e' }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>Architecture</div>
            <div style={{ fontSize: '1.25rem', fontWeight: '600', color: '#1a1a2e' }}>MobileNetV2</div>
            <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.25rem' }}>7-class | 96x96 | CPU</div>
          </div>
          
          <div style={{ padding: '1.25rem', background: '#fafafa', borderRadius: '8px', borderLeft: '3px solid #1a1a2e' }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>Inference Time</div>
            <div style={{ fontSize: '1.25rem', fontWeight: '600', color: '#1a1a2e' }}>~150ms</div>
            <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.25rem' }}>Per prediction</div>
          </div>
        </div>
      </div>

      {/* Model Performance */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0 }}>Model Performance</h2>
          <span style={{ fontSize: '0.75rem', padding: '0.25rem 0.75rem', background: '#f1f5f9', borderRadius: '9999px', color: '#64748b' }}>Test Set</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '2rem' }}>
          {performanceData.map((item, idx) => (
            <div key={idx} style={{ textAlign: 'center', padding: '1.5rem', background: '#fafafa', borderRadius: '8px' }}>
              <div style={{ fontSize: '2rem', fontWeight: '700', color: '#1a1a2e' }}>{item.value.toFixed(1)}%</div>
              <div style={{ fontSize: '0.875rem', color: '#64748b', marginTop: '0.25rem' }}>{item.metric}</div>
            </div>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={performanceData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} stroke="#64748b" fontSize={12} />
            <YAxis type="category" dataKey="metric" width={80} stroke="#64748b" fontSize={12} />
            <Tooltip 
              formatter={(value) => `${value.toFixed(1)}%`}
              contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
            />
            <Bar dataKey="value" fill="#1a1a2e" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Per-Class Performance */}
      <div className="card">
        <h2>Class-wise Metrics</h2>
        <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
          Precision, Recall, and F1-Score for each diagnostic class
        </p>

        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={classPerformance}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="class" stroke="#64748b" fontSize={12} />
            <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} stroke="#64748b" fontSize={12} />
            <Tooltip 
              formatter={(value) => `${value}%`}
              contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)' }}
            />
            <Legend />
            <Bar dataKey="precision" name="Precision" fill="#1a1a2e" />
            <Bar dataKey="recall" name="Recall" fill="#4a5568" />
            <Bar dataKey="f1" name="F1-Score" fill="#94a3b8" />
          </BarChart>
        </ResponsiveContainer>

        <div style={{ marginTop: '1.5rem', padding: '1rem', background: '#f8fafc', borderRadius: '8px', fontSize: '0.875rem', color: '#64748b' }}>
          <strong style={{ color: '#1a1a2e' }}>Note:</strong> High-risk classes (MEL, BCC, AKIEC) show lower metrics due to class imbalance. Training uses weighted loss to improve minority class detection.
        </div>
      </div>

      {/* Training History */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0 }}>Training History</h2>
          <span style={{ fontSize: '0.75rem', padding: '0.25rem 0.75rem', background: '#f1f5f9', borderRadius: '9999px', color: '#64748b' }}>30 Epochs</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
          <div>
            <h4 style={{ textAlign: 'center', marginBottom: '1rem', color: '#1a1a2e', fontSize: '0.875rem', fontWeight: '600' }}>Accuracy</h4>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={trainingHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} />
                <YAxis domain={[50, 90]} tickFormatter={(v) => `${v}%`} stroke="#64748b" fontSize={12} />
                <Tooltip 
                  formatter={(value) => `${value}%`}
                  contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }}
                />
                <Legend />
                <Line type="monotone" dataKey="trainAcc" name="Train" stroke="#1a1a2e" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="valAcc" name="Val" stroke="#94a3b8" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div>
            <h4 style={{ textAlign: 'center', marginBottom: '1rem', color: '#1a1a2e', fontSize: '0.875rem', fontWeight: '600' }}>Loss</h4>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={trainingHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="epoch" stroke="#64748b" fontSize={12} />
                <YAxis domain={[0, 0.6]} stroke="#64748b" fontSize={12} />
                <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }} />
                <Legend />
                <Line type="monotone" dataKey="trainLoss" name="Train" stroke="#1a1a2e" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="valLoss" name="Val" stroke="#94a3b8" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Dataset Distribution */}
      <div className="card">
        <h2>Dataset Distribution</h2>
        <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
          Training data balance across 7 diagnostic classes
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', alignItems: 'center' }}>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={datasetDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              >
                {datasetDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value, name) => [`${value} images`, name]}
                contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }}
              />
            </PieChart>
          </ResponsiveContainer>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {datasetDistribution.map(item => (
              <div 
                key={item.name}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.75rem 1rem',
                  background: '#fafafa',
                  borderRadius: '6px',
                  borderLeft: `3px solid ${item.color}`
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <span style={{ fontWeight: '600', color: '#1a1a2e' }}>{item.name}</span>
                  {item.isMalignant && (
                    <span style={{ 
                      fontSize: '0.625rem', 
                      padding: '0.125rem 0.5rem', 
                      background: '#fee2e2', 
                      color: '#991b1b',
                      borderRadius: '9999px',
                      fontWeight: '500',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em'
                    }}>
                      High Risk
                    </span>
                  )}
                </div>
                <span style={{ fontWeight: '700', color: '#1a1a2e' }}>{item.value.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>

        <div style={{ marginTop: '2rem', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
          <div style={{ textAlign: 'center', padding: '1.25rem', background: '#fafafa', borderRadius: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>Total Images</div>
            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#1a1a2e' }}>{totalImages.toLocaleString()}</div>
          </div>
          <div style={{ textAlign: 'center', padding: '1.25rem', background: '#fafafa', borderRadius: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>Benign</div>
            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#1a1a2e' }}>{(totalImages - malignantCount).toLocaleString()}</div>
          </div>
          <div style={{ textAlign: 'center', padding: '1.25rem', background: '#fafafa', borderRadius: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>Malignant</div>
            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#1a1a2e' }}>{malignantCount.toLocaleString()}</div>
          </div>
          <div style={{ textAlign: 'center', padding: '1.25rem', background: '#fafafa', borderRadius: '8px' }}>
            <div style={{ fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.5rem' }}>Imbalance Ratio</div>
            <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#1a1a2e' }}>58:1</div>
          </div>
        </div>
      </div>

      {/* OOD Detection */}
      <div className="card">
        <h2>Out-of-Distribution Detection</h2>
        <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
          Non-skin images are rejected using entropy and confidence thresholds
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
          <div style={{ textAlign: 'center', padding: '1.5rem', background: '#fafafa', borderRadius: '8px' }}>
            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#1a1a2e' }}>60%</div>
            <div style={{ fontSize: '0.875rem', color: '#64748b', marginTop: '0.25rem' }}>Confidence Threshold</div>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.25rem' }}>Minimum required</div>
          </div>
          <div style={{ textAlign: 'center', padding: '1.5rem', background: '#fafafa', borderRadius: '8px' }}>
            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#1a1a2e' }}>1.3</div>
            <div style={{ fontSize: '0.875rem', color: '#64748b', marginTop: '0.25rem' }}>Entropy Threshold</div>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.25rem' }}>Maximum allowed</div>
          </div>
          <div style={{ textAlign: 'center', padding: '1.5rem', background: '#fafafa', borderRadius: '8px' }}>
            <div style={{ fontSize: '2rem', fontWeight: '700', color: '#1a1a2e' }}>1.95</div>
            <div style={{ fontSize: '0.875rem', color: '#64748b', marginTop: '0.25rem' }}>Max Entropy</div>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.25rem' }}>log(7 classes)</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Monitoring;
