import React, { useState } from 'react';
import './App.css';
import Prediction from './components/Prediction';
import DataManagement from './components/DataManagement';
import Monitoring from './components/Monitoring';

function App() {
  const [activeTab, setActiveTab] = useState('prediction');

  return (
    <div className="App">
      <header className="App-header">
        <div>
          <h1>dobby ditector</h1>
          <p>AI-Powered Melanoma Detection</p>
        </div>
      </header>
      
      <nav className="nav-tabs">
        <button 
          className={activeTab === 'prediction' ? 'active' : ''} 
          onClick={() => setActiveTab('prediction')}
        >
          Prediction
        </button>
        <button 
          className={activeTab === 'data' ? 'active' : ''} 
          onClick={() => setActiveTab('data')}
        >
          Data Management
        </button>
        <button 
          className={activeTab === 'monitoring' ? 'active' : ''} 
          onClick={() => setActiveTab('monitoring')}
        >
          Monitoring
        </button>
      </nav>

      <main className="main-content">
        {activeTab === 'prediction' && <Prediction />}
        {activeTab === 'data' && <DataManagement />}
        {activeTab === 'monitoring' && <Monitoring />}
      </main>

      <footer className="App-footer">
        <p>© 2025 Skin Cancer Detection System | African Leadership University</p>
      </footer>
    </div>
  );
}

export default App;
