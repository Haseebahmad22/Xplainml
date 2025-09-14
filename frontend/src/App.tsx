import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import './index.css';
import './styles/components.css';

// Components
import Navbar from './components/Navbar';
import Home from './pages/Home';
import DataUpload from './pages/DataUpload';
import ModelTraining from './pages/ModelTraining';
import Explanations from './pages/Explanations';
import Visualizations from './pages/Visualizations';
import Predictions from './pages/Predictions';

// Context
import { AppProvider } from './context/AppContext';

function App() {
  return (
    <AppProvider>
      <Router>
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
          <Navbar />
          
          <main className="container mx-auto px-4 py-8">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/upload" element={<DataUpload />} />
              <Route path="/training" element={<ModelTraining />} />
              <Route path="/explanations" element={<Explanations />} />
              <Route path="/visualizations" element={<Visualizations />} />
              <Route path="/predictions" element={<Predictions />} />
            </Routes>
          </main>
          
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#363636',
                color: '#fff',
              },
              success: {
                duration: 3000,
                iconTheme: {
                  primary: '#4ade80',
                  secondary: '#fff',
                },
              },
              error: {
                duration: 5000,
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#fff',
                },
              },
            }}
          />
        </div>
      </Router>
    </AppProvider>
  );
}

export default App;
