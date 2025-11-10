import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
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
import { ThemeProvider } from './context/ThemeContext';

function PageTransitions() {
  const location = useLocation();
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route
          path="/"
          element={
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
              <Home />
            </motion.div>
          }
        />
        <Route
          path="/upload"
          element={
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
              <DataUpload />
            </motion.div>
          }
        />
        <Route
          path="/training"
          element={
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
              <ModelTraining />
            </motion.div>
          }
        />
        <Route
          path="/explanations"
          element={
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
              <Explanations />
            </motion.div>
          }
        />
        <Route
          path="/visualizations"
          element={
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
              <Visualizations />
            </motion.div>
          }
        />
        <Route
          path="/predictions"
          element={
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.25 }}>
              <Predictions />
            </motion.div>
          }
        />
      </Routes>
    </AnimatePresence>
  );
}

function App() {
  return (
    <AppProvider>
      <ThemeProvider>
        <Router>
          <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
            <Navbar />
            <main className="container mx-auto px-4 py-8">
              <PageTransitions />
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
      </ThemeProvider>
    </AppProvider>
  );
}

export default App;
