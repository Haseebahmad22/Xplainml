import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  HomeIcon,
  DocumentArrowUpIcon,
  CogIcon,
  EyeIcon,
  CpuChipIcon,
  Bars3Icon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { useAppContext } from '../context/AppContext';
import { useTheme } from '../context/ThemeContext';

// NOTE: Routes must match those defined in App.tsx (<Route path="/upload" ...>, <Route path="/training" ...>)
// Previously these were '/data-upload' and '/model-training' causing navigation not to work.
const navigation = [
  { name: 'Home', href: '/', icon: HomeIcon },
  { name: 'Upload Data', href: '/upload', icon: DocumentArrowUpIcon },
  { name: 'Train Models', href: '/training', icon: CogIcon },
  { name: 'Explanations', href: '/explanations', icon: EyeIcon },
  { name: 'Predictions', href: '/predictions', icon: CpuChipIcon },
];

function Navbar() {
  const { state, dispatch } = useAppContext();
  const { mode, toggle } = useTheme();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  
  const isActive = (href: string) => {
    if (href === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(href);
  };

  const resetApp = () => {
    dispatch({ type: 'RESET_STATE' });
  };

  const getStepStatus = (step: string) => {
    switch (step) {
      case 'Upload Data':
        return state.data ? 'completed' : 'pending';
      case 'Train Models':
        return state.models.length > 0 ? 'completed' : 'pending';
      case 'Explanations':
        return state.explanations.length > 0 ? 'completed' : 'pending';
      case 'Predictions':
        return 'pending';
      default:
        return 'pending';
    }
  };

  return (
    <nav className="shadow-sm border-b border-gray-200 sticky top-0 z-50 backdrop-blur bg-white/80 dark:bg-gray-900/80 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Desktop Navigation */}
          <div className="flex">
            {/* Logo */}
            <div className="flex-shrink-0 flex items-center">
              <Link to="/" className="flex items-center space-x-3 group">
                <motion.div
                  whileHover={{ rotate: 3, scale: 1.05 }}
                  className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center shadow-lg"
                >
                  <span className="text-white font-bold text-lg tracking-tight">X</span>
                </motion.div>
                <div className="flex flex-col -space-y-1">
                  <span className="text-lg font-bold text-gray-900 dark:text-gray-100 leading-snug">XplainML</span>
                  <span className="text-[10px] font-medium text-blue-600 dark:text-blue-300 tracking-wider">INTERPRETABILITY</span>
                </div>
              </Link>
            </div>

            {/* Desktop Menu */}
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              {navigation.map((item) => {
                const Icon = item.icon;
                const active = isActive(item.href);
                const status = getStepStatus(item.name);
                
                return (
                  <motion.div whileHover={{ y: -2 }} key={item.name}>
                    <Link
                      to={item.href}
                      className={`relative inline-flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                        active
                          ? 'bg-blue-50 text-blue-700 shadow-sm dark:bg-blue-900/30 dark:text-blue-300'
                          : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
                      }`}
                    >
                      <Icon className="w-4 h-4 mr-2 opacity-70" />
                      <span>{item.name}</span>
                      {status === 'completed' && (
                        <span className="ml-2 inline-block w-2 h-2 bg-green-500 rounded-full shadow"></span>
                      )}
                      {active && (
                        <motion.span
                          layoutId="nav-active-pill"
                          className="absolute inset-0 rounded-lg -z-10 bg-gradient-to-r from-blue-500/10 to-purple-500/10"
                          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                        />
                      )}
                    </Link>
                  </motion.div>
                );
              })}
            </div>
          </div>

          {/* Progress & Reset */}
          <div className="hidden sm:flex sm:items-center sm:space-x-4">
            {/* Progress indicator */}
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <span>Progress:</span>
              <div className="flex space-x-1">
                {navigation.slice(1).map((item, index) => {
                  const status = getStepStatus(item.name);
                  return (
                    <div
                      key={item.name}
                      className={`w-3 h-3 rounded-full ${
                        status === 'completed' ? 'bg-green-500' : 'bg-gray-300'
                      }`}
                      title={item.name}
                    />
                  );
                })}
              </div>
            </div>

            {/* Reset button */}
            <div className="flex items-center space-x-2">
              <button
                onClick={toggle}
                className="text-xs px-3 py-2 rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
              >
                {mode === 'light' ? 'Dark' : 'Light'} Mode
              </button>
              <button
                onClick={resetApp}
                className="text-xs px-3 py-2 rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-600 dark:text-gray-300 hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors"
              >
                Reset
              </button>
            </div>
          </div>

          {/* Mobile menu button */}
          <div className="sm:hidden flex items-center">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100"
            >
              {mobileMenuOpen ? (
                <XMarkIcon className="block h-6 w-6" />
              ) : (
                <Bars3Icon className="block h-6 w-6" />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileMenuOpen && (
        <div className="sm:hidden">
          <div className="pt-2 pb-3 space-y-1">
            {navigation.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.href);
              const status = getStepStatus(item.name);
              
              return (
                <motion.div key={item.name} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}>
                  <Link
                    to={item.href}
                    className={`block px-4 py-3 rounded-lg text-sm font-medium ${
                      active
                        ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
                    }`}
                    onClick={() => setMobileMenuOpen(false)}
                  >
                    <div className="flex items-center">
                      <Icon className="w-5 h-5 mr-3 opacity-70" />
                      <span>{item.name}</span>
                      {status === 'completed' && (
                        <div className="w-2 h-2 bg-green-500 rounded-full ml-2"></div>
                      )}
                    </div>
                  </Link>
                </motion.div>
              );
            })}
            
            {/* Mobile progress & reset */}
            <div className="pl-3 pr-4 py-2 border-t border-gray-200 dark:border-gray-700 mt-2">
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">Progress:</span>
                <div className="flex space-x-1">
                  {navigation.slice(1).map((item) => {
                    const status = getStepStatus(item.name);
                    return (
                      <div
                        key={item.name}
                        className={`w-3 h-3 rounded-full ${
                          status === 'completed' ? 'bg-green-500' : 'bg-gray-300'
                        }`}
                      />
                    );
                  })}
                </div>
                <div className="flex items-center space-x-2 ml-3">
                  <button
                    onClick={toggle}
                    className="text-xs px-3 py-2 rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  >
                    {mode === 'light' ? 'Dark' : 'Light'}
                  </button>
                  <button
                    onClick={resetApp}
                    className="text-xs px-3 py-2 rounded-md border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-600 dark:text-gray-300 hover:bg-red-50 dark:hover:bg-red-900/30 transition-colors"
                  >
                    Reset
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}

export default Navbar;