import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
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

const navigation = [
  { name: 'Home', href: '/', icon: HomeIcon },
  { name: 'Upload Data', href: '/data-upload', icon: DocumentArrowUpIcon },
  { name: 'Train Models', href: '/model-training', icon: CogIcon },
  { name: 'Explanations', href: '/explanations', icon: EyeIcon },
  { name: 'Predictions', href: '/predictions', icon: CpuChipIcon },
];

function Navbar() {
  const { state, dispatch } = useAppContext();
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
    <nav className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Desktop Navigation */}
          <div className="flex">
            {/* Logo */}
            <div className="flex-shrink-0 flex items-center">
              <Link to="/" className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">X</span>
                </div>
                <span className="text-xl font-bold text-gray-900">XplainML</span>
              </Link>
            </div>

            {/* Desktop Menu */}
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              {navigation.map((item) => {
                const Icon = item.icon;
                const active = isActive(item.href);
                const status = getStepStatus(item.name);
                
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200 ${
                      active
                        ? 'border-blue-500 text-gray-900'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <Icon className="w-4 h-4 mr-2" />
                    {item.name}
                    {status === 'completed' && (
                      <div className="w-2 h-2 bg-green-500 rounded-full ml-2"></div>
                    )}
                  </Link>
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
            <button
              onClick={resetApp}
              className="text-sm text-gray-500 hover:text-gray-700 px-3 py-1 rounded-md border border-gray-300 hover:border-gray-400 transition-colors duration-200"
            >
              Reset
            </button>
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
                <Link
                  key={item.name}
                  to={item.href}
                  className={`block pl-3 pr-4 py-2 border-l-4 text-base font-medium ${
                    active
                      ? 'bg-blue-50 border-blue-500 text-blue-700'
                      : 'border-transparent text-gray-600 hover:text-gray-800 hover:bg-gray-50 hover:border-gray-300'
                  }`}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <div className="flex items-center">
                    <Icon className="w-5 h-5 mr-3" />
                    {item.name}
                    {status === 'completed' && (
                      <div className="w-2 h-2 bg-green-500 rounded-full ml-2"></div>
                    )}
                  </div>
                </Link>
              );
            })}
            
            {/* Mobile progress & reset */}
            <div className="pl-3 pr-4 py-2 border-t border-gray-200 mt-2">
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
                <button
                  onClick={resetApp}
                  className="ml-4 text-sm text-gray-500 hover:text-gray-700 px-2 py-1 rounded border border-gray-300"
                >
                  Reset
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
}

export default Navbar;