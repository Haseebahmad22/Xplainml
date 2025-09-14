import React from 'react';
import { Link } from 'react-router-dom';
import { useAppContext } from '../context/AppContext';
import { 
  RocketLaunchIcon,
  ChartBarIcon,
  CpuChipIcon,
  EyeIcon,
  DocumentArrowUpIcon,
  CogIcon,
  SparklesIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';

const features = [
  {
    name: 'Smart Data Analysis',
    description: 'Upload CSV/Excel files and get instant insights about your data.',
    icon: DocumentArrowUpIcon,
    color: 'blue',
  },
  {
    name: 'Multiple ML Models',
    description: 'Train and compare different algorithms to find the best fit.',
    icon: CogIcon,
    color: 'purple',
  },
  {
    name: 'AI Explanations',
    description: 'Understand why your model makes decisions with SHAP and LIME.',
    icon: EyeIcon,
    color: 'green',
  },
  {
    name: 'Beautiful Visualizations',
    description: 'Interactive charts and dashboards for better insights.',
    icon: ChartBarIcon,
    color: 'yellow',
  },
  {
    name: 'Real-time Predictions',
    description: 'Make predictions on new data instantly with confidence scores.',
    icon: CpuChipIcon,
    color: 'red',
  },
  {
    name: 'Easy to Use',
    description: 'No coding required - intuitive interface for everyone.',
    icon: SparklesIcon,
    color: 'indigo',
  },
];

function Home() {
  const { state } = useAppContext();

  const steps = [
    {
      name: 'Upload Data',
      description: 'Get your data ready for analysis',
      status: state.data ? 'completed' : 'not_started',
    },
    {
      name: 'Train Models',
      description: 'Build machine learning models',
      status: state.models.length > 0 ? 'completed' : 'not_started',
    },
    {
      name: 'Generate Explanations',
      description: 'Understand model predictions',
      status: state.explanations.length > 0 ? 'completed' : 'not_started',
    },
    {
      name: 'Make Predictions',
      description: 'Use models for new predictions',
      status: 'not_started',
    },
  ];

  const getStepStatus = (status: string) => {
    return status === 'completed';
  };

  const currentStep = steps.findIndex(step => step.status !== 'completed') + 1;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 space-y-12">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-700 text-white">
        <div className="max-w-7xl mx-auto px-6 py-20 text-center">
          <div className="mb-6">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto">
              <RocketLaunchIcon className="w-8 h-8 text-white" />
            </div>
          </div>
          
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Welcome to{' '}
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              XplainML
            </span>
          </h1>
          
          <p className="text-xl text-blue-100 mb-8 max-w-3xl mx-auto">
            Transform your machine learning models into interpretable insights with our beautiful, 
            intuitive dashboard. Make AI explainable and accessible to everyone.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/data-upload"
              className="btn btn-primary inline-flex items-center space-x-2"
            >
              <DocumentArrowUpIcon className="w-5 h-5" />
              <span>Get Started</span>
            </Link>
            
            {currentStep > 1 && (
              <Link
                to={`/${['', 'data-upload', 'model-training', 'explanations', 'predictions'][currentStep]}`}
                className="btn btn-secondary inline-flex items-center space-x-2"
              >
                <span>Continue to Step {currentStep}</span>
              </Link>
            )}
          </div>
        </div>
      </div>

      {/* Progress Overview */}
      {currentStep > 1 && (
        <div className="max-w-7xl mx-auto px-6">
          <div className="card p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Your Progress</h2>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {steps.map((step, index) => {
                const isCompleted = getStepStatus(step.status);
                const isCurrent = currentStep === index + 1;
                
                return (
                  <div
                    key={step.name}
                    className={`p-4 rounded-lg border-2 transition-all duration-200 ${
                      isCompleted
                        ? 'border-green-200 bg-green-50'
                        : isCurrent
                        ? 'border-blue-200 bg-blue-50'
                        : 'border-gray-200 bg-gray-50'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                        isCompleted
                          ? 'bg-green-500'
                          : isCurrent
                          ? 'bg-blue-500'
                          : 'bg-gray-400'
                      }`}>
                        {isCompleted ? (
                          <CheckCircleIcon className="w-5 h-5 text-white" />
                        ) : (
                          <span className="text-white font-semibold text-sm">{index + 1}</span>
                        )}
                      </div>
                      <div className="flex-1">
                        <h3 className="font-semibold text-gray-900">{step.name}</h3>
                        <p className="text-sm text-gray-600 mt-1">{step.description}</p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Features Grid */}
      <div className="max-w-7xl mx-auto px-6">
        <h2 className="text-3xl font-bold text-gray-900 text-center mb-12">
          Powerful Features for Everyone
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature) => {
            const Icon = feature.icon;
            return (
              <div key={feature.name} className="card p-6 text-center hover-lift">
                <div className={`w-12 h-12 mx-auto mb-4 rounded-lg bg-${feature.color}-100 flex items-center justify-center`}>
                  <Icon className={`w-6 h-6 text-${feature.color}-600`} />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">{feature.name}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Quick Stats */}
      {(state.data || state.models.length > 0 || state.explanations.length > 0) && (
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {state.data && (
              <div className="metric-card blue p-6">
                <h3 className="text-lg font-semibold mb-2">Dataset</h3>
                <p className="text-2xl font-bold">{state.data.shape[0].toLocaleString()}</p>
                <p className="text-sm opacity-90">samples, {state.data.columns.length} features</p>
              </div>
            )}
            
            {state.models.length > 0 && (
              <div className="metric-card purple p-6">
                <h3 className="text-lg font-semibold mb-2">Models Trained</h3>
                <p className="text-2xl font-bold">{state.models.length}</p>
                <p className="text-sm opacity-90">machine learning models</p>
              </div>
            )}
            
            {state.explanations.length > 0 && (
              <div className="metric-card green p-6">
                <h3 className="text-lg font-semibold mb-2">Explanations</h3>
                <p className="text-2xl font-bold">{state.explanations.length}</p>
                <p className="text-sm opacity-90">explanation methods generated</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Call to Action */}
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center bg-gradient-to-r from-blue-50 to-purple-50 p-8 rounded-2xl">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Ready to Make Your AI Explainable?
          </h2>
          <p className="text-gray-600 mb-6">
            Join thousands of data scientists and ML engineers who trust XplainML for interpretable AI.
          </p>
          
          {!state.data ? (
            <Link to="/data-upload" className="btn btn-primary">
              Upload Your First Dataset
            </Link>
          ) : (
            <div className="space-x-4">
              <Link to="/model-training" className="btn btn-primary">
                Train New Model
              </Link>
              <Link to="/predictions" className="btn btn-secondary">
                Make Predictions
              </Link>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Home;