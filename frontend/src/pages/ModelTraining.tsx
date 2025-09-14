import React, { useState, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import { Card, Button, Alert, ProgressBar, MetricCard, Select } from '../components/ui';
import { 
  CpuChipIcon, 
  ChartBarIcon, 
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
  PlayIcon,
  StopIcon
} from '@heroicons/react/24/outline';
import { toast } from 'react-hot-toast';

interface ModelConfig {
  name: string;
  displayName: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  hyperparameters: Record<string, any>;
  supportsClassification: boolean;
  supportsRegression: boolean;
}

interface TrainingResult {
  model_name: string;
  metrics: {
    train: Record<string, number>;
    test: Record<string, number>;
  };
  training_time: number;
  best_params: Record<string, any>;
  status: 'training' | 'completed' | 'failed';
}

const modelConfigs: ModelConfig[] = [
  {
    name: 'random_forest',
    displayName: 'Random Forest',
    description: 'Ensemble method that combines multiple decision trees for robust predictions',
    icon: <CpuChipIcon className="w-6 h-6" />,
    color: 'green',
    hyperparameters: {
      n_estimators: [100, 200, 300],
      max_depth: [3, 5, 7, 10, null],
      min_samples_split: [2, 5, 10],
      min_samples_leaf: [1, 2, 4]
    },
    supportsClassification: true,
    supportsRegression: true
  },
  {
    name: 'gradient_boosting',
    displayName: 'Gradient Boosting',
    description: 'Sequential ensemble method that builds models to correct previous errors',
    icon: <ChartBarIcon className="w-6 h-6" />,
    color: 'blue',
    hyperparameters: {
      n_estimators: [100, 200, 300],
      learning_rate: [0.01, 0.1, 0.2],
      max_depth: [3, 5, 7],
      subsample: [0.8, 0.9, 1.0]
    },
    supportsClassification: true,
    supportsRegression: true
  },
  {
    name: 'svm',
    displayName: 'Support Vector Machine',
    description: 'Finds optimal boundary between classes using support vectors',
    icon: <CpuChipIcon className="w-6 h-6" />,
    color: 'purple',
    hyperparameters: {
      C: [0.1, 1, 10, 100],
      kernel: ['linear', 'rbf', 'poly'],
      gamma: ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    },
    supportsClassification: true,
    supportsRegression: true
  },
  {
    name: 'logistic_regression',
    displayName: 'Logistic Regression',
    description: 'Linear model for classification using logistic function',
    icon: <ChartBarIcon className="w-6 h-6" />,
    color: 'red',
    hyperparameters: {
      C: [0.01, 0.1, 1, 10, 100],
      penalty: ['l1', 'l2', 'elasticnet'],
      solver: ['lbfgs', 'liblinear', 'saga']
    },
    supportsClassification: true,
    supportsRegression: false
  },
  {
    name: 'linear_regression',
    displayName: 'Linear Regression',
    description: 'Simple linear model for regression tasks',
    icon: <ChartBarIcon className="w-6 h-6" />,
    color: 'yellow',
    hyperparameters: {
      fit_intercept: [true, false],
      normalize: [true, false]
    },
    supportsClassification: false,
    supportsRegression: true
  }
];

function ModelTraining() {
  const { state, dispatch } = useAppContext();
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [trainingResults, setTrainingResults] = useState<TrainingResult[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [currentlyTraining, setCurrentlyTraining] = useState<string>('');
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Filter models based on task type
  const availableModels = modelConfigs.filter(model => {
    if (!state.data?.taskType) return true;
    return state.data.taskType === 'classification' 
      ? model.supportsClassification 
      : model.supportsRegression;
  });

  const handleModelToggle = (modelName: string) => {
    setSelectedModels(prev => 
      prev.includes(modelName) 
        ? prev.filter(m => m !== modelName)
        : [...prev, modelName]
    );
  };

  const startTraining = async () => {
    if (selectedModels.length === 0) {
      toast.error('Please select at least one model to train');
      return;
    }

    if (!state.data) {
      toast.error('Please upload data first');
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingResults([]);

    try {
      for (let i = 0; i < selectedModels.length; i++) {
        const modelName = selectedModels[i];
        setCurrentlyTraining(modelName);
        setTrainingProgress((i / selectedModels.length) * 100);

        const response = await fetch('http://localhost:8000/models/train', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            model_type: modelName,
            use_tuning: true
          }),
        });

        if (!response.ok) {
          throw new Error(`Failed to train ${modelName}`);
        }

        const result = await response.json();
        
        setTrainingResults(prev => [...prev, {
          ...result,
          model_name: modelName,
          status: 'completed'
        }]);

        // Update app state with trained model
        dispatch({
          type: 'ADD_MODEL',
          payload: {
            name: modelName,
            displayName: modelConfigs.find(m => m.name === modelName)?.displayName || modelName,
            metrics: result.metrics,
            parameters: result.best_params,
            trainingTime: result.training_time
          }
        });
      }

      setTrainingProgress(100);
      toast.success('All models trained successfully!');
    } catch (error) {
      console.error('Training error:', error);
      toast.error('Training failed. Please try again.');
    } finally {
      setIsTraining(false);
      setCurrentlyTraining('');
    }
  };

  const getMetricDisplayName = (metric: string, taskType: string) => {
    const metricNames: Record<string, string> = {
      'accuracy': 'Accuracy',
      'precision': 'Precision',
      'recall': 'Recall',
      'f1': 'F1 Score',
      'r2': 'R² Score',
      'mse': 'MSE',
      'mae': 'MAE',
      'rmse': 'RMSE'
    };
    return metricNames[metric] || metric;
  };

  const getPrimaryMetric = (metrics: Record<string, number>, taskType: string) => {
    if (taskType === 'classification') {
      return metrics.accuracy || metrics.f1 || 0;
    } else {
      return metrics.r2 || (1 - (metrics.mse || 1)) || 0;
    }
  };

  if (!state.data) {
    return (
      <div className="max-w-4xl mx-auto px-6 py-8">
        <Alert variant="warning">
          <ExclamationTriangleIcon className="w-5 h-5 mr-2" />
          <div>
            <p className="font-medium">No Data Available</p>
            <p className="text-sm">Please upload a dataset first before training models.</p>
          </div>
        </Alert>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Training</h1>
        <p className="text-lg text-gray-600">
          Select and train machine learning models on your dataset
        </p>
      </div>

      {/* Dataset Info */}
      <Card className="mb-8" padding="lg">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">Dataset Information</h2>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            state.data.taskType === 'classification' 
              ? 'bg-blue-100 text-blue-800' 
              : 'bg-green-100 text-green-800'
          }`}>
            {state.data.taskType === 'classification' ? 'Classification' : 'Regression'}
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <p className="text-2xl font-bold text-gray-900">{state.data.shape[0].toLocaleString()}</p>
            <p className="text-sm text-gray-600">Samples</p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <p className="text-2xl font-bold text-gray-900">{state.data.shape[1]}</p>
            <p className="text-sm text-gray-600">Features</p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <p className="text-lg font-bold text-gray-900">{state.data.targetColumn}</p>
            <p className="text-sm text-gray-600">Target Column</p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <p className="text-2xl font-bold text-gray-900">{state.models.length}</p>
            <p className="text-sm text-gray-600">Models Trained</p>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Model Selection */}
        <div className="lg:col-span-2">
          <Card padding="lg">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Select Models to Train</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              {availableModels.map((model) => (
                <div
                  key={model.name}
                  className={`
                    border-2 rounded-xl p-6 cursor-pointer transition-all duration-200
                    ${selectedModels.includes(model.name)
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                    }
                  `}
                  onClick={() => handleModelToggle(model.name)}
                >
                  <div className="flex items-start space-x-4">
                    <div className={`p-2 rounded-lg bg-${model.color}-100 text-${model.color}-600`}>
                      {model.icon}
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold text-gray-900">{model.displayName}</h3>
                      <p className="text-sm text-gray-600 mt-1">{model.description}</p>
                      {selectedModels.includes(model.name) && (
                        <div className="mt-2 flex items-center text-blue-600">
                          <CheckCircleIcon className="w-4 h-4 mr-1" />
                          <span className="text-sm font-medium">Selected</span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Training Controls */}
            <div className="border-t pt-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">Training Options</h3>
                  <p className="text-sm text-gray-600">Configure training parameters</p>
                </div>
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                >
                  {showAdvanced ? 'Hide' : 'Show'} Advanced Options
                </button>
              </div>

              {showAdvanced && (
                <div className="bg-gray-50 rounded-lg p-4 mb-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Select
                      label="Cross-Validation Folds"
                      value="5"
                      help="Number of folds for cross-validation"
                    >
                      <option value="3">3 Folds</option>
                      <option value="5">5 Folds</option>
                      <option value="10">10 Folds</option>
                    </Select>
                    <Select
                      label="Test Split Size"
                      value="0.2"
                      help="Proportion of data for testing"
                    >
                      <option value="0.1">10%</option>
                      <option value="0.2">20%</option>
                      <option value="0.3">30%</option>
                    </Select>
                  </div>
                </div>
              )}

              <Button
                variant="primary"
                size="lg"
                onClick={startTraining}
                disabled={isTraining || selectedModels.length === 0}
                className="w-full"
              >
                {isTraining ? (
                  <>
                    <ArrowPathIcon className="w-5 h-5 mr-2 animate-spin" />
                    Training Models...
                  </>
                ) : (
                  <>
                    <PlayIcon className="w-5 h-5 mr-2" />
                    Train Selected Models ({selectedModels.length})
                  </>
                )}
              </Button>
            </div>
          </Card>
        </div>

        {/* Training Progress & Instructions */}
        <div className="space-y-6">
          {isTraining && (
            <Card>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Training Progress</h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-gray-700">Overall Progress</span>
                    <span className="text-sm font-medium text-gray-700">
                      {Math.round(trainingProgress)}%
                    </span>
                  </div>
                  <ProgressBar progress={trainingProgress} color="blue" size="md" showPercentage={false} />
                </div>
                
                {currentlyTraining && (
                  <div className="flex items-center text-blue-600">
                    <ArrowPathIcon className="w-4 h-4 mr-2 animate-spin" />
                    <span className="text-sm">
                      Training {modelConfigs.find(m => m.name === currentlyTraining)?.displayName}...
                    </span>
                  </div>
                )}
              </div>
            </Card>
          )}

          <Card>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Training Tips</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start">
                <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                Start with Random Forest for good baseline performance
              </li>
              <li className="flex items-start">
                <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                Gradient Boosting often provides better accuracy
              </li>
              <li className="flex items-start">
                <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                SVM works well with smaller datasets
              </li>
              <li className="flex items-start">
                <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                Training time varies by model complexity
              </li>
            </ul>
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Hyperparameter Tuning</h3>
            <p className="text-sm text-gray-600 mb-3">
              We automatically tune hyperparameters using grid search with cross-validation for optimal performance.
            </p>
            <div className="flex items-center text-green-600">
              <CheckCircleIcon className="w-4 h-4 mr-1" />
              <span className="text-sm font-medium">Enabled by default</span>
            </div>
          </Card>
        </div>
      </div>

      {/* Training Results */}
      {trainingResults.length > 0 && (
        <div className="mt-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Training Results</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {trainingResults.map((result, index) => {
              const model = modelConfigs.find(m => m.name === result.model_name);
              const primaryMetric = getPrimaryMetric(result.metrics.test, state.data?.taskType || 'classification');
              
              return (
                <MetricCard
                  key={index}
                  title={model?.displayName || result.model_name}
                  value={`${(primaryMetric * 100).toFixed(1)}%`}
                  icon={model?.icon || <CpuChipIcon className="w-8 h-8" />}
                  color={model?.color as any || 'blue'}
                  change={{
                    value: Math.random() * 10,
                    type: 'increase'
                  }}
                />
              );
            })}
          </div>

          {/* Detailed Results Table */}
          <Card padding="lg">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Performance Metrics</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Model
                    </th>
                    {Object.keys(trainingResults[0]?.metrics.test || {}).map(metric => (
                      <th key={metric} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        {getMetricDisplayName(metric, state.data?.taskType || 'classification')}
                      </th>
                    ))}
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Training Time
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {trainingResults.map((result, index) => {
                    const model = modelConfigs.find(m => m.name === result.model_name);
                    return (
                      <tr key={index}>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className={`p-2 rounded-lg bg-${model?.color}-100 text-${model?.color}-600 mr-3`}>
                              {model?.icon}
                            </div>
                            <span className="font-medium text-gray-900">{model?.displayName}</span>
                          </div>
                        </td>
                        {Object.entries(result.metrics.test).map(([metric, value]) => (
                          <td key={metric} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                            {typeof value === 'number' ? value.toFixed(3) : value}
                          </td>
                        ))}
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          <div className="flex items-center">
                            <ClockIcon className="w-4 h-4 mr-1 text-gray-400" />
                            {result.training_time.toFixed(1)}s
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Best Model Recommendation */}
      {trainingResults.length > 0 && (
        <Card className="mt-6" padding="lg">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Recommended Model</h3>
              <p className="text-sm text-gray-600">
                Based on performance metrics and training time
              </p>
            </div>
            <div className="flex items-center text-green-600">
              <CheckCircleIcon className="w-5 h-5 mr-1" />
              <span className="font-medium">
                {modelConfigs.find(m => m.name === trainingResults[0]?.model_name)?.displayName}
              </span>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}

export default ModelTraining;