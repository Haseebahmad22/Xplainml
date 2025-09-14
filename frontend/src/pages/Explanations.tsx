import React, { useState, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import { Card, Button, Alert, Select, LoadingSpinner } from '../components/ui';
import { 
  EyeIcon, 
  ChartBarIcon, 
  CpuChipIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  ArrowDownIcon,
  ArrowUpIcon,
  GlobeAltIcon,
  UserIcon
} from '@heroicons/react/24/outline';
import { toast } from 'react-hot-toast';

interface ExplanationResult {
  method: string;
  type: 'global' | 'local';
  model_name: string;
  data: any;
  feature_importance?: Record<string, number>;
  sample_explanations?: Array<{
    sample_index: number;
    feature_contributions: Record<string, number>;
    prediction: number;
    actual?: number;
  }>;
}

interface ChartData {
  labels: string[];
  values: number[];
  colors: string[];
}

function Explanations() {
  const { state, dispatch } = useAppContext();
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [selectedMethod, setSelectedMethod] = useState<string>('shap');
  const [explanations, setExplanations] = useState<ExplanationResult[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedSample, setSelectedSample] = useState<number>(0);

  const explanationMethods = [
    {
      value: 'shap',
      label: 'SHAP (SHapley Additive exPlanations)',
      description: 'Unified framework for interpreting predictions using game theory',
      supportsGlobal: true,
      supportsLocal: true
    },
    {
      value: 'lime',
      label: 'LIME (Local Interpretable Model-Agnostic Explanations)',
      description: 'Explains individual predictions by learning local interpretable models',
      supportsGlobal: false,
      supportsLocal: true
    },
    {
      value: 'permutation',
      label: 'Permutation Importance',
      description: 'Measures feature importance by observing model performance changes',
      supportsGlobal: true,
      supportsLocal: false
    }
  ];

  useEffect(() => {
    if (state.models.length > 0 && !selectedModel) {
      setSelectedModel(state.models[0].name);
    }
  }, [state.models, selectedModel]);

  const generateExplanations = async (type: 'global' | 'local') => {
    if (!selectedModel) {
      toast.error('Please select a model first');
      return;
    }

    setIsGenerating(true);
    try {
      const response = await fetch('http://localhost:8000/explanations/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: selectedModel,
          method: selectedMethod,
          explanation_type: type,
          sample_size: type === 'local' ? 10 : undefined
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate explanations');
      }

      const result = await response.json();
      
      const newExplanation: ExplanationResult = {
        method: selectedMethod,
        type: type,
        model_name: selectedModel,
        data: result,
        feature_importance: result.feature_importance,
        sample_explanations: result.sample_explanations
      };

      setExplanations(prev => [...prev, newExplanation]);
      
      // Update app state
      dispatch({
        type: 'ADD_EXPLANATION',
        payload: newExplanation
      });

      toast.success(`${type.charAt(0).toUpperCase() + type.slice(1)} explanations generated successfully!`);
    } catch (error) {
      console.error('Explanation error:', error);
      toast.error('Failed to generate explanations. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const prepareFeatureImportanceChart = (importance: Record<string, number>): ChartData => {
    const sortedFeatures = Object.entries(importance)
      .sort(([,a], [,b]) => Math.abs(b) - Math.abs(a))
      .slice(0, 10);

    return {
      labels: sortedFeatures.map(([feature]) => feature),
      values: sortedFeatures.map(([, value]) => value),
      colors: sortedFeatures.map(([, value]) => value >= 0 ? '#10b981' : '#ef4444')
    };
  };

  const FeatureImportanceChart = ({ data }: { data: ChartData }) => {
    const maxValue = Math.max(...data.values.map(Math.abs));
    
    return (
      <div className="space-y-3">
        {data.labels.map((label, index) => {
          const value = data.values[index];
          const width = (Math.abs(value) / maxValue) * 100;
          const isPositive = value >= 0;
          
          return (
            <div key={label} className="flex items-center space-x-4">
              <div className="w-32 text-sm text-gray-700 truncate" title={label}>
                {label}
              </div>
              <div className="flex-1 relative">
                <div className="flex items-center h-8">
                  <div 
                    className={`h-6 rounded ${isPositive ? 'bg-green-500' : 'bg-red-500'} transition-all duration-500`}
                    style={{ width: `${width}%` }}
                  />
                  <span className="ml-2 text-sm font-medium text-gray-900">
                    {value.toFixed(3)}
                  </span>
                  {isPositive ? (
                    <ArrowUpIcon className="w-4 h-4 ml-1 text-green-500" />
                  ) : (
                    <ArrowDownIcon className="w-4 h-4 ml-1 text-red-500" />
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const LocalExplanationView = ({ explanation }: { explanation: ExplanationResult }) => {
    if (!explanation.sample_explanations) return null;

    const currentSample = explanation.sample_explanations[selectedSample];
    if (!currentSample) return null;

    const chartData = prepareFeatureImportanceChart(currentSample.feature_contributions);

    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h4 className="text-lg font-semibold text-gray-900">Sample Explanation</h4>
          <Select
            value={selectedSample.toString()}
            onChange={(value) => setSelectedSample(parseInt(value))}
          >
            {explanation.sample_explanations.map((_, index) => (
              <option key={index} value={index.toString()}>
                Sample {index + 1}
              </option>
            ))}
          </Select>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <p className="text-2xl font-bold text-blue-600">
              {currentSample.prediction.toFixed(3)}
            </p>
            <p className="text-sm text-gray-600">Prediction</p>
          </div>
          {currentSample.actual !== undefined && (
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <p className="text-2xl font-bold text-green-600">
                {currentSample.actual.toFixed(3)}
              </p>
              <p className="text-sm text-gray-600">Actual</p>
            </div>
          )}
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <p className="text-2xl font-bold text-purple-600">
              {currentSample.sample_index + 1}
            </p>
            <p className="text-sm text-gray-600">Sample Index</p>
          </div>
        </div>

        <div>
          <h5 className="font-medium text-gray-900 mb-4">Feature Contributions</h5>
          <FeatureImportanceChart data={chartData} />
        </div>
      </div>
    );
  };

  if (!state.data) {
    return (
      <div className="max-w-4xl mx-auto px-6 py-8">
        <Alert variant="warning">
          <ExclamationTriangleIcon className="w-5 h-5 mr-2" />
          <div>
            <p className="font-medium">No Data Available</p>
            <p className="text-sm">Please upload a dataset first before generating explanations.</p>
          </div>
        </Alert>
      </div>
    );
  }

  if (state.models.length === 0) {
    return (
      <div className="max-w-4xl mx-auto px-6 py-8">
        <Alert variant="warning">
          <ExclamationTriangleIcon className="w-5 h-5 mr-2" />
          <div>
            <p className="font-medium">No Models Available</p>
            <p className="text-sm">Please train some models first before generating explanations.</p>
          </div>
        </Alert>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Explanations</h1>
        <p className="text-lg text-gray-600">
          Understand how your models make predictions with SHAP, LIME, and other explanation methods
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Controls */}
        <div className="space-y-6">
          <Card>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Generate Explanations</h3>
            
            <div className="space-y-4">
              <Select
                label="Select Model"
                value={selectedModel}
                onChange={setSelectedModel}
                required
              >
                {state.models.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.displayName}
                  </option>
                ))}
              </Select>

              <Select
                label="Explanation Method"
                value={selectedMethod}
                onChange={setSelectedMethod}
                required
              >
                {explanationMethods.map((method) => (
                  <option key={method.value} value={method.value}>
                    {method.label}
                  </option>
                ))}
              </Select>

              <div className="bg-blue-50 p-4 rounded-lg">
                <p className="text-sm text-blue-800">
                  {explanationMethods.find(m => m.value === selectedMethod)?.description}
                </p>
              </div>
            </div>
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Explanation Types</h3>
            
            <div className="space-y-3">
              <Button
                variant="primary"
                size="md"
                onClick={() => generateExplanations('global')}
                disabled={isGenerating || !explanationMethods.find(m => m.value === selectedMethod)?.supportsGlobal}
                className="w-full"
              >
                <GlobeAltIcon className="w-5 h-5 mr-2" />
                {isGenerating ? 'Generating...' : 'Global Explanations'}
              </Button>
              
              <Button
                variant="secondary"
                size="md"
                onClick={() => generateExplanations('local')}
                disabled={isGenerating || !explanationMethods.find(m => m.value === selectedMethod)?.supportsLocal}
                className="w-full"
              >
                <UserIcon className="w-5 h-5 mr-2" />
                {isGenerating ? 'Generating...' : 'Local Explanations'}
              </Button>
            </div>

            {isGenerating && (
              <div className="mt-4 text-center">
                <LoadingSpinner size="md" color="blue" />
                <p className="text-sm text-gray-600 mt-2">
                  Generating explanations...
                </p>
              </div>
            )}
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Explanation Methods</h3>
            <div className="space-y-4">
              {explanationMethods.map((method) => (
                <div key={method.value} className="border-l-4 border-blue-500 pl-4">
                  <h4 className="font-medium text-gray-900">{method.label}</h4>
                  <p className="text-sm text-gray-600 mt-1">{method.description}</p>
                  <div className="flex space-x-4 mt-2">
                    {method.supportsGlobal && (
                      <span className="text-xs px-2 py-1 bg-green-100 text-green-800 rounded">
                        Global
                      </span>
                    )}
                    {method.supportsLocal && (
                      <span className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">
                        Local
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-6">
          {explanations.length === 0 ? (
            <Card className="text-center py-12">
              <EyeIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No Explanations Yet</h3>
              <p className="text-gray-600">
                Generate explanations to understand how your models make predictions
              </p>
            </Card>
          ) : (
            explanations.map((explanation, index) => (
              <Card key={index} padding="lg">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-xl font-bold text-gray-900">
                      {explanation.method.toUpperCase()} - {explanation.type.charAt(0).toUpperCase() + explanation.type.slice(1)} Explanations
                    </h3>
                    <p className="text-gray-600">
                      Model: {state.models.find(m => m.name === explanation.model_name)?.displayName}
                    </p>
                  </div>
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                    explanation.type === 'global' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-blue-100 text-blue-800'
                  }`}>
                    {explanation.type === 'global' ? 'Global' : 'Local'}
                  </div>
                </div>

                {explanation.type === 'global' && explanation.feature_importance && (
                  <div>
                    <h4 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance</h4>
                    <FeatureImportanceChart data={prepareFeatureImportanceChart(explanation.feature_importance)} />
                  </div>
                )}

                {explanation.type === 'local' && (
                  <LocalExplanationView explanation={explanation} />
                )}

                <div className="mt-6 pt-6 border-t">
                  <Alert variant="info">
                    <InformationCircleIcon className="w-5 h-5 mr-2" />
                    <div className="text-sm">
                      {explanation.type === 'global' 
                        ? 'Global explanations show which features are most important across all predictions.'
                        : 'Local explanations show how individual features contribute to specific predictions.'
                      }
                    </div>
                  </Alert>
                </div>
              </Card>
            ))
          )}
        </div>
      </div>

      {/* Legend */}
      <Card className="mt-8" padding="lg">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Understanding the Charts</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Color Coding</h4>
            <div className="space-y-2">
              <div className="flex items-center">
                <div className="w-4 h-4 bg-green-500 rounded mr-3"></div>
                <span className="text-sm text-gray-700">Positive contribution (increases prediction)</span>
              </div>
              <div className="flex items-center">
                <div className="w-4 h-4 bg-red-500 rounded mr-3"></div>
                <span className="text-sm text-gray-700">Negative contribution (decreases prediction)</span>
              </div>
            </div>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-2">Bar Length</h4>
            <p className="text-sm text-gray-700">
              The length of each bar represents the magnitude of the feature's contribution to the prediction.
              Longer bars indicate more important features.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}

export default Explanations;