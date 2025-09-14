import React, { useState, useCallback } from 'react';
import { useAppContext } from '../context/AppContext';
import { Card, Button, Alert, Input, Select, LoadingSpinner } from '../components/ui';
import { 
  CubeTransparentIcon, 
  DocumentArrowUpIcon, 
  TableCellsIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ArrowDownTrayIcon,
  PlusIcon,
  TrashIcon,
  PlayIcon
} from '@heroicons/react/24/outline';
import { toast } from 'react-hot-toast';

interface PredictionInput {
  [key: string]: string | number;
}

interface PredictionResult {
  prediction: number;
  confidence?: number;
  probabilities?: Record<string, number>;
}

interface BatchPredictionResult {
  predictions: Array<{
    input: PredictionInput;
    prediction: number;
    confidence?: number;
  }>;
  summary: {
    total: number;
    avg_prediction: number;
    min_prediction: number;
    max_prediction: number;
  };
}

function Predictions() {
  const { state } = useAppContext();
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [predictionMode, setPredictionMode] = useState<'single' | 'batch' | 'file'>('single');
  
  // Single prediction state
  const [singleInput, setSingleInput] = useState<PredictionInput>({});
  const [singleResult, setSingleResult] = useState<PredictionResult | null>(null);
  
  // Batch prediction state
  const [batchInputs, setBatchInputs] = useState<PredictionInput[]>([{}]);
  const [batchResults, setBatchResults] = useState<BatchPredictionResult | null>(null);
  
  // File prediction state
  const [isDragging, setIsDragging] = useState(false);
  const [fileResults, setFileResults] = useState<any>(null);
  
  const [isLoading, setIsLoading] = useState(false);

  // Get feature columns (excluding target)
  const getFeatureColumns = () => {
    if (!state.data) return [];
    return state.data.columns.filter(col => col !== state.data?.targetColumn);
  };

  const handleSingleInputChange = (column: string, value: string) => {
    setSingleInput(prev => ({
      ...prev,
      [column]: isNaN(Number(value)) ? value : Number(value)
    }));
  };

  const handleBatchInputChange = (index: number, column: string, value: string) => {
    setBatchInputs(prev => {
      const newInputs = [...prev];
      newInputs[index] = {
        ...newInputs[index],
        [column]: isNaN(Number(value)) ? value : Number(value)
      };
      return newInputs;
    });
  };

  const addBatchInput = () => {
    setBatchInputs(prev => [...prev, {}]);
  };

  const removeBatchInput = (index: number) => {
    if (batchInputs.length > 1) {
      setBatchInputs(prev => prev.filter((_, i) => i !== index));
    }
  };

  const makeSinglePrediction = async () => {
    if (!selectedModel) {
      toast.error('Please select a model first');
      return;
    }

    const featureColumns = getFeatureColumns();
    const missingColumns = featureColumns.filter(col => !(col in singleInput) || singleInput[col] === '');
    
    if (missingColumns.length > 0) {
      toast.error(`Please fill in all fields: ${missingColumns.join(', ')}`);
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/predictions/single', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: selectedModel,
          input_data: singleInput
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to make prediction');
      }

      const result = await response.json();
      setSingleResult(result);
      toast.success('Prediction completed successfully!');
    } catch (error) {
      console.error('Prediction error:', error);
      toast.error('Failed to make prediction. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const makeBatchPredictions = async () => {
    if (!selectedModel) {
      toast.error('Please select a model first');
      return;
    }

    const featureColumns = getFeatureColumns();
    const validInputs = batchInputs.filter(input => {
      return featureColumns.every(col => col in input && input[col] !== '');
    });

    if (validInputs.length === 0) {
      toast.error('Please fill in at least one complete set of inputs');
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/predictions/batch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_name: selectedModel,
          input_data: validInputs
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to make batch predictions');
      }

      const result = await response.json();
      setBatchResults(result);
      toast.success(`Batch predictions completed for ${validInputs.length} samples!`);
    } catch (error) {
      console.error('Batch prediction error:', error);
      toast.error('Failed to make batch predictions. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, []);

  const handleFileUpload = async (file: File) => {
    if (!selectedModel) {
      toast.error('Please select a model first');
      return;
    }

    if (!file.name.match(/\.(csv|xlsx|xls)$/i)) {
      toast.error('Please upload a CSV or Excel file');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_name', selectedModel);

    try {
      const response = await fetch('http://localhost:8000/predictions/file', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process file predictions');
      }

      const result = await response.json();
      setFileResults(result);
      toast.success('File predictions completed successfully!');
    } catch (error) {
      console.error('File prediction error:', error);
      toast.error('Failed to process file predictions. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const downloadResults = (results: any, filename: string) => {
    const dataStr = JSON.stringify(results, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = filename;
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  if (!state.data) {
    return (
      <div className="max-w-4xl mx-auto px-6 py-8">
        <Alert variant="warning">
          <ExclamationTriangleIcon className="w-5 h-5 mr-2" />
          <div>
            <p className="font-medium">No Data Available</p>
            <p className="text-sm">Please upload a dataset first before making predictions.</p>
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
            <p className="text-sm">Please train some models first before making predictions.</p>
          </div>
        </Alert>
      </div>
    );
  }

  const featureColumns = getFeatureColumns();

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Make Predictions</h1>
        <p className="text-lg text-gray-600">
          Use your trained models to make predictions on new data
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Controls */}
        <div className="space-y-6">
          <Card>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Selection</h3>
            <Select
              label="Select Model"
              value={selectedModel}
              onChange={setSelectedModel}
              required
            >
              <option value="">Choose a model...</option>
              {state.models.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.displayName}
                </option>
              ))}
            </Select>
          </Card>

          <Card>
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Mode</h3>
            <div className="space-y-2">
              {[
                { value: 'single', label: 'Single Prediction', icon: CubeTransparentIcon, desc: 'Make one prediction' },
                { value: 'batch', label: 'Batch Predictions', icon: TableCellsIcon, desc: 'Multiple predictions at once' },
                { value: 'file', label: 'File Upload', icon: DocumentArrowUpIcon, desc: 'Upload CSV/Excel file' }
              ].map((mode) => {
                const Icon = mode.icon;
                return (
                  <button
                    key={mode.value}
                    onClick={() => setPredictionMode(mode.value as any)}
                    className={`w-full p-4 text-left rounded-lg border transition-all duration-200 ${
                      predictionMode === mode.value
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <Icon className={`w-5 h-5 mt-0.5 ${
                        predictionMode === mode.value ? 'text-blue-600' : 'text-gray-500'
                      }`} />
                      <div>
                        <p className="font-medium text-gray-900">{mode.label}</p>
                        <p className="text-sm text-gray-600">{mode.desc}</p>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </Card>

          {selectedModel && (
            <Card>
              <h3 className="text-lg font-semibold text-gray-900 mb-3">Model Info</h3>
              <div className="space-y-2 text-sm">
                <p><span className="font-medium">Model:</span> {state.models.find(m => m.name === selectedModel)?.displayName}</p>
                <p><span className="font-medium">Task:</span> {state.data.taskType}</p>
                <p><span className="font-medium">Features:</span> {featureColumns.length}</p>
              </div>
            </Card>
          )}
        </div>

        {/* Main Content */}
        <div className="lg:col-span-3">
          {predictionMode === 'single' && (
            <Card padding="lg">
              <h2 className="text-xl font-bold text-gray-900 mb-6">Single Prediction</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                {featureColumns.map((column) => (
                  <Input
                    key={column}
                    label={column}
                    placeholder={`Enter ${column} value`}
                    value={singleInput[column]?.toString() || ''}
                    onChange={(value) => handleSingleInputChange(column, value)}
                    required
                  />
                ))}
              </div>

              <Button
                variant="primary"
                size="lg"
                onClick={makeSinglePrediction}
                disabled={isLoading || !selectedModel}
                loading={isLoading}
                className="w-full mb-6"
              >
                <PlayIcon className="w-5 h-5 mr-2" />
                Make Prediction
              </Button>

              {singleResult && (
                <div className="border-t pt-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Result</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center p-6 bg-blue-50 rounded-lg">
                      <p className="text-3xl font-bold text-blue-600">
                        {typeof singleResult.prediction === 'number' 
                          ? singleResult.prediction.toFixed(3)
                          : singleResult.prediction
                        }
                      </p>
                      <p className="text-sm text-gray-600">Prediction</p>
                    </div>
                    {singleResult.confidence && (
                      <div className="text-center p-6 bg-green-50 rounded-lg">
                        <p className="text-3xl font-bold text-green-600">
                          {(singleResult.confidence * 100).toFixed(1)}%
                        </p>
                        <p className="text-sm text-gray-600">Confidence</p>
                      </div>
                    )}
                    {state.data.taskType === 'classification' && singleResult.probabilities && (
                      <div className="p-6 bg-purple-50 rounded-lg">
                        <p className="text-lg font-bold text-purple-600 mb-2">Class Probabilities</p>
                        {Object.entries(singleResult.probabilities).map(([cls, prob]) => (
                          <div key={cls} className="flex justify-between text-sm">
                            <span>{cls}:</span>
                            <span className="font-medium">{(prob * 100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </Card>
          )}

          {predictionMode === 'batch' && (
            <Card padding="lg">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-gray-900">Batch Predictions</h2>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={addBatchInput}
                >
                  <PlusIcon className="w-4 h-4 mr-1" />
                  Add Input
                </Button>
              </div>

              <div className="space-y-6 mb-6">
                {batchInputs.map((input, index) => (
                  <div key={index} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-medium text-gray-900">Input {index + 1}</h3>
                      {batchInputs.length > 1 && (
                        <Button
                          variant="danger"
                          size="sm"
                          onClick={() => removeBatchInput(index)}
                        >
                          <TrashIcon className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {featureColumns.map((column) => (
                        <Input
                          key={column}
                          label={column}
                          placeholder={`Enter ${column}`}
                          value={input[column]?.toString() || ''}
                          onChange={(value) => handleBatchInputChange(index, column, value)}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <Button
                variant="primary"
                size="lg"
                onClick={makeBatchPredictions}
                disabled={isLoading || !selectedModel}
                loading={isLoading}
                className="w-full mb-6"
              >
                <PlayIcon className="w-5 h-5 mr-2" />
                Make Batch Predictions
              </Button>

              {batchResults && (
                <div className="border-t pt-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">Batch Results</h3>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => downloadResults(batchResults, 'batch_predictions.json')}
                    >
                      <ArrowDownTrayIcon className="w-4 h-4 mr-1" />
                      Download
                    </Button>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <p className="text-2xl font-bold text-blue-600">{batchResults.summary.total}</p>
                      <p className="text-sm text-gray-600">Total Predictions</p>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <p className="text-2xl font-bold text-green-600">
                        {batchResults.summary.avg_prediction.toFixed(3)}
                      </p>
                      <p className="text-sm text-gray-600">Average</p>
                    </div>
                    <div className="text-center p-4 bg-purple-50 rounded-lg">
                      <p className="text-2xl font-bold text-purple-600">
                        {batchResults.summary.min_prediction.toFixed(3)}
                      </p>
                      <p className="text-sm text-gray-600">Minimum</p>
                    </div>
                    <div className="text-center p-4 bg-orange-50 rounded-lg">
                      <p className="text-2xl font-bold text-orange-600">
                        {batchResults.summary.max_prediction.toFixed(3)}
                      </p>
                      <p className="text-sm text-gray-600">Maximum</p>
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            #
                          </th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Prediction
                          </th>
                          {batchResults.predictions[0]?.confidence && (
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Confidence
                            </th>
                          )}
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {batchResults.predictions.map((result, index) => (
                          <tr key={index}>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {index + 1}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                              {typeof result.prediction === 'number' 
                                ? result.prediction.toFixed(3) 
                                : result.prediction
                              }
                            </td>
                            {result.confidence && (
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                {(result.confidence * 100).toFixed(1)}%
                              </td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </Card>
          )}

          {predictionMode === 'file' && (
            <Card padding="lg">
              <h2 className="text-xl font-bold text-gray-900 mb-6">File Predictions</h2>
              
              <div
                className={`
                  border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300
                  ${isDragging 
                    ? 'border-blue-400 bg-blue-50' 
                    : 'border-gray-300 hover:border-gray-400'
                  }
                  ${isLoading ? 'pointer-events-none opacity-50' : 'cursor-pointer'}
                `}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
                onDrop={handleFileDrop}
                onClick={() => document.getElementById('file-input-predictions')?.click()}
              >
                {isLoading ? (
                  <div className="space-y-4">
                    <LoadingSpinner size="lg" color="blue" />
                    <p className="text-lg font-medium text-gray-900">Processing file...</p>
                  </div>
                ) : (
                  <>
                    <DocumentArrowUpIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      Drop your prediction file here
                    </h3>
                    <p className="text-gray-600 mb-4">
                      CSV or Excel file with the same features as your training data
                    </p>
                    <Button variant="primary" size="lg">
                      Choose File
                    </Button>
                  </>
                )}
              </div>

              <input
                id="file-input-predictions"
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                className="hidden"
                disabled={isLoading}
              />

              {fileResults && (
                <div className="mt-8 border-t pt-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-gray-900">File Results</h3>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => downloadResults(fileResults, 'file_predictions.json')}
                    >
                      <ArrowDownTrayIcon className="w-4 h-4 mr-1" />
                      Download Results
                    </Button>
                  </div>

                  <Alert variant="success">
                    <CheckCircleIcon className="w-5 h-5 mr-2" />
                    <div>
                      <p className="font-medium">File processed successfully!</p>
                      <p className="text-sm">
                        Generated {fileResults.total_predictions || 0} predictions
                      </p>
                    </div>
                  </Alert>
                </div>
              )}
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

export default Predictions;