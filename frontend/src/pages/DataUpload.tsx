import React, { useState, useCallback } from 'react';
import { useAppContext } from '../context/AppContext';
import { Card, Button, Alert, ProgressBar, LoadingSpinner } from '../components/ui';
import { 
  CloudArrowUpIcon, 
  DocumentTextIcon, 
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ArrowRightIcon
} from '@heroicons/react/24/outline';
import { toast } from 'react-hot-toast';

interface DataSummary {
  fileName: string;
  fileSize: number;
  rows: number;
  columns: number;
  columnTypes: Record<string, string>;
  missingValues: Record<string, number>;
  preview: any[];
  targetColumn?: string;
  taskType?: 'classification' | 'regression';
}

function DataUpload() {
  const { state, dispatch } = useAppContext();
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [dataSummary, setDataSummary] = useState<DataSummary | null>(null);
  const [selectedTarget, setSelectedTarget] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleFileUpload = async (file: File) => {
    if (!file.name.match(/\.(csv|xlsx|xls)$/i)) {
      toast.error('Please upload a CSV or Excel file');
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/data/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to upload file');
      }

      const result = await response.json();
      setDataSummary({
        fileName: file.name,
        fileSize: file.size,
        rows: result.shape[0],
        columns: result.shape[1],
        columnTypes: result.column_types,
        missingValues: result.missing_values,
        preview: result.preview,
        targetColumn: undefined,
        taskType: undefined
      });

      toast.success('File uploaded successfully!');
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to upload file. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleTargetSelection = async () => {
    if (!selectedTarget) {
      toast.error('Please select a target column');
      return;
    }

    setIsProcessing(true);
    try {
      const response = await fetch('http://localhost:8000/data/preprocess', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          target_column: selectedTarget,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to process data');
      }

      const result = await response.json();
      
      // Update data summary with task type
      setDataSummary(prev => prev ? {
        ...prev,
        targetColumn: selectedTarget,
        taskType: result.task_type
      } : null);

      // Update app state
      dispatch({
        type: 'SET_DATA',
        payload: {
          fileName: dataSummary?.fileName || '',
          shape: [dataSummary?.rows || 0, dataSummary?.columns || 0],
          columns: Object.keys(dataSummary?.columnTypes || {}),
          targetColumn: selectedTarget,
          taskType: result.task_type,
          preview: dataSummary?.preview || []
        }
      });

      toast.success('Data processed successfully!');
    } catch (error) {
      console.error('Processing error:', error);
      toast.error('Failed to process data. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getColumnColor = (type: string) => {
    switch (type) {
      case 'int64':
      case 'float64':
        return 'text-blue-600 bg-blue-100';
      case 'object':
        return 'text-green-600 bg-green-100';
      case 'bool':
        return 'text-purple-600 bg-purple-100';
      case 'datetime64[ns]':
        return 'text-orange-600 bg-orange-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload Your Dataset</h1>
        <p className="text-lg text-gray-600">
          Start by uploading your CSV or Excel file to begin the machine learning journey
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Upload Section */}
        <div className="lg:col-span-2">
          <div className="card p-8">
            <div className="text-center">
              <div
                className={`
                  border-2 border-dashed rounded-xl p-12 transition-all duration-300
                  ${isDragging 
                    ? 'border-blue-400 bg-blue-50' 
                    : 'border-gray-300 hover:border-gray-400'
                  }
                  ${isUploading ? 'pointer-events-none opacity-50' : 'cursor-pointer'}
                `}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input')?.click()}
              >
                {isUploading ? (
                  <div className="space-y-4">
                    <div className="spinner w-12 h-12 mx-auto"></div>
                    <p className="text-lg font-medium text-gray-900">Uploading...</p>
                    <div className="progress-container h-3">
                      <div className="progress-bar" style={{width: '50%'}}></div>
                    </div>
                  </div>
                ) : (
                  <>
                    <CloudArrowUpIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">
                      Drop your file here, or click to select
                    </h3>
                    <p className="text-gray-600 mb-4">
                      Supports CSV, Excel (.xlsx, .xls) files up to 100MB
                    </p>
                    <button className="btn btn-primary">
                      Choose File
                    </button>
                  </>
                )}
              </div>
              
              <input
                id="file-input"
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileSelect}
                className="hidden"
                disabled={isUploading}
              />
            </div>
          </div>

          {/* Data Summary */}
          {dataSummary && (
            <div className="card p-8 mt-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-gray-900">Dataset Summary</h2>
                <div className="flex items-center text-green-600">
                  <CheckCircleIcon className="w-5 h-5 mr-1" />
                  <span className="text-sm font-medium">Uploaded</span>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <DocumentTextIcon className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                  <p className="text-2xl font-bold text-blue-600">{dataSummary.rows.toLocaleString()}</p>
                  <p className="text-sm text-gray-600">Rows</p>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <ChartBarIcon className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                  <p className="text-2xl font-bold text-purple-600">{dataSummary.columns}</p>
                  <p className="text-sm text-gray-600">Columns</p>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <DocumentTextIcon className="w-8 h-8 text-green-600 mx-auto mb-2" />
                  <p className="text-2xl font-bold text-green-600">{formatFileSize(dataSummary.fileSize)}</p>
                  <p className="text-sm text-gray-600">File Size</p>
                </div>
                <div className="text-center p-4 bg-orange-50 rounded-lg">
                  <ExclamationTriangleIcon className="w-8 h-8 text-orange-600 mx-auto mb-2" />
                  <p className="text-2xl font-bold text-orange-600">
                    {Object.values(dataSummary.missingValues).reduce((a, b) => a + b, 0)}
                  </p>
                  <p className="text-sm text-gray-600">Missing Values</p>
                </div>
              </div>

              {/* Column Types */}
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Column Types</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                  {Object.entries(dataSummary.columnTypes).map(([column, type]) => (
                    <div
                      key={column}
                      className={`px-3 py-2 rounded-lg text-sm font-medium ${getColumnColor(type)}`}
                    >
                      <div className="truncate" title={column}>{column}</div>
                      <div className="text-xs opacity-75">{type}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Target Column Selection */}
              <div className="border-t pt-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Select Target Column</h3>
                <p className="text-gray-600 mb-4">
                  Choose the column you want to predict (target variable)
                </p>
                
                <div className="flex flex-wrap gap-2 mb-4">
                  {Object.keys(dataSummary.columnTypes).map((column) => (
                    <button
                      key={column}
                      onClick={() => setSelectedTarget(column)}
                      className={`px-4 py-2 rounded-lg border transition-all duration-200 ${
                        selectedTarget === column
                          ? 'border-blue-500 bg-blue-50 text-blue-700'
                          : 'border-gray-300 bg-white text-gray-700 hover:border-gray-400'
                      }`}
                    >
                      {column}
                    </button>
                  ))}
                </div>

                <button
                  className="btn btn-primary w-full flex items-center justify-center"
                  onClick={handleTargetSelection}
                  disabled={!selectedTarget || isProcessing}
                >
                  {isProcessing ? (
                    <>
                      <div className="spinner w-4 h-4 mr-2"></div>
                      Processing...
                    </>
                  ) : (
                    <>
                      Process Data
                      <ArrowRightIcon className="w-5 h-5 ml-2" />
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Instructions */}
        <div className="space-y-6">
          <div className="card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Upload Instructions</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start">
                <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                Upload CSV or Excel files (max 100MB)
              </li>
              <li className="flex items-start">
                <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                Ensure your data has column headers
              </li>
              <li className="flex items-start">
                <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                Select the target column you want to predict
              </li>
              <li className="flex items-start">
                <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                Missing values will be handled automatically
              </li>
            </ul>
          </div>

          <div className="card p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Supported Formats</h3>
            <div className="space-y-3">
              <div className="flex items-center p-3 bg-green-50 rounded-lg">
                <DocumentTextIcon className="w-6 h-6 text-green-600 mr-3" />
                <div>
                  <p className="font-medium text-green-900">CSV Files</p>
                  <p className="text-sm text-green-700">Comma-separated values</p>
                </div>
              </div>
              <div className="flex items-center p-3 bg-blue-50 rounded-lg">
                <DocumentTextIcon className="w-6 h-6 text-blue-600 mr-3" />
                <div>
                  <p className="font-medium text-blue-900">Excel Files</p>
                  <p className="text-sm text-blue-700">.xlsx, .xls formats</p>
                </div>
              </div>
            </div>
          </div>

          {dataSummary?.taskType && (
            <div className="alert success p-4">
              <div className="flex items-center">
                <CheckCircleIcon className="w-5 h-5 mr-2" />
                <div>
                  <p className="font-medium">Task Type Detected</p>
                  <p className="text-sm">
                    {dataSummary.taskType === 'classification' ? 'Classification' : 'Regression'} problem
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Data Preview */}
      {dataSummary?.preview && (
        <div className="card p-8 mt-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Preview</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {Object.keys(dataSummary.columnTypes).map((column) => (
                    <th
                      key={column}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {column}
                      {column === selectedTarget && (
                        <span className="ml-1 text-blue-600">(Target)</span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {dataSummary.preview.slice(0, 5).map((row, index) => (
                  <tr key={index}>
                    {Object.keys(dataSummary.columnTypes).map((column) => (
                      <td key={column} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {row[column]?.toString() || 'N/A'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default DataUpload;