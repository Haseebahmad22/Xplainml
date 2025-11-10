import React, { useState, useCallback } from 'react';
import { useAppContext } from '../context/AppContext';
import { Stepper, Skeleton } from '../components/ui';
import { CloudArrowUpIcon, DocumentTextIcon, ChartBarIcon, ExclamationTriangleIcon, CheckCircleIcon, ArrowRightIcon, ArrowLeftIcon } from '@heroicons/react/24/outline';
import { toast } from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

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
  const { dispatch } = useAppContext();
  const navigate = useNavigate();
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [dataSummary, setDataSummary] = useState<DataSummary | null>(null);
  const [selectedTarget, setSelectedTarget] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  const steps = [
    { id: 'upload', label: 'Upload' },
    { id: 'summary', label: 'Summary' },
    { id: 'target', label: 'Target' },
    { id: 'process', label: 'Process' }
  ];

  const resetAll = () => {
    setIsDragging(false);
    setIsUploading(false);
    setDataSummary(null);
    setSelectedTarget('');
    setIsProcessing(false);
    setCurrentStep(0);
  };
  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); }, []);
  const handleDragLeave = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); }, []);
  const handleDrop = useCallback((e: React.DragEvent) => { e.preventDefault(); setIsDragging(false); const files = Array.from(e.dataTransfer.files); if (files[0]) handleFileUpload(files[0]); }, []);
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => { const file = e.target.files?.[0]; if (file) handleFileUpload(file); };

  const handleFileUpload = async (file: File) => {
    if (!file.name.match(/\.(csv|xlsx|xls)$/i)) { toast.error('Please upload a CSV or Excel file'); return; }
    setIsUploading(true);
    const formData = new FormData(); formData.append('file', file);
    try {
      const response = await fetch('http://localhost:8000/data/upload', { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Failed to upload file');
      const result = await response.json();
      setDataSummary({ fileName: file.name, fileSize: file.size, rows: result.shape[0], columns: result.shape[1], columnTypes: result.column_types, missingValues: result.missing_values, preview: result.preview });
      setCurrentStep(1);
      toast.success('File uploaded successfully!');
    } catch (err) { console.error(err); toast.error('Failed to upload file.'); } finally { setIsUploading(false); }
  };

  const handleTargetSelection = async () => {
    if (!selectedTarget) { toast.error('Select a target column'); return; }
    setIsProcessing(true);
    try {
      const response = await fetch('http://localhost:8000/data/preprocess', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ target_column: selectedTarget }) });
      if (!response.ok) throw new Error('Failed to process');
      const result = await response.json();
      setDataSummary(prev => prev ? { ...prev, targetColumn: selectedTarget, taskType: result.task_type } : null);
      if (dataSummary) {
        dispatch({ type: 'SET_DATA', payload: { fileName: dataSummary.fileName, shape: [dataSummary.rows, dataSummary.columns], columns: Object.keys(dataSummary.columnTypes), targetColumn: selectedTarget, taskType: result.task_type, preview: dataSummary.preview } });
      }
      toast.success('Data processed');
      setCurrentStep(3);
    } catch (err) { console.error(err); toast.error('Processing failed'); } finally { setIsProcessing(false); }
  };

  const formatFileSize = (bytes: number) => { if (bytes === 0) return '0 B'; const k = 1024; const sizes = ['B','KB','MB','GB']; const i = Math.floor(Math.log(bytes)/Math.log(k)); return `${(bytes/Math.pow(k,i)).toFixed(2)} ${sizes[i]}`; };
  const getColumnColor = (type: string) => { switch (type) { case 'int64': case 'float64': return 'text-blue-600 bg-blue-100'; case 'object': return 'text-green-600 bg-green-100'; case 'bool': return 'text-purple-600 bg-purple-100'; case 'datetime64[ns]': return 'text-orange-600 bg-orange-100'; default: return 'text-gray-600 bg-gray-100'; } };

  return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
          <div className="bg-gradient-to-r from-blue-600 to-purple-700 text-white">
            <div className="max-w-7xl mx-auto px-6 py-12">
              <h1 className="text-3xl font-bold mb-2">Upload Your Dataset</h1>
              <p className="text-sm opacity-90">Start by uploading your CSV or Excel file</p>
            </div>
          </div>

          <div className="max-w-7xl mx-auto px-6 py-8">
            <div className="mb-6 flex items-center justify-between">
              <Stepper steps={steps} current={currentStep} onStepClick={idx => { if (idx < currentStep) setCurrentStep(idx); }} />
              <div className="hidden md:flex gap-2">
                {currentStep > 0 && <button className="btn" onClick={() => setCurrentStep(currentStep-1)}><span className="inline-flex items-center text-gray-700"><ArrowLeftIcon className="w-4 h-4 mr-1"/>Back</span></button>}
                {currentStep === 1 && <button className="btn btn-primary" disabled={!dataSummary} onClick={() => setCurrentStep(2)}>Select Target<ArrowRightIcon className="w-4 h-4 ml-2"/></button>}
              </div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2">
                {currentStep === 0 && (
                  <div className="card p-8 card-shadow">
                    <div className="text-center">
                      <div className={`border-2 border-dashed rounded-xl p-12 transition-all duration-300 bg-white ${isDragging ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400'} ${isUploading ? 'pointer-events-none opacity-50' : 'cursor-pointer'}`} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop} onClick={() => document.getElementById('file-input')?.click()}>
                        {isUploading ? (
                          <div className="space-y-4">
                            <div className="spinner w-12 h-12 mx-auto"></div>
                            <p className="text-lg font-medium text-gray-900">Uploading...</p>
                            <div className="progress-container h-3"><div className="progress-bar" style={{ width: '50%' }}></div></div>
                          </div>
                        ) : (
                          <>
                            <CloudArrowUpIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                            <h3 className="text-xl font-semibold text-gray-900 mb-2">Drop your file here, or click to select</h3>
                            <p className="text-gray-600 mb-4">Supports CSV, Excel (.xlsx, .xls) files up to 100MB</p>
                            <button className="btn btn-primary">Choose File</button>
                          </>
                        )}
                      </div>
                      <input id="file-input" type="file" accept=".csv,.xlsx,.xls" onChange={handleFileSelect} className="hidden" disabled={isUploading} />
                    </div>
                  </div>
                )}

                {currentStep >= 1 && (
                  <div className="card p-8 card-shadow mt-6">
                    <div className="flex items-center justify-between mb-6">
                      <h2 className="text-xl font-bold text-gray-900">Dataset Summary</h2>
                      {dataSummary ? (
                        <div className="flex items-center text-green-600"><CheckCircleIcon className="w-5 h-5 mr-1" /><span className="text-sm font-medium">Uploaded</span></div>
                      ) : <div className="text-sm text-gray-500">Awaiting upload…</div>}
                    </div>
                    {dataSummary ? (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                        <div className="text-center p-4 bg-blue-50 rounded-lg"><DocumentTextIcon className="w-8 h-8 text-blue-600 mx-auto mb-2" /><p className="text-2xl font-bold text-blue-600">{dataSummary.rows.toLocaleString()}</p><p className="text-sm text-gray-600">Rows</p></div>
                        <div className="text-center p-4 bg-purple-50 rounded-lg"><ChartBarIcon className="w-8 h-8 text-purple-600 mx-auto mb-2" /><p className="text-2xl font-bold text-purple-600">{dataSummary.columns}</p><p className="text-sm text-gray-600">Columns</p></div>
                        <div className="text-center p-4 bg-green-50 rounded-lg"><DocumentTextIcon className="w-8 h-8 text-green-600 mx-auto mb-2" /><p className="text-2xl font-bold text-green-600">{formatFileSize(dataSummary.fileSize)}</p><p className="text-sm text-gray-600">File Size</p></div>
                        <div className="text-center p-4 bg-orange-50 rounded-lg"><ExclamationTriangleIcon className="w-8 h-8 text-orange-600 mx-auto mb-2" /><p className="text-2xl font-bold text-orange-600">{Object.values(dataSummary.missingValues).reduce((a,b)=>a+b,0)}</p><p className="text-sm text-gray-600">Missing Values</p></div>
                      </div>
                    ) : (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6"><Skeleton height={96}/><Skeleton height={96}/><Skeleton height={96}/><Skeleton height={96}/></div>
                    )}
                    <div className="mb-2">
                      <h3 className="text-lg font-semibold text-gray-900 mb-3">Column Types</h3>
                      {dataSummary ? (
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                          {Object.entries(dataSummary.columnTypes).map(([col, type]) => (
                            <div key={col} className={`px-3 py-2 rounded-lg text-sm font-medium ${getColumnColor(type)}`}><div className="truncate" title={col}>{col}</div><div className="text-xs opacity-75">{type}</div></div>
                          ))}
                        </div>
                      ) : (
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">{Array.from({length:8}).map((_,i)=><Skeleton key={i} height={40}/> )}</div>
                      )}
                    </div>
                    {currentStep === 1 && (
                      <div className="mt-6">
                        <button className="btn btn-primary w-full flex items-center justify-center" disabled={!dataSummary} onClick={() => setCurrentStep(2)}>Continue to Target Selection <ArrowRightIcon className="w-5 h-5 ml-2"/></button>
                      </div>
                    )}
                  </div>
                )}

                {currentStep === 2 && dataSummary && (
                  <div className="card p-8 mt-6 card-shadow">
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Select Target Column</h3>
                    <p className="text-gray-600 mb-4">Choose the column you want to predict</p>
                    <div className="flex flex-wrap gap-2 mb-4">
                      {Object.keys(dataSummary.columnTypes).map(c => (
                        <button key={c} onClick={() => setSelectedTarget(c)} className={`target-chip ${selectedTarget === c ? 'selected' : ''}`}>{c}</button>
                      ))}
                    </div>
                    <div className="flex gap-2">
                      <button className="btn" onClick={() => setCurrentStep(1)}><span className="inline-flex items-center text-gray-700"><ArrowLeftIcon className="w-4 h-4 mr-1"/>Back</span></button>
                      <button className="btn btn-primary flex-1 flex items-center justify-center" disabled={!selectedTarget || isProcessing} onClick={handleTargetSelection}>{isProcessing ? (<><div className="spinner w-4 h-4 mr-2"></div>Processing...</>) : (<>Process Data<ArrowRightIcon className="w-5 h-5 ml-2"/></>)}</button>
                    </div>
                  </div>
                )}

                {currentStep === 3 && dataSummary && (
                  <div className="card p-8 mt-6 card-shadow">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center text-green-600"><CheckCircleIcon className="w-6 h-6 mr-2" /><div><h3 className="text-lg font-semibold text-gray-900">Data Processed</h3><p className="text-sm text-gray-600">Task Type: {dataSummary.taskType}</p></div></div>
                      <div className="flex gap-2"><button className="btn" onClick={resetAll}>Upload another file</button><button className="btn btn-primary" onClick={() => navigate('/training')}>Go to Training</button></div>
                    </div>
                  </div>
                )}
              </div>
              <div className="space-y-6 lg:sticky lg:top-24 self-start">
                <div className="card p-6 card-shadow">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Upload Instructions</h3>
                  <ul className="space-y-2 text-sm text-gray-600">
                    <li className="flex items-start"><span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>Upload CSV or Excel files (max 100MB)</li>
                    <li className="flex items-start"><span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>Ensure your data has column headers</li>
                    <li className="flex items-start"><span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>Select the target column you want to predict</li>
                    <li className="flex items-start"><span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>Missing values will be handled automatically</li>
                  </ul>
                </div>
                <div className="card p-6 card-shadow">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Supported Formats</h3>
                  <div className="space-y-3">
                    <div className="flex items-center p-3 bg-green-50 rounded-lg"><DocumentTextIcon className="w-6 h-6 text-green-600 mr-3" /><div><p className="font-medium text-green-900">CSV Files</p><p className="text-sm text-green-700">Comma-separated values</p></div></div>
                    <div className="flex items-center p-3 bg-blue-50 rounded-lg"><DocumentTextIcon className="w-6 h-6 text-blue-600 mr-3" /><div><p className="font-medium text-blue-900">Excel Files</p><p className="text-sm text-blue-700">.xlsx, .xls formats</p></div></div>
                  </div>
                </div>
                {dataSummary?.taskType && (
                  <div className="alert success p-4">
                    <div className="flex items-center"><CheckCircleIcon className="w-5 h-5 mr-2" /><div><p className="font-medium">Task Type Detected</p><p className="text-sm">{dataSummary.taskType === 'classification' ? 'Classification' : 'Regression'} problem</p></div></div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {dataSummary && currentStep >= 1 && (
            <div className="card p-8 mt-8">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Data Preview</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      {Object.keys(dataSummary.columnTypes).map(col => (
                        <th key={col} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          {col}{col === selectedTarget && <span className="ml-1 text-blue-600">(Target)</span>}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {dataSummary.preview.slice(0,5).map((row,idx) => (
                      <tr key={idx}>
                        {Object.keys(dataSummary.columnTypes).map(col => (
                          <td key={col} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{row[col]?.toString() || 'N/A'}</td>
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