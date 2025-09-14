// Data Types
export interface DataSummary {
  originalShape: [number, number];
  featureNames: string[];
  taskType: 'classification' | 'regression';
  targetColumn: string;
  missingValues: Record<string, number>;
  dataTypes: Record<string, string>;
}

export interface PreprocessedData {
  XTrain: any[][];
  XTest: any[][];
  yTrain: any[];
  yTest: any[];
  featureNames: string[];
  taskType: 'classification' | 'regression';
  originalShape: [number, number];
  preprocessorSettings: PreprocessorSettings;
}

export interface PreprocessorSettings {
  missingStrategy: 'mean' | 'median' | 'most_frequent';
  encodingType: 'auto' | 'label' | 'onehot';
  scalingType: 'standard' | 'minmax' | 'none';
  testSize: number;
}

// Model Types
export interface ModelConfig {
  type: 'linear' | 'random_forest' | 'xgboost' | 'neural_network';
  parameters?: Record<string, any>;
  useTuning: boolean;
  cvFolds: number;
}

export interface ModelMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
  r2?: number;
  mae?: number;
  rmse?: number;
  confusionMatrix?: number[][];
}

export interface TrainedModel {
  id: string;
  type: string;
  metrics: {
    train: ModelMetrics;
    test: ModelMetrics;
  };
  bestParams?: Record<string, any>;
  bestScore?: number;
  trainingTime: number;
  createdAt: string;
}

// Explanation Types
export interface ExplanationConfig {
  methods: ('shap' | 'lime' | 'permutation')[];
  numSamples: number;
}

export interface FeatureImportance {
  [featureName: string]: number;
}

export interface GlobalExplanation {
  method: string;
  featureImportance: FeatureImportance;
  topFeatures: string[];
}

export interface LocalExplanation {
  method: string;
  sampleIndex: number;
  prediction: any;
  featureContributions: Record<string, number>;
  explanation?: string;
}

// API Types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface UploadResponse {
  filename: string;
  dataSummary: DataSummary;
}

export interface PredictionRequest {
  data: Record<string, any>[] | Record<string, any>;
  modelId: string;
}

export interface PredictionResponse {
  predictions: any[];
  probabilities?: number[][];
  confidence?: number[];
}

// App State Types
export interface AppState {
  currentStep: number;
  dataLoaded: boolean;
  modelTrained: boolean;
  explanationsGenerated: boolean;
  
  // Data
  uploadedFile?: File;
  dataSummary?: DataSummary;
  preprocessedData?: PreprocessedData;
  
  // Model
  currentModel?: TrainedModel;
  availableModels: TrainedModel[];
  
  // Explanations
  globalExplanations: GlobalExplanation[];
  localExplanations: LocalExplanation[];
  
  // UI State
  isLoading: boolean;
  loadingMessage: string;
  error?: string;
}

// Component Props Types
export interface StepIndicatorProps {
  currentStep: number;
  steps: Array<{
    id: number;
    name: string;
    description: string;
    completed: boolean;
  }>;
}

export interface MetricCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color?: 'blue' | 'green' | 'purple' | 'red' | 'yellow';
  change?: {
    value: number;
    type: 'increase' | 'decrease';
  };
}

export interface ChartProps {
  data: any;
  type: 'bar' | 'line' | 'scatter' | 'pie' | 'heatmap';
  title?: string;
  xAxisLabel?: string;
  yAxisLabel?: string;
  height?: number;
}

// Form Types
export interface DataUploadForm {
  file: File | null;
  targetColumn: string;
  preprocessorSettings: PreprocessorSettings;
}

export interface ModelTrainingForm {
  modelType: ModelConfig['type'];
  useTuning: boolean;
  cvFolds: number;
  customParameters?: Record<string, any>;
}

export interface PredictionForm {
  predictionType: 'single' | 'batch' | 'file';
  features?: Record<string, any>;
  file?: File;
}