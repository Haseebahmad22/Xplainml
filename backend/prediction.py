"""
Prediction module for XplainML
Handles making predictions on new data with trained models
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class Predictor:
    """
    Unified prediction interface for trained models
    """
    
    def __init__(self, model=None, preprocessor=None):
        """
        Initialize predictor
        
        Args:
            model: Trained ML model
            preprocessor: Fitted data preprocessor
        """
        self.model = model
        self.preprocessor = preprocessor
        self.prediction_history = []
    
    def predict(self, X, return_probabilities=False, include_confidence=False):
        """
        Make predictions on new data
        
        Args:
            X (DataFrame/array): Input features
            return_probabilities (bool): Return probabilities for classification
            include_confidence (bool): Include confidence scores
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        if not self.model.is_fitted:
            raise ValueError("Model is not fitted. Please train the model first.")
        
        # Preprocess data if preprocessor is available
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform_new_data(X)
        else:
            X_processed = X
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        
        result = {
            'predictions': predictions,
            'input_shape': X.shape,
            'model_type': self.model.model_type,
            'task_type': self.model.task_type
        }
        
        # Add probabilities for classification
        if (self.model.task_type == 'classification' and 
            return_probabilities and 
            hasattr(self.model, 'predict_proba')):
            try:
                probabilities = self.model.predict_proba(X_processed)
                result['probabilities'] = probabilities
                
                # Add confidence scores
                if include_confidence:
                    if probabilities.shape[1] == 2:
                        # Binary classification - confidence is max probability
                        confidence = np.max(probabilities, axis=1)
                    else:
                        # Multi-class - confidence is max probability
                        confidence = np.max(probabilities, axis=1)
                    
                    result['confidence'] = confidence
                    
            except Exception as e:
                print(f"Warning: Could not compute probabilities: {str(e)}")
        
        # Add confidence for regression (based on feature similarity to training data)
        if self.model.task_type == 'regression' and include_confidence:
            # Simple confidence based on prediction variance (for ensemble models)
            if hasattr(self.model.model, 'estimators_'):
                try:
                    # For Random Forest and similar ensemble methods
                    individual_predictions = np.array([
                        estimator.predict(X_processed) 
                        for estimator in self.model.model.estimators_
                    ])
                    prediction_variance = np.var(individual_predictions, axis=0)
                    # Higher variance = lower confidence
                    confidence = 1 / (1 + prediction_variance)
                    result['confidence'] = confidence
                except:
                    pass
        
        # Store prediction in history
        self._add_to_history(X, result)
        
        return result
    
    def predict_single(self, sample, return_probabilities=False, include_confidence=False):
        """
        Make prediction for a single sample
        
        Args:
            sample (dict/Series): Single sample as dictionary or pandas Series
            return_probabilities (bool): Return probabilities for classification
            include_confidence (bool): Include confidence scores
            
        Returns:
            dict: Single prediction result
        """
        # Convert single sample to DataFrame
        if isinstance(sample, dict):
            X = pd.DataFrame([sample])
        elif isinstance(sample, pd.Series):
            X = pd.DataFrame([sample])
        else:
            raise ValueError("Sample must be a dictionary or pandas Series")
        
        result = self.predict(X, return_probabilities, include_confidence)
        
        # Extract single prediction
        single_result = {
            'prediction': result['predictions'][0],
            'model_type': result['model_type'],
            'task_type': result['task_type']
        }
        
        if 'probabilities' in result:
            single_result['probabilities'] = result['probabilities'][0]
        
        if 'confidence' in result:
            single_result['confidence'] = result['confidence'][0]
        
        return single_result
    
    def predict_batch(self, data_path, output_path=None, chunk_size=1000):
        """
        Make predictions on large datasets in batches
        
        Args:
            data_path (str): Path to input data file
            output_path (str): Path to save predictions (optional)
            chunk_size (int): Size of batches for processing
            
        Returns:
            DataFrame: Predictions for all samples
        """
        print(f"Starting batch prediction on {data_path}")
        
        # Read data in chunks
        try:
            if data_path.endswith('.csv'):
                data_iterator = pd.read_csv(data_path, chunksize=chunk_size)
            elif data_path.endswith(('.xlsx', '.xls')):
                # For Excel, read all at once and then chunk
                full_data = pd.read_excel(data_path)
                data_iterator = [full_data[i:i+chunk_size] 
                               for i in range(0, len(full_data), chunk_size)]
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            raise ValueError(f"Error reading data file: {str(e)}")
        
        all_predictions = []
        total_processed = 0
        
        for chunk_idx, chunk in enumerate(data_iterator):
            print(f"Processing chunk {chunk_idx + 1}, samples: {len(chunk)}")
            
            # Make predictions for this chunk
            chunk_result = self.predict(chunk, return_probabilities=True, include_confidence=True)
            
            # Create DataFrame with predictions
            chunk_predictions = pd.DataFrame()
            chunk_predictions['prediction'] = chunk_result['predictions']
            
            if 'probabilities' in chunk_result:
                proba = chunk_result['probabilities']
                if proba.shape[1] == 2:
                    chunk_predictions['probability_class_0'] = proba[:, 0]
                    chunk_predictions['probability_class_1'] = proba[:, 1]
                else:
                    for i in range(proba.shape[1]):
                        chunk_predictions[f'probability_class_{i}'] = proba[:, i]
            
            if 'confidence' in chunk_result:
                chunk_predictions['confidence'] = chunk_result['confidence']
            
            # Add original data (optional)
            chunk_predictions = pd.concat([chunk.reset_index(drop=True), 
                                         chunk_predictions.reset_index(drop=True)], axis=1)
            
            all_predictions.append(chunk_predictions)
            total_processed += len(chunk)
            
        # Combine all predictions
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        
        print(f"Batch prediction completed. Total samples processed: {total_processed}")
        
        # Save to file if output path provided
        if output_path:
            if output_path.endswith('.csv'):
                final_predictions.to_csv(output_path, index=False)
            elif output_path.endswith(('.xlsx', '.xls')):
                final_predictions.to_excel(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        return final_predictions
    
    def explain_prediction(self, sample, explainer=None):
        """
        Explain a single prediction using SHAP or LIME
        
        Args:
            sample: Single sample to explain
            explainer: Fitted explainer object
            
        Returns:
            dict: Explanation results
        """
        if explainer is None:
            print("No explainer provided. Use the explainer module to create explanations.")
            return None
        
        # This will be implemented when we have the explainer module
        return explainer.explain_instance(sample, self.model)
    
    def _add_to_history(self, X, result):
        """Add prediction to history"""
        self.prediction_history.append({
            'timestamp': pd.Timestamp.now(),
            'input_shape': X.shape,
            'predictions': result['predictions'],
            'model_type': result['model_type']
        })
        
        # Keep only last 100 predictions to avoid memory issues
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
    
    def get_prediction_summary(self):
        """Get summary of recent predictions"""
        if not self.prediction_history:
            return "No predictions made yet."
        
        summary = {
            'total_predictions': sum(h['input_shape'][0] for h in self.prediction_history),
            'prediction_sessions': len(self.prediction_history),
            'model_type': self.prediction_history[-1]['model_type'],
            'latest_prediction_time': self.prediction_history[-1]['timestamp']
        }
        
        return summary
    
    def save_predictor(self, filepath):
        """Save the predictor (model + preprocessor)"""
        predictor_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'prediction_history': self.prediction_history
        }
        
        joblib.dump(predictor_data, filepath)
        print(f"Predictor saved to {filepath}")
    
    @classmethod
    def load_predictor(cls, filepath):
        """Load a saved predictor"""
        predictor_data = joblib.load(filepath)
        
        instance = cls(
            model=predictor_data['model'],
            preprocessor=predictor_data['preprocessor']
        )
        instance.prediction_history = predictor_data.get('prediction_history', [])
        
        print(f"Predictor loaded from {filepath}")
        return instance


class PredictionValidator:
    """
    Validate predictions and detect potential issues
    """
    
    def __init__(self, model, X_train=None, y_train=None):
        """
        Initialize validator
        
        Args:
            model: Trained model
            X_train: Training features (for reference)
            y_train: Training targets (for reference)
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        
        # Compute training data statistics if available
        if X_train is not None:
            self.train_stats = {
                'mean': X_train.mean() if hasattr(X_train, 'mean') else None,
                'std': X_train.std() if hasattr(X_train, 'std') else None,
                'min': X_train.min() if hasattr(X_train, 'min') else None,
                'max': X_train.max() if hasattr(X_train, 'max') else None
            }
        else:
            self.train_stats = None
    
    def validate_input(self, X):
        """
        Validate input data for prediction
        
        Args:
            X: Input features
            
        Returns:
            dict: Validation results
        """
        issues = []
        warnings = []
        
        # Check for missing values
        if hasattr(X, 'isnull') and X.isnull().any().any():
            issues.append("Input contains missing values")
        
        # Check feature count
        expected_features = len(self.model.feature_names) if self.model.feature_names else None
        if expected_features and X.shape[1] != expected_features:
            issues.append(f"Expected {expected_features} features, got {X.shape[1]}")
        
        # Check for data drift (if training stats available)
        if self.train_stats is not None and hasattr(X, 'select_dtypes'):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in self.train_stats['mean'].index:
                    # Check if values are within reasonable range of training data
                    train_min = self.train_stats['min'][col]
                    train_max = self.train_stats['max'][col]
                    
                    if X[col].min() < train_min - 3 * self.train_stats['std'][col]:
                        warnings.append(f"Column '{col}' has values much lower than training data")
                    
                    if X[col].max() > train_max + 3 * self.train_stats['std'][col]:
                        warnings.append(f"Column '{col}' has values much higher than training data")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def validate_predictions(self, predictions, X=None):
        """
        Validate prediction outputs
        
        Args:
            predictions: Model predictions
            X: Input features (optional)
            
        Returns:
            dict: Validation results
        """
        issues = []
        warnings = []
        
        # Check for NaN predictions
        if np.any(np.isnan(predictions)):
            issues.append("Predictions contain NaN values")
        
        # Check for infinite predictions
        if np.any(np.isinf(predictions)):
            issues.append("Predictions contain infinite values")
        
        # For classification, check if predictions are valid
        if self.model.task_type == 'classification':
            unique_preds = np.unique(predictions)
            if self.y_train is not None:
                valid_classes = np.unique(self.y_train)
                invalid_preds = set(unique_preds) - set(valid_classes)
                if invalid_preds:
                    issues.append(f"Predictions contain invalid classes: {invalid_preds}")
        
        # For regression, check for extreme values
        if self.model.task_type == 'regression' and self.y_train is not None:
            train_min, train_max = self.y_train.min(), self.y_train.max()
            train_range = train_max - train_min
            
            pred_min, pred_max = predictions.min(), predictions.max()
            
            if pred_min < train_min - 2 * train_range:
                warnings.append("Predictions contain very low values compared to training data")
            
            if pred_max > train_max + 2 * train_range:
                warnings.append("Predictions contain very high values compared to training data")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'prediction_stats': {
                'min': np.min(predictions),
                'max': np.max(predictions),
                'mean': np.mean(predictions),
                'std': np.std(predictions)
            }
        }


def create_prediction_report(predictor, X_test, y_test=None):
    """
    Create a comprehensive prediction report
    
    Args:
        predictor: Fitted Predictor object
        X_test: Test features
        y_test: Test targets (optional)
        
    Returns:
        dict: Comprehensive prediction report
    """
    print("Generating prediction report...")
    
    # Make predictions
    result = predictor.predict(X_test, return_probabilities=True, include_confidence=True)
    
    # Validate predictions
    validator = PredictionValidator(predictor.model)
    input_validation = validator.validate_input(X_test)
    pred_validation = validator.validate_predictions(result['predictions'], X_test)
    
    report = {
        'prediction_summary': {
            'total_samples': len(result['predictions']),
            'model_type': result['model_type'],
            'task_type': result['task_type'],
            'prediction_time': pd.Timestamp.now()
        },
        'input_validation': input_validation,
        'prediction_validation': pred_validation,
        'predictions': result['predictions']
    }
    
    # Add evaluation metrics if true labels available
    if y_test is not None:
        metrics = predictor.model.evaluate(X_test, y_test)
        report['evaluation_metrics'] = metrics
    
    # Add probabilities and confidence if available
    if 'probabilities' in result:
        report['probabilities'] = result['probabilities']
    
    if 'confidence' in result:
        report['confidence_stats'] = {
            'mean_confidence': np.mean(result['confidence']),
            'min_confidence': np.min(result['confidence']),
            'max_confidence': np.max(result['confidence']),
            'low_confidence_samples': np.sum(result['confidence'] < 0.7)
        }
    
    # Summary statistics
    if result['task_type'] == 'classification':
        unique, counts = np.unique(result['predictions'], return_counts=True)
        report['class_distribution'] = dict(zip(unique, counts))
    else:
        report['prediction_distribution'] = {
            'min': np.min(result['predictions']),
            'max': np.max(result['predictions']),
            'mean': np.mean(result['predictions']),
            'std': np.std(result['predictions'])
        }
    
    print("Prediction report generated successfully!")
    return report


if __name__ == "__main__":
    print("XplainML Prediction Module")
    
    # Example usage
    from data_preprocessing import DataPreprocessor, create_sample_data
    from models import MLModel
    
    # Create sample data and train a model
    create_sample_data('data/sample_data.csv')
    
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline('data/sample_data.csv', 'target')
    
    # Train a model
    model = MLModel('random_forest', data['task_type'])
    model.fit(data['X_train'], data['y_train'])
    
    # Create predictor
    predictor = Predictor(model, preprocessor)
    
    # Make predictions
    result = predictor.predict(data['X_test'], return_probabilities=True, include_confidence=True)
    print(f"Made predictions for {len(result['predictions'])} samples")
    
    # Generate report
    report = create_prediction_report(predictor, data['X_test'], data['y_test'])
    print("Example prediction completed!")