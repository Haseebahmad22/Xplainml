"""
Machine Learning Models module for XplainML
Supports various ML algorithms including deep learning with PyTorch
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_squared_error, mean_absolute_error, r2_score, roc_auc_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports (will handle import errors gracefully)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Neural network models will be disabled.")

# XGBoost import (handle gracefully)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. XGBoost models will be disabled.")


class MLModel:
    """
    Unified interface for different machine learning models
    """
    
    def __init__(self, model_type='random_forest', task_type='classification', **kwargs):
        """
        Initialize ML model
        
        Args:
            model_type (str): Type of model ('linear', 'random_forest', 'xgboost', 'neural_network')
            task_type (str): 'classification' or 'regression'
            **kwargs: Additional model parameters
        """
        self.model_type = model_type
        self.task_type = task_type
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.kwargs = kwargs
        
        # Initialize the appropriate model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified model type"""
        
        if self.model_type == 'linear':
            if self.task_type == 'classification':
                self.model = LogisticRegression(random_state=42, **self.kwargs)
            else:
                self.model = LinearRegression(**self.kwargs)
                
        elif self.model_type == 'random_forest':
            if self.task_type == 'classification':
                self.model = RandomForestClassifier(
                    n_estimators=100, random_state=42, **self.kwargs
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=100, random_state=42, **self.kwargs
                )
                
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Please install it to use XGBoost models.")
            
            if self.task_type == 'classification':
                self.model = xgb.XGBClassifier(random_state=42, **self.kwargs)
            else:
                self.model = xgb.XGBRegressor(random_state=42, **self.kwargs)
                
        elif self.model_type == 'neural_network':
            if not PYTORCH_AVAILABLE:
                raise ImportError("PyTorch is not installed. Please install it to use neural network models.")
            
            # Neural network will be initialized during fit
            self.model = None
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X, y, validation_split=0.2, epochs=100, batch_size=32, verbose=True):
        """
        Train the model
        
        Args:
            X (DataFrame/array): Training features
            y (Series/array): Training targets
            validation_split (float): Validation split for neural networks
            epochs (int): Training epochs for neural networks
            batch_size (int): Batch size for neural networks
            verbose (bool): Print training progress
        """
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        if self.model_type == 'neural_network':
            self._fit_neural_network(X, y, validation_split, epochs, batch_size, verbose)
        else:
            if verbose:
                print(f"Training {self.model_type} model...")
            self.model.fit(X, y)
            
        self.is_fitted = True
        
        if verbose:
            print("Model training completed!")
    
    def _fit_neural_network(self, X, y, validation_split, epochs, batch_size, verbose):
        """Fit PyTorch neural network"""
        
        # Convert to numpy if pandas
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        
        if self.task_type == 'classification':
            # Handle multi-class classification
            unique_labels = np.unique(y)
            if len(unique_labels) == 2:
                y_tensor = torch.FloatTensor(y).unsqueeze(1)
                output_size = 1
            else:
                y_tensor = torch.LongTensor(y)
                output_size = len(unique_labels)
        else:
            y_tensor = torch.FloatTensor(y).unsqueeze(1)
            output_size = 1
        
        # Split into train/validation
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        
        X_train, X_val = X_tensor[train_idx], X_tensor[val_idx]
        y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize neural network
        input_size = X.shape[1]
        hidden_sizes = self.kwargs.get('hidden_sizes', [64, 32])
        dropout_rate = self.kwargs.get('dropout_rate', 0.2)
        
        self.model = NeuralNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dropout_rate=dropout_rate,
            task_type=self.task_type
        )
        
        # Training setup
        if self.task_type == 'classification':
            if output_size == 1:
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(self.model.parameters(), lr=self.kwargs.get('learning_rate', 0.001))
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            # Validation
            self.model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    epoch_val_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Store training history
        self.training_history = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.model_type == 'neural_network':
            return self._predict_neural_network(X)
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities (classification only)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        if self.model_type == 'neural_network':
            return self._predict_proba_neural_network(X)
        else:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                # For models without predict_proba, use decision_function
                scores = self.model.decision_function(X)
                if scores.ndim == 1:
                    # Binary classification
                    proba_pos = 1 / (1 + np.exp(-scores))
                    return np.column_stack([1 - proba_pos, proba_pos])
                else:
                    # Multi-class classification
                    exp_scores = np.exp(scores)
                    return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    def _predict_neural_network(self, X):
        """Neural network predictions"""
        if hasattr(X, 'values'):
            X = X.values
            
        X_tensor = torch.FloatTensor(X)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if self.task_type == 'classification':
                if outputs.shape[1] == 1:
                    # Binary classification
                    predictions = (torch.sigmoid(outputs) > 0.5).float().squeeze().numpy()
                else:
                    # Multi-class classification
                    predictions = torch.argmax(outputs, dim=1).numpy()
            else:
                # Regression
                predictions = outputs.squeeze().numpy()
                
        return predictions
    
    def _predict_proba_neural_network(self, X):
        """Neural network probability predictions"""
        if hasattr(X, 'values'):
            X = X.values
            
        X_tensor = torch.FloatTensor(X)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if outputs.shape[1] == 1:
                # Binary classification
                proba_pos = torch.sigmoid(outputs).squeeze().numpy()
                return np.column_stack([1 - proba_pos, proba_pos])
            else:
                # Multi-class classification
                probabilities = torch.softmax(outputs, dim=1).numpy()
                return probabilities
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions = self.predict(X)
        
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(y, predictions, average='weighted', zero_division=0)
            }
            
            # Add AUC for binary classification
            if len(np.unique(y)) == 2:
                try:
                    proba = self.predict_proba(X)
                    metrics['auc'] = roc_auc_score(y, proba[:, 1])
                except:
                    pass
                    
        else:  # Regression
            metrics = {
                'mse': mean_squared_error(y, predictions),
                'mae': mean_absolute_error(y, predictions),
                'r2': r2_score(y, predictions),
                'rmse': np.sqrt(mean_squared_error(y, predictions))
            }
        
        return metrics
    
    def get_feature_importance(self):
        """Get feature importance (if available)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models
            importance = np.abs(self.model.coef_).flatten()
        else:
            # For neural networks and other models
            return None
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        else:
            return importance
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        if self.model_type == 'neural_network':
            # Save PyTorch model state dict
            model_data['model_state_dict'] = self.model.state_dict()
            model_data['model_config'] = {
                'input_size': self.model.input_size,
                'hidden_sizes': self.model.hidden_sizes,
                'output_size': self.model.output_size,
                'dropout_rate': self.model.dropout_rate,
                'task_type': self.model.task_type
            }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a saved model"""
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(
            model_type=model_data['model_type'],
            task_type=model_data['task_type']
        )
        
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        
        if model_data['model_type'] == 'neural_network':
            # Reconstruct PyTorch model
            config = model_data['model_config']
            instance.model = NeuralNetwork(
                input_size=config['input_size'],
                hidden_sizes=config['hidden_sizes'],
                output_size=config['output_size'],
                dropout_rate=config['dropout_rate'],
                task_type=config['task_type']
            )
            instance.model.load_state_dict(model_data['model_state_dict'])
            instance.model.eval()
        
        print(f"Model loaded from {filepath}")
        return instance


class NeuralNetwork(nn.Module):
    """
    PyTorch Neural Network for both classification and regression
    """
    
    def __init__(self, input_size, hidden_sizes=[64, 32], output_size=1, 
                 dropout_rate=0.2, task_type='classification'):
        super(NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.task_type = task_type
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ModelTuner:
    """
    Automated hyperparameter tuning for ML models
    """
    
    def __init__(self, model_type, task_type, search_type='grid'):
        self.model_type = model_type
        self.task_type = task_type
        self.search_type = search_type
        self.best_model = None
        self.best_params = None
        self.best_score = None
    
    def get_param_grid(self):
        """Get parameter grid for different models"""
        
        if self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == 'xgboost':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif self.model_type == 'linear':
            if self.task_type == 'classification':
                return {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            else:
                return {
                    'fit_intercept': [True, False],
                    'normalize': [True, False]
                }
        else:
            return {}
    
    def tune(self, X, y, cv=5, scoring=None):
        """
        Perform hyperparameter tuning
        
        Args:
            X: Training features
            y: Training targets
            cv: Cross-validation folds
            scoring: Scoring metric
        """
        
        # Initialize base model
        model = MLModel(self.model_type, self.task_type)
        
        # Get parameter grid
        param_grid = self.get_param_grid()
        
        if not param_grid:
            print(f"No hyperparameter tuning available for {self.model_type}")
            model.fit(X, y)
            self.best_model = model
            return model
        
        # Set default scoring
        if scoring is None:
            if self.task_type == 'classification':
                scoring = 'accuracy'
            else:
                scoring = 'r2'
        
        # Perform search
        if self.search_type == 'grid':
            search = GridSearchCV(
                model.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
            )
        else:
            search = RandomizedSearchCV(
                model.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, n_iter=20
            )
        
        print(f"Performing {self.search_type} search with {cv}-fold CV...")
        search.fit(X, y)
        
        # Store results
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        
        # Create best model
        self.best_model = MLModel(self.model_type, self.task_type, **self.best_params)
        self.best_model.model = search.best_estimator_
        self.best_model.is_fitted = True
        self.best_model.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best {scoring} score: {self.best_score:.4f}")
        
        return self.best_model


def compare_models(X_train, y_train, X_test, y_test, task_type='classification'):
    """
    Compare multiple models and return results
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        task_type: 'classification' or 'regression'
        
    Returns:
        dict: Results for each model
    """
    
    model_types = ['linear', 'random_forest']
    
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')
    
    if PYTORCH_AVAILABLE:
        model_types.append('neural_network')
    
    results = {}
    
    print(f"Comparing models for {task_type} task...")
    print("=" * 50)
    
    for model_type in model_types:
        print(f"\nTraining {model_type} model...")
        
        try:
            # Train model
            model = MLModel(model_type, task_type)
            model.fit(X_train, y_train, verbose=False)
            
            # Evaluate
            train_metrics = model.evaluate(X_train, y_train)
            test_metrics = model.evaluate(X_test, y_test)
            
            results[model_type] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
            
            print(f"✓ {model_type} completed")
            if task_type == 'classification':
                print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            else:
                print(f"  Test R²: {test_metrics['r2']:.4f}")
                
        except Exception as e:
            print(f"✗ {model_type} failed: {str(e)}")
            results[model_type] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    print("XplainML Models Module")
    
    # Example usage with sample data
    from data_preprocessing import DataPreprocessor, create_sample_data
    
    # Create and preprocess sample data
    create_sample_data('data/sample_data.csv')
    preprocessor = DataPreprocessor()
    
    data = preprocessor.preprocess_pipeline(
        'data/sample_data.csv',
        target_column='target'
    )
    
    # Compare models
    results = compare_models(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        task_type=data['task_type']
    )
    
    print("\nModel comparison completed!")