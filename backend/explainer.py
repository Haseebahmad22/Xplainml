"""
Explainability module for XplainML
Provides model interpretability using SHAP and LIME
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# SHAP imports (handle gracefully if not available)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# LIME imports (handle gracefully if not available)
try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

# Sklearn imports for permutation importance
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error


class ModelExplainer:
    """
    Comprehensive model explanation using multiple techniques
    """
    
    def __init__(self, model, X_train, y_train, feature_names=None):
        """
        Initialize explainer
        
        Args:
            model: Trained ML model
            X_train: Training features
            y_train: Training targets
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names or (
            list(X_train.columns) if hasattr(X_train, 'columns') 
            else [f'feature_{i}' for i in range(X_train.shape[1])]
        )
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        # Setup explainers
        self._setup_explainers()
    
    def _setup_explainers(self):
        """Setup SHAP and LIME explainers"""
        
        # Setup SHAP explainer
        if SHAP_AVAILABLE:
            try:
                if hasattr(self.model, 'model'):
                    # For our MLModel wrapper
                    model_obj = self.model.model
                else:
                    model_obj = self.model
                
                # Choose appropriate SHAP explainer based on model type
                if hasattr(model_obj, 'predict_proba'):
                    # Tree-based models
                    if hasattr(model_obj, 'estimators_'):
                        self.shap_explainer = shap.TreeExplainer(model_obj)
                    else:
                        # For other models, use KernelExplainer
                        background = shap.sample(self.X_train, min(100, len(self.X_train)))
                        self.shap_explainer = shap.KernelExplainer(
                            model_obj.predict_proba, background
                        )
                else:
                    # For regression or models without predict_proba
                    background = shap.sample(self.X_train, min(100, len(self.X_train)))
                    self.shap_explainer = shap.KernelExplainer(
                        model_obj.predict, background
                    )
                
                print("SHAP explainer initialized successfully")
                
            except Exception as e:
                print(f"Failed to initialize SHAP explainer: {str(e)}")
                self.shap_explainer = None
        
        # Setup LIME explainer
        if LIME_AVAILABLE:
            try:
                # Convert to numpy if pandas
                X_train_array = self.X_train.values if hasattr(self.X_train, 'values') else self.X_train
                
                if self.model.task_type == 'classification':
                    mode = 'classification'
                    class_names = np.unique(self.y_train).astype(str)
                else:
                    mode = 'regression'
                    class_names = None
                
                self.lime_explainer = lime_tabular.LimeTabularExplainer(
                    X_train_array,
                    feature_names=self.feature_names,
                    class_names=class_names,
                    mode=mode,
                    discretize_continuous=True
                )
                
                print("LIME explainer initialized successfully")
                
            except Exception as e:
                print(f"Failed to initialize LIME explainer: {str(e)}")
                self.lime_explainer = None
    
    def explain_global(self, method='all', max_display=20):
        """
        Generate global explanations (feature importance across all data)
        
        Args:
            method (str): 'shap', 'lime', 'permutation', or 'all'
            max_display (int): Maximum features to display
            
        Returns:
            dict: Global explanation results
        """
        results = {}
        
        print("Generating global explanations...")
        
        # SHAP global explanation
        if (method in ['shap', 'all'] and 
            SHAP_AVAILABLE and 
            self.shap_explainer is not None):
            
            try:
                print("Computing SHAP values...")
                # Use a subset for efficiency
                X_sample = shap.sample(self.X_train, min(500, len(self.X_train)))
                shap_values = self.shap_explainer.shap_values(X_sample)
                
                # Handle multi-output case
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                
                # Compute mean absolute SHAP values
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                # Create feature importance ranking
                feature_importance = dict(zip(self.feature_names, mean_abs_shap))
                sorted_features = sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                
                results['shap'] = {
                    'feature_importance': dict(sorted_features[:max_display]),
                    'shap_values': shap_values,
                    'sample_data': X_sample,
                    'method': 'SHAP'
                }
                
                print("✓ SHAP global explanation completed")
                
            except Exception as e:
                print(f"SHAP global explanation failed: {str(e)}")
        
        # Permutation importance
        if method in ['permutation', 'all']:
            try:
                print("Computing permutation importance...")
                
                if hasattr(self.model, 'model'):
                    model_obj = self.model.model
                else:
                    model_obj = self.model
                
                # Choose scoring metric
                if self.model.task_type == 'classification':
                    scoring = 'accuracy'
                else:
                    scoring = 'neg_mean_squared_error'
                
                # Use a subset for efficiency
                n_samples = min(1000, len(self.X_train))
                indices = np.random.choice(len(self.X_train), n_samples, replace=False)
                X_subset = self.X_train.iloc[indices] if hasattr(self.X_train, 'iloc') else self.X_train[indices]
                y_subset = self.y_train.iloc[indices] if hasattr(self.y_train, 'iloc') else self.y_train[indices]
                
                perm_importance = permutation_importance(
                    model_obj, X_subset, y_subset,
                    scoring=scoring, n_repeats=5, random_state=42
                )
                
                # Create feature importance ranking
                feature_importance = dict(zip(self.feature_names, perm_importance.importances_mean))
                sorted_features = sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                
                results['permutation'] = {
                    'feature_importance': dict(sorted_features[:max_display]),
                    'importances_mean': perm_importance.importances_mean,
                    'importances_std': perm_importance.importances_std,
                    'method': 'Permutation Importance'
                }
                
                print("✓ Permutation importance completed")
                
            except Exception as e:
                print(f"Permutation importance failed: {str(e)}")
        
        # Built-in feature importance (if available)
        if method in ['builtin', 'all']:
            try:
                importance = self.model.get_feature_importance()
                if importance is not None:
                    if isinstance(importance, dict):
                        sorted_features = sorted(importance.items(), 
                                               key=lambda x: x[1], reverse=True)
                    else:
                        feature_importance = dict(zip(self.feature_names, importance))
                        sorted_features = sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True)
                    
                    results['builtin'] = {
                        'feature_importance': dict(sorted_features[:max_display]),
                        'method': 'Built-in Importance'
                    }
                    
                    print("✓ Built-in feature importance completed")
                    
            except Exception as e:
                print(f"Built-in importance failed: {str(e)}")
        
        return results
    
    def explain_local(self, instance, method='all', num_features=10):
        """
        Generate local explanations for a single instance
        
        Args:
            instance: Single sample to explain (DataFrame row or array)
            method (str): 'shap', 'lime', or 'all'
            num_features (int): Number of features to include in explanation
            
        Returns:
            dict: Local explanation results
        """
        results = {}
        
        # Convert instance to appropriate format
        if hasattr(instance, 'values'):
            instance_array = instance.values.reshape(1, -1)
            instance_df = pd.DataFrame([instance.values], columns=self.feature_names)
        else:
            instance_array = np.array(instance).reshape(1, -1)
            instance_df = pd.DataFrame([instance], columns=self.feature_names)
        
        print("Generating local explanations...")
        
        # SHAP local explanation
        if (method in ['shap', 'all'] and 
            SHAP_AVAILABLE and 
            self.shap_explainer is not None):
            
            try:
                print("Computing SHAP values for instance...")
                shap_values = self.shap_explainer.shap_values(instance_array)
                
                # Handle multi-output case
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                
                shap_values = shap_values[0]  # Get values for the single instance
                
                # Create feature contributions
                feature_contributions = dict(zip(self.feature_names, shap_values))
                sorted_contributions = sorted(feature_contributions.items(), 
                                            key=lambda x: abs(x[1]), reverse=True)
                
                results['shap'] = {
                    'feature_contributions': dict(sorted_contributions[:num_features]),
                    'shap_values': shap_values,
                    'expected_value': self.shap_explainer.expected_value,
                    'prediction_explanation': self._create_text_explanation(
                        sorted_contributions[:num_features], 'SHAP'
                    ),
                    'method': 'SHAP'
                }
                
                print("✓ SHAP local explanation completed")
                
            except Exception as e:
                print(f"SHAP local explanation failed: {str(e)}")
        
        # LIME local explanation
        if (method in ['lime', 'all'] and 
            LIME_AVAILABLE and 
            self.lime_explainer is not None):
            
            try:
                print("Computing LIME explanation for instance...")
                
                # Define prediction function
                if hasattr(self.model, 'model'):
                    if self.model.task_type == 'classification':
                        predict_fn = self.model.model.predict_proba
                    else:
                        predict_fn = self.model.model.predict
                else:
                    if self.model.task_type == 'classification':
                        predict_fn = self.model.predict_proba
                    else:
                        predict_fn = self.model.predict
                
                # Generate LIME explanation
                lime_exp = self.lime_explainer.explain_instance(
                    instance_array[0], 
                    predict_fn,
                    num_features=num_features
                )
                
                # Extract feature contributions
                feature_contributions = dict(lime_exp.as_list())
                
                results['lime'] = {
                    'feature_contributions': feature_contributions,
                    'explanation_object': lime_exp,
                    'prediction_explanation': self._create_text_explanation(
                        list(feature_contributions.items()), 'LIME'
                    ),
                    'method': 'LIME'
                }
                
                print("✓ LIME local explanation completed")
                
            except Exception as e:
                print(f"LIME local explanation failed: {str(e)}")
        
        # Add prediction information
        try:
            prediction = self.model.predict(instance_df)[0]
            results['prediction_info'] = {
                'prediction': prediction,
                'instance_values': dict(zip(self.feature_names, instance_array[0]))
            }
            
            if self.model.task_type == 'classification':
                try:
                    probabilities = self.model.predict_proba(instance_df)[0]
                    results['prediction_info']['probabilities'] = probabilities
                except:
                    pass
                    
        except Exception as e:
            print(f"Failed to get prediction info: {str(e)}")
        
        return results
    
    def _create_text_explanation(self, feature_contributions, method):
        """Create human-readable text explanation"""
        if not feature_contributions:
            return "No significant feature contributions found."
        
        explanation = f"{method} Explanation:\n"
        explanation += "The prediction is primarily influenced by:\n"
        
        for i, (feature, contribution) in enumerate(feature_contributions[:5]):
            direction = "increases" if contribution > 0 else "decreases"
            explanation += f"{i+1}. {feature}: {direction} prediction by {abs(contribution):.4f}\n"
        
        return explanation
    
    def compare_explanations(self, instance, methods=['shap', 'lime']):
        """
        Compare explanations from different methods
        
        Args:
            instance: Single sample to explain
            methods: List of methods to compare
            
        Returns:
            dict: Comparison results
        """
        explanations = {}
        
        for method in methods:
            try:
                explanation = self.explain_local(instance, method=method)
                if method in explanation:
                    explanations[method] = explanation[method]['feature_contributions']
            except Exception as e:
                print(f"Failed to get {method} explanation: {str(e)}")
        
        # Compare feature rankings
        comparison = {}
        if len(explanations) >= 2:
            method_names = list(explanations.keys())
            
            for i in range(len(method_names)):
                for j in range(i+1, len(method_names)):
                    method1, method2 = method_names[i], method_names[j]
                    
                    # Get top features from each method
                    features1 = set(list(explanations[method1].keys())[:5])
                    features2 = set(list(explanations[method2].keys())[:5])
                    
                    # Calculate overlap
                    overlap = len(features1.intersection(features2))
                    total_unique = len(features1.union(features2))
                    
                    comparison[f"{method1}_vs_{method2}"] = {
                        'overlap_ratio': overlap / total_unique if total_unique > 0 else 0,
                        'common_features': list(features1.intersection(features2)),
                        'unique_to_method1': list(features1 - features2),
                        'unique_to_method2': list(features2 - features1)
                    }
        
        return {
            'individual_explanations': explanations,
            'comparison': comparison
        }
    
    def analyze_feature_interactions(self, features=None, sample_size=100):
        """
        Analyze feature interactions using SHAP (if available)
        
        Args:
            features: List of feature names to analyze (None for all)
            sample_size: Number of samples to use for analysis
            
        Returns:
            dict: Interaction analysis results
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return "SHAP not available for interaction analysis"
        
        try:
            print("Analyzing feature interactions...")
            
            # Sample data for efficiency
            X_sample = shap.sample(self.X_train, min(sample_size, len(self.X_train)))
            
            # Compute SHAP interaction values (if supported)
            if hasattr(self.shap_explainer, 'shap_interaction_values'):
                interaction_values = self.shap_explainer.shap_interaction_values(X_sample)
                
                # Compute interaction strengths
                n_features = len(self.feature_names)
                interaction_matrix = np.abs(interaction_values).mean(axis=0)
                
                # Find strongest interactions
                interactions = []
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        strength = interaction_matrix[i, j]
                        interactions.append({
                            'feature1': self.feature_names[i],
                            'feature2': self.feature_names[j],
                            'interaction_strength': strength
                        })
                
                # Sort by strength
                interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
                
                return {
                    'top_interactions': interactions[:10],
                    'interaction_matrix': interaction_matrix,
                    'feature_names': self.feature_names
                }
            else:
                return "Interaction analysis not supported for this model type"
                
        except Exception as e:
            return f"Interaction analysis failed: {str(e)}"
    
    def generate_explanation_report(self, X_test, num_samples=5):
        """
        Generate comprehensive explanation report
        
        Args:
            X_test: Test data
            num_samples: Number of individual samples to explain
            
        Returns:
            dict: Comprehensive explanation report
        """
        print("Generating comprehensive explanation report...")
        
        report = {
            'global_explanations': self.explain_global(),
            'sample_explanations': [],
            'feature_interactions': self.analyze_feature_interactions(),
            'summary': {}
        }
        
        # Explain individual samples
        sample_indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        
        for idx in sample_indices:
            sample = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
            explanation = self.explain_local(sample)
            explanation['sample_index'] = int(idx)
            report['sample_explanations'].append(explanation)
        
        # Create summary
        report['summary'] = {
            'model_type': self.model.model_type,
            'task_type': self.model.task_type,
            'num_features': len(self.feature_names),
            'explanation_methods': list(report['global_explanations'].keys()),
            'samples_explained': len(report['sample_explanations'])
        }
        
        print("Explanation report completed!")
        return report


def quick_explain(model, X_train, y_train, instance, feature_names=None):
    """
    Quick explanation function for single instances
    
    Args:
        model: Trained model
        X_train: Training data
        y_train: Training targets
        instance: Single sample to explain
        feature_names: Feature names
        
    Returns:
        dict: Quick explanation results
    """
    explainer = ModelExplainer(model, X_train, y_train, feature_names)
    return explainer.explain_local(instance)


if __name__ == "__main__":
    print("XplainML Explainability Module")
    
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
    
    # Create explainer
    explainer = ModelExplainer(model, data['X_train'], data['y_train'])
    
    # Global explanations
    global_exp = explainer.explain_global()
    print("Global explanations completed!")
    
    # Local explanation for first test sample
    sample = data['X_test'].iloc[0]
    local_exp = explainer.explain_local(sample)
    print("Local explanation completed!")
    
    print("Example explanation completed!")