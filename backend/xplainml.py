#!/usr/bin/env python3
"""
XplainML - Interpretable Machine Learning for Tabular Data
Main CLI interface for the XplainML tool

Usage:
    python xplainml.py --dataset data.csv --target target_column --model random_forest --explain shap --output html
"""

import os
import sys
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

# Add backend directory to path (we're already in backend)
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import XplainML modules
try:
    from data_preprocessing import DataPreprocessor, create_sample_data
    from models import MLModel, ModelTuner, compare_models
    from prediction import Predictor, create_prediction_report
    from explainer import ModelExplainer
    from visualizer import ModelVisualizer
except ImportError as e:
    print(f"Error importing XplainML modules: {e}")
    print("Make sure all required packages are installed.")
    sys.exit(1)

import pandas as pd
import numpy as np


class XplainMLCLI:
    """
    Main CLI class for XplainML
    """
    
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.explainer = None
        self.visualizer = None
        self.data = None
        
    def parse_arguments(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description='XplainML - Interpretable Machine Learning for Tabular Data',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic usage with Random Forest and SHAP explanations
  python xplainml.py --dataset data.csv --target target_col --model random_forest --explain shap
  
  # Compare multiple models
  python xplainml.py --dataset data.csv --target target_col --compare-models
  
  # Hyperparameter tuning
  python xplainml.py --dataset data.csv --target target_col --model xgboost --tune
  
  # Generate sample data
  python xplainml.py --generate-sample --samples 1000
  
  # Make predictions on new data
  python xplainml.py --predict new_data.csv --model-file trained_model.pkl
            """
        )
        
        # Data arguments
        parser.add_argument('--dataset', type=str, help='Path to dataset file (CSV or Excel)')
        parser.add_argument('--target', type=str, help='Target column name')
        parser.add_argument('--test-size', type=float, default=0.2, help='Test set proportion (default: 0.2)')
        
        # Model arguments
        parser.add_argument('--model', type=str, 
                          choices=['linear', 'random_forest', 'xgboost', 'neural_network'],
                          default='random_forest', help='Model type (default: random_forest)')
        parser.add_argument('--task-type', type=str, choices=['auto', 'classification', 'regression'],
                          default='auto', help='Task type (default: auto-detect)')
        
        # Training arguments
        parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
        parser.add_argument('--compare-models', action='store_true', help='Compare multiple models')
        parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds (default: 5)')
        
        # Preprocessing arguments
        parser.add_argument('--missing-strategy', type=str, 
                          choices=['mean', 'median', 'most_frequent'], default='mean',
                          help='Missing value imputation strategy (default: mean)')
        parser.add_argument('--encoding', type=str, choices=['auto', 'label', 'onehot'], 
                          default='auto', help='Categorical encoding (default: auto)')
        parser.add_argument('--scaling', type=str, choices=['standard', 'minmax', 'none'],
                          default='standard', help='Feature scaling (default: standard)')
        
        # Explanation arguments
        parser.add_argument('--explain', type=str, choices=['shap', 'lime', 'permutation', 'all'],
                          default='shap', help='Explanation method (default: shap)')
        parser.add_argument('--explain-samples', type=int, default=5,
                          help='Number of samples to explain individually (default: 5)')
        
        # Output arguments
        parser.add_argument('--output', type=str, choices=['text', 'html', 'json'],
                          default='text', help='Output format (default: text)')
        parser.add_argument('--output-dir', type=str, default='output',
                          help='Output directory (default: output)')
        parser.add_argument('--save-model', type=str, help='Save trained model to file')
        parser.add_argument('--save-plots', action='store_true', help='Save visualization plots')
        
        # Utility arguments
        parser.add_argument('--generate-sample', action='store_true', 
                          help='Generate sample dataset')
        parser.add_argument('--samples', type=int, default=1000,
                          help='Number of samples to generate (default: 1000)')
        parser.add_argument('--predict', type=str, help='Make predictions on new data file')
        parser.add_argument('--model-file', type=str, help='Load trained model from file')
        parser.add_argument('--verbose', action='store_true', help='Verbose output')
        
        return parser.parse_args()
    
    def generate_sample_data(self, n_samples=1000, filename='sample_data.csv'):
        """Generate sample dataset"""
        print(f"Generating sample dataset with {n_samples} samples...")
        
        # Ensure data directory exists (relative to project root)
        data_dir = os.path.join('..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        
        df = create_sample_data(filepath, n_samples)
        print(f"Sample data saved to {filepath}")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
        return filepath
    
    def load_and_preprocess_data(self, dataset_path, target_column, args):
        """Load and preprocess the dataset"""
        print("=== Data Loading and Preprocessing ===")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor()
        
        # Run preprocessing pipeline
        self.data = self.preprocessor.preprocess_pipeline(
            file_path=dataset_path,
            target_column=target_column,
            test_size=args.test_size,
            missing_strategy=args.missing_strategy,
            encoding_type=args.encoding,
            scaling_type=args.scaling
        )
        
        # Override task type if specified
        if args.task_type != 'auto':
            self.data['task_type'] = args.task_type
        
        print(f"✓ Data preprocessing completed")
        print(f"Task type: {self.data['task_type']}")
        print(f"Features: {len(self.data['feature_names'])}")
        print(f"Training samples: {self.data['X_train'].shape[0]}")
        print(f"Test samples: {self.data['X_test'].shape[0]}")
        
        return self.data
    
    def train_model(self, args):
        """Train the specified model"""
        print("=== Model Training ===")
        
        if args.compare_models:
            print("Comparing multiple models...")
            results = compare_models(
                self.data['X_train'], self.data['y_train'],
                self.data['X_test'], self.data['y_test'],
                task_type=self.data['task_type']
            )
            
            # Select best model based on performance
            best_model_type = None
            best_score = -np.inf
            
            for model_type, result in results.items():
                if 'test_metrics' in result:
                    if self.data['task_type'] == 'classification':
                        score = result['test_metrics'].get('accuracy', 0)
                    else:
                        score = result['test_metrics'].get('r2', -np.inf)
                    
                    if score > best_score:
                        best_score = score
                        best_model_type = model_type
                        self.model = result['model']
            
            print(f"Best model: {best_model_type} (score: {best_score:.4f})")
            return results
            
        else:
            # Train single model
            if args.tune:
                print(f"Training {args.model} with hyperparameter tuning...")
                tuner = ModelTuner(args.model, self.data['task_type'])
                self.model = tuner.tune(self.data['X_train'], self.data['y_train'], cv=args.cv_folds)
            else:
                print(f"Training {args.model} model...")
                self.model = MLModel(args.model, self.data['task_type'])
                self.model.fit(self.data['X_train'], self.data['y_train'])
            
            # Evaluate model
            train_metrics = self.model.evaluate(self.data['X_train'], self.data['y_train'])
            test_metrics = self.model.evaluate(self.data['X_test'], self.data['y_test'])
            
            print(f"✓ Model training completed")
            print(f"Training metrics: {train_metrics}")
            print(f"Test metrics: {test_metrics}")
            
            return {
                'model': self.model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
    
    def explain_model(self, args):
        """Generate model explanations"""
        print("=== Model Explanations ===")
        
        # Initialize explainer
        self.explainer = ModelExplainer(
            self.model, 
            self.data['X_train'], 
            self.data['y_train'],
            feature_names=self.data['feature_names']
        )
        
        # Generate explanations
        explanation_results = self.explainer.generate_explanation_report(
            self.data['X_test'],
            num_samples=args.explain_samples
        )
        
        print(f"✓ Model explanations completed")
        print(f"Methods used: {list(explanation_results['global_explanations'].keys())}")
        print(f"Samples explained: {len(explanation_results['sample_explanations'])}")
        
        return explanation_results
    
    def create_visualizations(self, explanation_results, args):
        """Create visualizations"""
        print("=== Creating Visualizations ===")
        
        # Initialize visualizer
        self.visualizer = ModelVisualizer(self.model, self.explainer)
        
        if args.save_plots:
            # Ensure output directory exists (relative to project root)
            output_dir = os.path.join('..', args.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save all plots
            self.visualizer.save_all_plots(explanation_results, output_dir)
            print(f"✓ Visualizations saved to {output_dir}")
        
        # Create interactive dashboard
        dashboard = self.visualizer.create_explanation_dashboard(explanation_results)
        
        if args.output == 'html':
            output_dir = os.path.join('..', args.output_dir)
            dashboard_path = os.path.join(output_dir, 'explanation_dashboard.html')
            os.makedirs(output_dir, exist_ok=True)
            dashboard.write_html(dashboard_path)
            print(f"✓ Interactive dashboard saved to {dashboard_path}")
        
        return dashboard
    
    def save_results(self, results, explanation_results, args):
        """Save results in specified format"""
        print("=== Saving Results ===")
        
        output_dir = os.path.join('..', args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare results summary
        summary = {
            'model_info': {
                'model_type': self.model.model_type,
                'task_type': self.data['task_type'],
                'features': self.data['feature_names'],
                'dataset_shape': self.data['original_shape']
            },
            'performance': results if isinstance(results, dict) else {},
            'explanations': {
                'global_methods': list(explanation_results['global_explanations'].keys()),
                'samples_explained': len(explanation_results['sample_explanations'])
            }
        }
        
        if args.output == 'json':
            # Save as JSON (excluding non-serializable objects)
            json_results = {
                'summary': summary,
                'feature_importance': {}
            }
            
            # Extract feature importance
            for method, data in explanation_results['global_explanations'].items():
                if 'feature_importance' in data:
                    json_results['feature_importance'][method] = data['feature_importance']
            
            json_path = os.path.join(args.output_dir, 'results.json')
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            print(f"✓ Results saved to {json_path}")
            
        elif args.output == 'text':
            # Save as text report
            text_path = os.path.join(args.output_dir, 'results.txt')
            with open(text_path, 'w') as f:
                f.write("XplainML Analysis Report\\n")
                f.write("=" * 50 + "\\n\\n")
                
                f.write(f"Model Type: {summary['model_info']['model_type']}\\n")
                f.write(f"Task Type: {summary['model_info']['task_type']}\\n")
                f.write(f"Number of Features: {len(summary['model_info']['features'])}\\n")
                f.write(f"Dataset Shape: {summary['model_info']['dataset_shape']}\\n\\n")
                
                # Performance metrics
                if 'test_metrics' in results:
                    f.write("Performance Metrics:\\n")
                    for metric, value in results['test_metrics'].items():
                        f.write(f"  {metric}: {value:.4f}\\n")
                    f.write("\\n")
                
                # Feature importance
                f.write("Feature Importance (Top 10):\\n")
                for method, data in explanation_results['global_explanations'].items():
                    if 'feature_importance' in data:
                        f.write(f"\\n{method.upper()} Method:\\n")
                        for i, (feature, importance) in enumerate(list(data['feature_importance'].items())[:10]):
                            f.write(f"  {i+1}. {feature}: {importance:.4f}\\n")
            
            print(f"✓ Text report saved to {text_path}")
        
        # Save model if requested
        if args.save_model:
            output_dir = os.path.join('..', args.output_dir)
            model_path = os.path.join(output_dir, args.save_model)
            self.model.save_model(model_path)
            print(f"✓ Model saved to {model_path}")
    
    def make_predictions(self, data_path, model_path):
        """Make predictions on new data"""
        print("=== Making Predictions ===")
        
        # Load model
        if model_path:
            self.model = MLModel.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        elif self.model is None:
            raise ValueError("No model available. Train a model first or provide --model-file")
        
        # Create predictor
        predictor = Predictor(self.model, self.preprocessor)
        
        # Make predictions
        predictions_df = predictor.predict_batch(data_path)
        
        # Save predictions
        output_path = data_path.replace('.csv', '_predictions.csv')
        predictions_df.to_csv(output_path, index=False)
        
        print(f"✓ Predictions saved to {output_path}")
        print(f"Predicted {len(predictions_df)} samples")
        
        return predictions_df
    
    def run(self):
        """Main execution function"""
        args = self.parse_arguments()
        
        try:
            # Generate sample data if requested
            if args.generate_sample:
                self.generate_sample_data(args.samples)
                return
            
            # Make predictions if requested
            if args.predict:
                self.make_predictions(args.predict, args.model_file)
                return
            
            # Validate required arguments
            if not args.dataset:
                print("Error: --dataset is required")
                return
            
            if not args.target and not args.predict:
                print("Error: --target is required")
                return
            
            print("XplainML - Interpretable Machine Learning")
            print("=" * 50)
            
            # Load and preprocess data
            self.load_and_preprocess_data(args.dataset, args.target, args)
            
            # Train model
            results = self.train_model(args)
            
            # Generate explanations
            explanation_results = self.explain_model(args)
            
            # Create visualizations
            self.create_visualizations(explanation_results, args)
            
            # Save results
            self.save_results(results, explanation_results, args)
            
            print("\\n" + "=" * 50)
            print("✓ XplainML analysis completed successfully!")
            print(f"Results saved to: {args.output_dir}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    cli = XplainMLCLI()
    cli.run()