"""
Visualization module for XplainML
Creates interactive and static visualizations for model explanations
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Matplotlib and Seaborn imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")

# Plotly imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

# SHAP imports for visualization
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ModelVisualizer:
    """
    Comprehensive visualization class for model explanations and data analysis
    """
    
    def __init__(self, model=None, explainer=None, figsize=(12, 8)):
        """
        Initialize visualizer
        
        Args:
            model: Trained ML model
            explainer: ModelExplainer instance
            figsize: Default figure size for matplotlib plots
        """
        self.model = model
        self.explainer = explainer
        self.figsize = figsize
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
    
    def plot_feature_importance(self, importance_dict, title="Feature Importance", 
                               max_features=20, plot_type='bar', interactive=True):
        """
        Plot feature importance
        
        Args:
            importance_dict: Dictionary of feature names and importance scores
            title: Plot title
            max_features: Maximum number of features to display
            plot_type: 'bar' or 'horizontal'
            interactive: Use Plotly for interactive plots
            
        Returns:
            Plot object or saves plot
        """
        if not importance_dict:
            print("No feature importance data available")
            return None
        
        # Prepare data
        features = list(importance_dict.keys())[:max_features]
        importances = list(importance_dict.values())[:max_features]
        
        if interactive and PLOTLY_AVAILABLE:
            # Create interactive Plotly plot
            if plot_type == 'horizontal':
                fig = go.Figure(go.Bar(
                    y=features[::-1],  # Reverse for better display
                    x=importances[::-1],
                    orientation='h',
                    marker_color=px.colors.qualitative.Set3
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    height=max(400, len(features) * 25),
                    showlegend=False
                )
            else:
                fig = go.Figure(go.Bar(
                    x=features,
                    y=importances,
                    marker_color=px.colors.qualitative.Set3
                ))
                fig.update_layout(
                    title=title,
                    xaxis_title="Features",
                    yaxis_title="Importance Score",
                    xaxis_tickangle=45
                )
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            # Create static matplotlib plot
            plt.figure(figsize=self.figsize)
            
            if plot_type == 'horizontal':
                plt.barh(range(len(features)), importances, color=self.color_palette[0])
                plt.yticks(range(len(features)), features)
                plt.xlabel("Importance Score")
                plt.ylabel("Features")
            else:
                plt.bar(range(len(features)), importances, color=self.color_palette[0])
                plt.xticks(range(len(features)), features, rotation=45, ha='right')
                plt.xlabel("Features")
                plt.ylabel("Importance Score")
            
            plt.title(title)
            plt.tight_layout()
            plt.show()
            
        else:
            print("No visualization library available")
            return None
    
    def plot_shap_summary(self, shap_values, feature_data, feature_names=None, 
                         plot_type='summary', interactive=True):
        """
        Plot SHAP summary plots
        
        Args:
            shap_values: SHAP values array
            feature_data: Feature data array
            feature_names: List of feature names
            plot_type: 'summary', 'bar', or 'waterfall'
            interactive: Use interactive plots if available
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available for summary plots")
            return None
        
        try:
            if plot_type == 'summary':
                shap.summary_plot(shap_values, feature_data, 
                                feature_names=feature_names, show=True)
            elif plot_type == 'bar':
                shap.summary_plot(shap_values, feature_data, 
                                feature_names=feature_names, plot_type='bar', show=True)
            elif plot_type == 'waterfall':
                # For single instance waterfall plot
                if len(shap_values.shape) > 1:
                    instance_idx = 0  # Use first instance
                    shap.waterfall_plot(
                        shap.Explanation(values=shap_values[instance_idx], 
                                       base_values=0, 
                                       data=feature_data[instance_idx],
                                       feature_names=feature_names)
                    )
                else:
                    shap.waterfall_plot(
                        shap.Explanation(values=shap_values, 
                                       base_values=0, 
                                       data=feature_data,
                                       feature_names=feature_names)
                    )
                    
        except Exception as e:
            print(f"SHAP plot failed: {str(e)}")
    
    def plot_partial_dependence(self, model, X_data, feature_name, 
                               num_points=50, interactive=True):
        """
        Plot partial dependence plot for a feature
        
        Args:
            model: Trained model
            X_data: Input data
            feature_name: Name of the feature
            num_points: Number of points to evaluate
            interactive: Use interactive plots
        """
        if feature_name not in X_data.columns:
            print(f"Feature {feature_name} not found in data")
            return None
        
        # Get feature values range
        feature_values = X_data[feature_name]
        min_val, max_val = feature_values.min(), feature_values.max()
        
        # Create range of values to evaluate
        if feature_values.dtype in ['int64', 'float64']:
            eval_values = np.linspace(min_val, max_val, num_points)
        else:
            # For categorical features
            eval_values = feature_values.unique()[:num_points]
        
        # Calculate partial dependence
        partial_deps = []
        base_instance = X_data.iloc[0].copy()
        
        for val in eval_values:
            base_instance[feature_name] = val
            pred = model.predict(pd.DataFrame([base_instance]))[0]
            partial_deps.append(pred)
        
        # Create plot
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eval_values,
                y=partial_deps,
                mode='lines+markers',
                name=f'Partial Dependence',
                line=dict(color=self.color_palette[0], width=2)
            ))
            
            fig.update_layout(
                title=f'Partial Dependence Plot: {feature_name}',
                xaxis_title=feature_name,
                yaxis_title='Prediction',
                showlegend=False
            )
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=self.figsize)
            plt.plot(eval_values, partial_deps, 'o-', color=self.color_palette[0], linewidth=2)
            plt.xlabel(feature_name)
            plt.ylabel('Prediction')
            plt.title(f'Partial Dependence Plot: {feature_name}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def plot_prediction_distribution(self, predictions, true_values=None, 
                                   task_type='classification', interactive=True):
        """
        Plot distribution of predictions
        
        Args:
            predictions: Model predictions
            true_values: True target values (optional)
            task_type: 'classification' or 'regression'
            interactive: Use interactive plots
        """
        if interactive and PLOTLY_AVAILABLE:
            if task_type == 'classification':
                # Classification: bar plot of class distribution
                unique, counts = np.unique(predictions, return_counts=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[str(x) for x in unique],
                    y=counts,
                    name='Predictions',
                    marker_color=self.color_palette[0]
                ))
                
                if true_values is not None:
                    true_unique, true_counts = np.unique(true_values, return_counts=True)
                    fig.add_trace(go.Bar(
                        x=[str(x) for x in true_unique],
                        y=true_counts,
                        name='True Values',
                        marker_color=self.color_palette[1]
                    ))
                
                fig.update_layout(
                    title='Prediction Distribution',
                    xaxis_title='Class',
                    yaxis_title='Count',
                    barmode='group'
                )
                
            else:
                # Regression: histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=predictions,
                    name='Predictions',
                    marker_color=self.color_palette[0],
                    opacity=0.7
                ))
                
                if true_values is not None:
                    fig.add_trace(go.Histogram(
                        x=true_values,
                        name='True Values',
                        marker_color=self.color_palette[1],
                        opacity=0.7
                    ))
                
                fig.update_layout(
                    title='Prediction Distribution',
                    xaxis_title='Value',
                    yaxis_title='Frequency',
                    barmode='overlay'
                )
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=self.figsize)
            
            if task_type == 'classification':
                unique, counts = np.unique(predictions, return_counts=True)
                plt.bar(range(len(unique)), counts, alpha=0.7, 
                       color=self.color_palette[0], label='Predictions')
                
                if true_values is not None:
                    true_unique, true_counts = np.unique(true_values, return_counts=True)
                    plt.bar(range(len(true_unique)), true_counts, alpha=0.7,
                           color=self.color_palette[1], label='True Values')
                
                plt.xticks(range(len(unique)), unique)
                plt.xlabel('Class')
                plt.ylabel('Count')
                
            else:
                plt.hist(predictions, alpha=0.7, color=self.color_palette[0], 
                        label='Predictions', bins=30)
                
                if true_values is not None:
                    plt.hist(true_values, alpha=0.7, color=self.color_palette[1],
                            label='True Values', bins=30)
                
                plt.xlabel('Value')
                plt.ylabel('Frequency')
            
            plt.title('Prediction Distribution')
            if true_values is not None:
                plt.legend()
            plt.tight_layout()
            plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, interactive=True):
        """
        Plot confusion matrix for classification
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            interactive: Use interactive plots
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [str(i) for i in range(len(cm))]
        
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12}
            ))
            
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted Label',
                yaxis_title='True Label'
            )
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=self.figsize)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.show()
    
    def plot_learning_curves(self, train_sizes, train_scores, val_scores, interactive=True):
        """
        Plot learning curves
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores
            val_scores: Validation scores
            interactive: Use interactive plots
        """
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=np.mean(train_scores, axis=1),
                mode='lines+markers',
                name='Training Score',
                line=dict(color=self.color_palette[0]),
                error_y=dict(
                    type='data',
                    array=np.std(train_scores, axis=1),
                    visible=True
                )
            ))
            
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=np.mean(val_scores, axis=1),
                mode='lines+markers',
                name='Validation Score',
                line=dict(color=self.color_palette[1]),
                error_y=dict(
                    type='data',
                    array=np.std(val_scores, axis=1),
                    visible=True
                )
            ))
            
            fig.update_layout(
                title='Learning Curves',
                xaxis_title='Training Set Size',
                yaxis_title='Score'
            )
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=self.figsize)
            
            plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-',
                    color=self.color_palette[0], label='Training Score')
            plt.fill_between(train_sizes, 
                           np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                           np.mean(train_scores, axis=1) + np.std(train_scores, axis=1),
                           alpha=0.1, color=self.color_palette[0])
            
            plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-',
                    color=self.color_palette[1], label='Validation Score')
            plt.fill_between(train_sizes,
                           np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                           np.mean(val_scores, axis=1) + np.std(val_scores, axis=1),
                           alpha=0.1, color=self.color_palette[1])
            
            plt.xlabel('Training Set Size')
            plt.ylabel('Score')
            plt.title('Learning Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def plot_residuals(self, y_true, y_pred, interactive=True):
        """
        Plot residuals for regression tasks
        
        Args:
            y_true: True values
            y_pred: Predicted values
            interactive: Use interactive plots
        """
        residuals = y_true - y_pred
        
        if interactive and PLOTLY_AVAILABLE:
            fig = make_subplots(rows=1, cols=2,
                              subplot_titles=['Residuals vs Predicted', 'Residuals Histogram'])
            
            # Residuals vs Predicted
            fig.add_trace(go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                marker=dict(color=self.color_palette[0]),
                name='Residuals'
            ), row=1, col=1)
            
            # Add horizontal line at y=0
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Residuals histogram
            fig.add_trace(go.Histogram(
                x=residuals,
                marker_color=self.color_palette[1],
                name='Residuals Distribution'
            ), row=1, col=2)
            
            fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
            fig.update_yaxes(title_text="Residuals", row=1, col=1)
            fig.update_xaxes(title_text="Residuals", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)
            
            fig.update_layout(title_text="Residual Analysis", showlegend=False)
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Residuals vs Predicted
            ax1.scatter(y_pred, residuals, alpha=0.6, color=self.color_palette[0])
            ax1.axhline(y=0, color='red', linestyle='--')
            ax1.set_xlabel('Predicted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residuals vs Predicted')
            ax1.grid(True, alpha=0.3)
            
            # Residuals histogram
            ax2.hist(residuals, bins=30, alpha=0.7, color=self.color_palette[1])
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Residuals Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def create_explanation_dashboard(self, explanation_results, save_html=False, filename=None):
        """
        Create an interactive dashboard with all explanations
        
        Args:
            explanation_results: Results from ModelExplainer
            save_html: Save as HTML file
            filename: HTML filename
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available for dashboard creation")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Global Feature Importance', 'Feature Importance Comparison',
                'Sample Explanation 1', 'Sample Explanation 2',
                'Prediction Distribution', 'Model Performance'
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Add global feature importance
        if 'global_explanations' in explanation_results:
            global_exp = explanation_results['global_explanations']
            if 'shap' in global_exp:
                importance = global_exp['shap']['feature_importance']
                features = list(importance.keys())[:10]
                values = list(importance.values())[:10]
                
                fig.add_trace(go.Bar(
                    x=features,
                    y=values,
                    name='SHAP Importance',
                    marker_color=self.color_palette[0]
                ), row=1, col=1)
        
        # Add sample explanations
        if 'sample_explanations' in explanation_results:
            samples = explanation_results['sample_explanations'][:2]
            for i, sample in enumerate(samples):
                if 'shap' in sample:
                    contributions = sample['shap']['feature_contributions']
                    features = list(contributions.keys())[:5]
                    values = list(contributions.values())[:5]
                    
                    fig.add_trace(go.Bar(
                        x=features,
                        y=values,
                        name=f'Sample {i+1}',
                        marker_color=self.color_palette[i+1]
                    ), row=2, col=i+1)
        
        fig.update_layout(
            title_text="XplainML Explanation Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_html and filename:
            fig.write_html(filename)
            print(f"Dashboard saved as {filename}")
        
        return fig
    
    def save_all_plots(self, explanation_results, output_dir='plots'):
        """
        Save all visualization plots to files
        
        Args:
            explanation_results: Results from ModelExplainer
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving plots to {output_dir}...")
        
        # Global feature importance plots
        if 'global_explanations' in explanation_results:
            global_exp = explanation_results['global_explanations']
            
            for method, data in global_exp.items():
                if 'feature_importance' in data:
                    fig = self.plot_feature_importance(
                        data['feature_importance'],
                        title=f'Global Feature Importance ({method})',
                        interactive=True
                    )
                    if fig:
                        fig.write_html(f"{output_dir}/global_importance_{method}.html")
        
        # Create dashboard
        dashboard = self.create_explanation_dashboard(explanation_results)
        if dashboard:
            dashboard.write_html(f"{output_dir}/explanation_dashboard.html")
        
        print("All plots saved successfully!")


def quick_visualize(model, X_data, y_data=None, feature_name=None):
    """
    Quick visualization function
    
    Args:
        model: Trained model
        X_data: Feature data
        y_data: Target data (optional)
        feature_name: Specific feature to analyze
    """
    visualizer = ModelVisualizer(model)
    
    # Feature importance
    importance = model.get_feature_importance()
    if importance:
        visualizer.plot_feature_importance(importance)
    
    # Partial dependence for specific feature
    if feature_name and feature_name in X_data.columns:
        visualizer.plot_partial_dependence(model, X_data, feature_name)
    
    # Predictions distribution
    predictions = model.predict(X_data)
    visualizer.plot_prediction_distribution(predictions, y_data, model.task_type)


if __name__ == "__main__":
    print("XplainML Visualization Module")
    
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
    
    # Create visualizer
    visualizer = ModelVisualizer(model)
    
    # Plot feature importance
    importance = model.get_feature_importance()
    if importance:
        fig = visualizer.plot_feature_importance(importance, interactive=False)
    
    print("Example visualization completed!")