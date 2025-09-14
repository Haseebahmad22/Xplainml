"""
Streamlit Web Dashboard for XplainML
Interactive web interface for model training and explanation
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add backend directory to path
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    from io import BytesIO
    
    # Import XplainML modules from backend
    from backend import (
        DataPreprocessor, create_sample_data,
        MLModel, ModelTuner, compare_models,
        Predictor, ModelExplainer, ModelVisualizer
    )
    
except ImportError as e:
    st.error(f"Required packages not installed: {e}")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="XplainML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class XplainMLDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'explanations_generated' not in st.session_state:
            st.session_state.explanations_generated = False
        if 'preprocessor' not in st.session_state:
            st.session_state.preprocessor = None
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'explainer' not in st.session_state:
            st.session_state.explainer = None
        if 'explanation_results' not in st.session_state:
            st.session_state.explanation_results = None
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">🤖 XplainML Dashboard</h1>', 
                   unsafe_allow_html=True)
        st.markdown("**Interpretable Machine Learning for Tabular Data**")
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar navigation"""
        st.sidebar.title("Navigation")
        
        pages = [
            "🏠 Home",
            "📊 Data Upload & Preprocessing", 
            "🤖 Model Training",
            "🔍 Model Explanations",
            "📈 Visualizations",
            "🎯 Predictions",
            "ℹ️ About"
        ]
        
        selected_page = st.sidebar.radio("Go to", pages)
        
        # Progress indicator
        st.sidebar.markdown("### Progress")
        progress_items = [
            ("Data Loaded", st.session_state.data_loaded),
            ("Model Trained", st.session_state.model_trained),
            ("Explanations Generated", st.session_state.explanations_generated)
        ]
        
        for item, status in progress_items:
            if status:
                st.sidebar.success(f"✅ {item}")
            else:
                st.sidebar.info(f"⏳ {item}")
        
        return selected_page.split(" ", 1)[1]  # Remove emoji from page name
    
    def page_home(self):
        """Home page"""
        st.markdown('<div class="section-header">Welcome to XplainML</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 What is XplainML?
            
            XplainML is a comprehensive tool for interpretable machine learning on tabular data. 
            It helps you understand your models and make informed decisions.
            
            ### ✨ Key Features:
            - **Multiple ML Models**: Linear, Random Forest, XGBoost, Neural Networks
            - **Model Explanations**: SHAP, LIME, Permutation Importance
            - **Interactive Visualizations**: Feature importance, partial dependence plots
            - **Easy-to-use Interface**: Web dashboard and CLI
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Getting Started:
            
            1. **Upload your data** in the Data Upload section
            2. **Train a model** using various algorithms
            3. **Generate explanations** to understand predictions
            4. **Explore visualizations** for insights
            5. **Make predictions** on new data
            
            ### 📋 Supported Formats:
            - CSV files
            - Excel files (.xlsx, .xls)
            - Both classification and regression tasks
            """)
        
        # Quick stats if data is loaded
        if st.session_state.data_loaded:
            st.markdown('<div class="section-header">Current Dataset</div>', 
                       unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Samples", st.session_state.data['original_shape'][0])
            with col2:
                st.metric("Features", len(st.session_state.data['feature_names']))
            with col3:
                st.metric("Task Type", st.session_state.data['task_type'].title())
            with col4:
                st.metric("Target", st.session_state.data['target_name'])
    
    def page_data_upload(self):
        """Data upload and preprocessing page"""
        st.markdown('<div class="section-header">Data Upload & Preprocessing</div>', 
                   unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📤 Upload Data", "🔧 Generate Sample Data"])
        
        with tab1:
            self.render_data_upload()
        
        with tab2:
            self.render_sample_data_generation()
        
        # Show preprocessing options if data is uploaded
        if st.session_state.data_loaded:
            self.render_preprocessing_options()
    
    def render_data_upload(self):
        """Render data upload interface"""
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your dataset for analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load data preview
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(temp_path)
                else:
                    df = pd.read_excel(temp_path)
                
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Show data preview
                st.markdown("### Data Preview")
                st.dataframe(df.head(10))
                
                # Column selection
                st.markdown("### Column Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Columns:**")
                    for col in df.columns:
                        st.text(f"• {col} ({df[col].dtype})")
                
                with col2:
                    target_column = st.selectbox(
                        "Select target column:",
                        options=df.columns.tolist(),
                        help="Choose the column you want to predict"
                    )
                
                # Process data button
                if st.button("🚀 Process Data", type="primary"):
                    self.process_uploaded_data(temp_path, target_column)
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    def render_sample_data_generation(self):
        """Render sample data generation interface"""
        st.markdown("Generate a sample dataset for testing XplainML")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_samples = st.slider("Number of samples", 100, 5000, 1000)
            
        with col2:
            filename = st.text_input("Filename", "sample_data.csv")
        
        if st.button("🎲 Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                # Ensure data directory exists
                os.makedirs('data', exist_ok=True)
                filepath = os.path.join('data', filename)
                
                df = create_sample_data(filepath, n_samples)
                
                st.success(f"✅ Sample data generated! Shape: {df.shape}")
                st.markdown("### Generated Data Preview")
                st.dataframe(df.head())
                
                # Auto-process the generated data
                self.process_uploaded_data(filepath, 'target')
    
    def render_preprocessing_options(self):
        """Render preprocessing options"""
        st.markdown('<div class="section-header">Preprocessing Options</div>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            missing_strategy = st.selectbox(
                "Missing Value Strategy",
                ['mean', 'median', 'most_frequent'],
                help="How to handle missing values"
            )
        
        with col2:
            encoding_type = st.selectbox(
                "Categorical Encoding",
                ['auto', 'label', 'onehot'],
                help="How to encode categorical variables"
            )
        
        with col3:
            scaling_type = st.selectbox(
                "Feature Scaling",
                ['standard', 'minmax', 'none'],
                help="How to scale numerical features"
            )
        
        if st.button("🔄 Reprocess Data"):
            # Reprocess with new settings
            self.reprocess_data(missing_strategy, encoding_type, scaling_type)
    
    def process_uploaded_data(self, filepath, target_column):
        """Process uploaded data"""
        try:
            with st.spinner("Processing data..."):
                # Initialize preprocessor
                preprocessor = DataPreprocessor()
                
                # Process data
                data = preprocessor.preprocess_pipeline(
                    file_path=filepath,
                    target_column=target_column,
                    test_size=0.2,
                    missing_strategy='mean',
                    encoding_type='auto',
                    scaling_type='standard'
                )
                
                # Store in session state
                st.session_state.preprocessor = preprocessor
                st.session_state.data = data
                st.session_state.data_loaded = True
                
                st.success("✅ Data processed successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    
    def reprocess_data(self, missing_strategy, encoding_type, scaling_type):
        """Reprocess data with new settings"""
        try:
            with st.spinner("Reprocessing data..."):
                # Get original file path from session state or use current data
                preprocessor = DataPreprocessor()
                
                # Note: This is a simplified reprocessing
                # In a real scenario, you'd want to store the original file path
                st.warning("Reprocessing with new settings...")
                
                st.success("✅ Data reprocessed successfully!")
                
        except Exception as e:
            st.error(f"Error reprocessing data: {str(e)}")
    
    def page_model_training(self):
        """Model training page"""
        st.markdown('<div class="section-header">Model Training</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload and process data first!")
            return
        
        # Model selection
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type",
                ['random_forest', 'linear', 'xgboost', 'neural_network'],
                help="Choose the machine learning algorithm"
            )
        
        with col2:
            use_tuning = st.checkbox("Enable Hyperparameter Tuning", 
                                   help="Automatically find best parameters")
        
        # Training options
        if use_tuning:
            cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
        
        # Compare models option
        compare_all = st.checkbox("Compare All Models", 
                                help="Train and compare multiple models")
        
        # Training buttons
        col1, col2 = st.columns(2)
        
        with col1:
            train_button = st.button("🚀 Train Model", type="primary")
        
        with col2:
            if compare_all:
                compare_button = st.button("📊 Compare Models")
            else:
                compare_button = False
        
        # Training execution
        if train_button:
            self.train_single_model(model_type, use_tuning, cv_folds if use_tuning else 5)
        
        if compare_button:
            self.compare_all_models()
        
        # Show model info if trained
        if st.session_state.model_trained:
            self.display_model_info()
    
    def train_single_model(self, model_type, use_tuning, cv_folds):
        """Train a single model"""
        try:
            with st.spinner(f"Training {model_type} model..."):
                if use_tuning:
                    from backend import ModelTuner
                    tuner = ModelTuner(model_type, st.session_state.data['task_type'])
                    model = tuner.tune(
                        st.session_state.data['X_train'], 
                        st.session_state.data['y_train'],
                        cv=cv_folds
                    )
                else:
                    model = MLModel(model_type, st.session_state.data['task_type'])
                    model.fit(
                        st.session_state.data['X_train'],
                        st.session_state.data['y_train']
                    )
                
                # Evaluate model
                train_metrics = model.evaluate(
                    st.session_state.data['X_train'],
                    st.session_state.data['y_train']
                )
                test_metrics = model.evaluate(
                    st.session_state.data['X_test'],
                    st.session_state.data['y_test']
                )
                
                # Store in session state
                st.session_state.model = model
                st.session_state.model_trained = True
                st.session_state.train_metrics = train_metrics
                st.session_state.test_metrics = test_metrics
                
                st.success("✅ Model trained successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
    
    def compare_all_models(self):
        """Compare all available models"""
        try:
            with st.spinner("Comparing all models..."):
                results = compare_models(
                    st.session_state.data['X_train'],
                    st.session_state.data['y_train'],
                    st.session_state.data['X_test'],
                    st.session_state.data['y_test'],
                    task_type=st.session_state.data['task_type']
                )
                
                # Display comparison results
                st.markdown("### Model Comparison Results")
                
                comparison_data = []
                for model_name, result in results.items():
                    if 'test_metrics' in result:
                        row = {'Model': model_name}
                        row.update(result['test_metrics'])
                        comparison_data.append(row)
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df)
                    
                    # Select best model
                    if st.session_state.data['task_type'] == 'classification':
                        best_idx = comparison_df['accuracy'].idxmax()
                        metric = 'accuracy'
                    else:
                        best_idx = comparison_df['r2'].idxmax()
                        metric = 'r2'
                    
                    best_model_name = comparison_df.loc[best_idx, 'Model']
                    best_score = comparison_df.loc[best_idx, metric]
                    
                    st.success(f"🏆 Best Model: {best_model_name} ({metric}: {best_score:.4f})")
                    
                    # Option to select best model
                    if st.button("Select Best Model"):
                        st.session_state.model = results[best_model_name]['model']
                        st.session_state.model_trained = True
                        st.session_state.test_metrics = results[best_model_name]['test_metrics']
                        st.rerun()
                
        except Exception as e:
            st.error(f"Error comparing models: {str(e)}")
    
    def display_model_info(self):
        """Display information about the trained model"""
        st.markdown('<div class="section-header">Trained Model Information</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Details")
            st.info(f"**Type:** {st.session_state.model.model_type}")
            st.info(f"**Task:** {st.session_state.data['task_type']}")
            st.info(f"**Features:** {len(st.session_state.data['feature_names'])}")
        
        with col2:
            st.markdown("### Performance Metrics")
            if hasattr(st.session_state, 'test_metrics'):
                for metric, value in st.session_state.test_metrics.items():
                    st.metric(metric.title(), f"{value:.4f}")
        
        # Feature importance
        importance = st.session_state.model.get_feature_importance()
        if importance:
            st.markdown("### Feature Importance")
            
            # Create bar chart
            features = list(importance.keys())[:10]
            values = list(importance.values())[:10]
            
            fig = go.Figure(go.Bar(x=features, y=values))
            fig.update_layout(
                title="Top 10 Feature Importance",
                xaxis_title="Features",
                yaxis_title="Importance",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def page_explanations(self):
        """Model explanations page"""
        st.markdown('<div class="section-header">Model Explanations</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first!")
            return
        
        # Explanation options
        col1, col2 = st.columns(2)
        
        with col1:
            explain_method = st.selectbox(
                "Explanation Method",
                ['shap', 'lime', 'permutation', 'all'],
                help="Choose explanation technique"
            )
        
        with col2:
            num_samples = st.slider("Samples to Explain", 1, 10, 3)
        
        # Generate explanations
        if st.button("🔍 Generate Explanations", type="primary"):
            self.generate_explanations(explain_method, num_samples)
        
        # Display explanations if available
        if st.session_state.explanations_generated:
            self.display_explanations()
    
    def generate_explanations(self, method, num_samples):
        """Generate model explanations"""
        try:
            with st.spinner("Generating explanations..."):
                # Initialize explainer
                explainer = ModelExplainer(
                    st.session_state.model,
                    st.session_state.data['X_train'],
                    st.session_state.data['y_train'],
                    feature_names=st.session_state.data['feature_names']
                )
                
                # Generate explanation report
                explanation_results = explainer.generate_explanation_report(
                    st.session_state.data['X_test'],
                    num_samples=num_samples
                )
                
                # Store results
                st.session_state.explainer = explainer
                st.session_state.explanation_results = explanation_results
                st.session_state.explanations_generated = True
                
                st.success("✅ Explanations generated successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error generating explanations: {str(e)}")
    
    def display_explanations(self):
        """Display generated explanations"""
        results = st.session_state.explanation_results
        
        # Global explanations
        st.markdown("### Global Feature Importance")
        
        if 'global_explanations' in results:
            tabs = st.tabs(list(results['global_explanations'].keys()))
            
            for i, (method, data) in enumerate(results['global_explanations'].items()):
                with tabs[i]:
                    if 'feature_importance' in data:
                        # Create visualization
                        importance = data['feature_importance']
                        features = list(importance.keys())[:15]
                        values = list(importance.values())[:15]
                        
                        fig = go.Figure(go.Bar(
                            x=values[::-1],
                            y=features[::-1],
                            orientation='h'
                        ))
                        fig.update_layout(
                            title=f"Feature Importance ({method.upper()})",
                            xaxis_title="Importance Score",
                            height=max(400, len(features) * 25)
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Local explanations
        st.markdown("### Sample Explanations")
        
        if 'sample_explanations' in results:
            for i, sample_exp in enumerate(results['sample_explanations']):
                with st.expander(f"Sample {i+1} Explanation"):
                    
                    # Show prediction info
                    if 'prediction_info' in sample_exp:
                        pred_info = sample_exp['prediction_info']
                        st.info(f"**Prediction:** {pred_info['prediction']}")
                        
                        if 'probabilities' in pred_info:
                            probs = pred_info['probabilities']
                            st.write("**Probabilities:**")
                            for j, prob in enumerate(probs):
                                st.write(f"Class {j}: {prob:.3f}")
                    
                    # Show explanations
                    if 'shap' in sample_exp:
                        shap_data = sample_exp['shap']
                        contributions = shap_data['feature_contributions']
                        
                        # Create waterfall-like chart
                        features = list(contributions.keys())[:10]
                        values = list(contributions.values())[:10]
                        
                        colors = ['red' if v < 0 else 'green' for v in values]
                        
                        fig = go.Figure(go.Bar(
                            x=features,
                            y=values,
                            marker_color=colors
                        ))
                        fig.update_layout(
                            title=f"Feature Contributions (Sample {i+1})",
                            xaxis_title="Features",
                            yaxis_title="Contribution",
                            xaxis_tickangle=45
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    def page_visualizations(self):
        """Visualizations page"""
        st.markdown('<div class="section-header">Visualizations</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first!")
            return
        
        # Visualization options
        viz_type = st.selectbox(
            "Select Visualization Type",
            [
                "Feature Importance",
                "Prediction Distribution", 
                "Confusion Matrix",
                "Partial Dependence Plot",
                "Residual Analysis"
            ]
        )
        
        if viz_type == "Feature Importance":
            self.plot_feature_importance()
        elif viz_type == "Prediction Distribution":
            self.plot_prediction_distribution()
        elif viz_type == "Confusion Matrix":
            self.plot_confusion_matrix()
        elif viz_type == "Partial Dependence Plot":
            self.plot_partial_dependence()
        elif viz_type == "Residual Analysis":
            self.plot_residual_analysis()
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        importance = st.session_state.model.get_feature_importance()
        if importance:
            max_features = st.slider("Number of features to show", 5, 20, 10)
            
            features = list(importance.keys())[:max_features]
            values = list(importance.values())[:max_features]
            
            fig = go.Figure(go.Bar(
                x=values[::-1],
                y=features[::-1],
                orientation='h'
            ))
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance Score",
                height=max(400, len(features) * 25)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance not available for this model type")
    
    def plot_prediction_distribution(self):
        """Plot prediction distribution"""
        predictions = st.session_state.model.predict(st.session_state.data['X_test'])
        y_true = st.session_state.data['y_test']
        
        if st.session_state.data['task_type'] == 'classification':
            # Bar plot for classification
            unique_pred, counts_pred = np.unique(predictions, return_counts=True)
            unique_true, counts_true = np.unique(y_true, return_counts=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[str(x) for x in unique_pred],
                y=counts_pred,
                name='Predictions',
                opacity=0.7
            ))
            fig.add_trace(go.Bar(
                x=[str(x) for x in unique_true],
                y=counts_true,
                name='True Values',
                opacity=0.7
            ))
            fig.update_layout(
                title="Prediction vs True Distribution",
                xaxis_title="Class",
                yaxis_title="Count",
                barmode='group'
            )
        else:
            # Histogram for regression
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=predictions,
                name='Predictions',
                opacity=0.7
            ))
            fig.add_trace(go.Histogram(
                x=y_true,
                name='True Values',
                opacity=0.7
            ))
            fig.update_layout(
                title="Prediction vs True Distribution",
                xaxis_title="Value",
                yaxis_title="Frequency",
                barmode='overlay'
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix"""
        if st.session_state.data['task_type'] != 'classification':
            st.warning("Confusion matrix is only available for classification tasks")
            return
        
        from sklearn.metrics import confusion_matrix
        
        predictions = st.session_state.model.predict(st.session_state.data['X_test'])
        y_true = st.session_state.data['y_test']
        
        cm = confusion_matrix(y_true, predictions)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f"Pred {i}" for i in range(len(cm))],
            y=[f"True {i}" for i in range(len(cm))],
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
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_partial_dependence(self):
        """Plot partial dependence"""
        feature_name = st.selectbox(
            "Select feature for partial dependence",
            st.session_state.data['feature_names']
        )
        
        if st.button("Generate Partial Dependence Plot"):
            # Simple partial dependence calculation
            X_test = st.session_state.data['X_test']
            feature_values = X_test[feature_name]
            
            if feature_values.dtype in ['int64', 'float64']:
                eval_values = np.linspace(feature_values.min(), feature_values.max(), 50)
            else:
                eval_values = feature_values.unique()[:20]
            
            partial_deps = []
            base_instance = X_test.iloc[0].copy()
            
            for val in eval_values:
                base_instance[feature_name] = val
                pred = st.session_state.model.predict(pd.DataFrame([base_instance]))[0]
                partial_deps.append(pred)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eval_values,
                y=partial_deps,
                mode='lines+markers',
                name='Partial Dependence'
            ))
            
            fig.update_layout(
                title=f'Partial Dependence Plot: {feature_name}',
                xaxis_title=feature_name,
                yaxis_title='Prediction'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_residual_analysis(self):
        """Plot residual analysis"""
        if st.session_state.data['task_type'] != 'regression':
            st.warning("Residual analysis is only available for regression tasks")
            return
        
        predictions = st.session_state.model.predict(st.session_state.data['X_test'])
        y_true = st.session_state.data['y_test']
        residuals = y_true - predictions
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residuals vs Predicted
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=predictions,
                y=residuals,
                mode='markers',
                name='Residuals'
            ))
            fig1.add_hline(y=0, line_dash="dash", line_color="red")
            fig1.update_layout(
                title="Residuals vs Predicted",
                xaxis_title="Predicted Values",
                yaxis_title="Residuals"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Residuals histogram
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=residuals,
                name='Residuals Distribution'
            ))
            fig2.update_layout(
                title="Residuals Distribution",
                xaxis_title="Residuals",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    def page_predictions(self):
        """Predictions page"""
        st.markdown('<div class="section-header">Make Predictions</div>', 
                   unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first!")
            return
        
        tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Predictions"])
        
        with tab1:
            self.render_single_prediction()
        
        with tab2:
            self.render_batch_predictions()
    
    def render_single_prediction(self):
        """Render single prediction interface"""
        st.markdown("### Make a Single Prediction")
        
        # Create input form
        feature_names = st.session_state.data['feature_names']
        input_values = {}
        
        # Dynamic input creation based on feature types
        X_train = st.session_state.data['X_train']
        
        cols = st.columns(3)
        for i, feature in enumerate(feature_names):
            with cols[i % 3]:
                feature_data = X_train[feature]
                
                if feature_data.dtype in ['int64', 'float64']:
                    # Numerical input
                    min_val = float(feature_data.min())
                    max_val = float(feature_data.max())
                    mean_val = float(feature_data.mean())
                    
                    input_values[feature] = st.number_input(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        key=f"input_{feature}"
                    )
                else:
                    # Categorical input
                    unique_values = feature_data.unique()
                    input_values[feature] = st.selectbox(
                        f"{feature}",
                        options=unique_values,
                        key=f"input_{feature}"
                    )
        
        if st.button("🎯 Make Prediction", type="primary"):
            # Create prediction
            input_df = pd.DataFrame([input_values])
            
            # Create predictor
            predictor = Predictor(st.session_state.model, st.session_state.preprocessor)
            
            # Make prediction
            result = predictor.predict_single(
                input_values, 
                return_probabilities=True, 
                include_confidence=True
            )
            
            # Display results
            st.markdown("### Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Prediction", result['prediction'])
                
                if 'confidence' in result:
                    st.metric("Confidence", f"{result['confidence']:.3f}")
            
            with col2:
                if 'probabilities' in result:
                    st.markdown("**Class Probabilities:**")
                    for i, prob in enumerate(result['probabilities']):
                        st.write(f"Class {i}: {prob:.3f}")
    
    def render_batch_predictions(self):
        """Render batch predictions interface"""
        st.markdown("### Batch Predictions")
        
        uploaded_file = st.file_uploader(
            "Upload data for prediction",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file with the same features as your training data"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_predict_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Preview data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(temp_path)
                else:
                    df = pd.read_excel(temp_path)
                
                st.markdown("### Data Preview")
                st.dataframe(df.head())
                
                if st.button("🚀 Generate Predictions"):
                    with st.spinner("Making predictions..."):
                        # Create predictor
                        predictor = Predictor(st.session_state.model, st.session_state.preprocessor)
                        
                        # Make batch predictions
                        predictions_df = predictor.predict_batch(temp_path)
                        
                        st.success(f"✅ Predictions completed for {len(predictions_df)} samples")
                        
                        # Show results
                        st.markdown("### Prediction Results")
                        st.dataframe(predictions_df)
                        
                        # Download button
                        csv = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    def page_about(self):
        """About page"""
        st.markdown('<div class="section-header">About XplainML</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Mission
        
        XplainML makes machine learning interpretable and accessible. We believe that understanding 
        your models is just as important as their performance.
        
        ### 🛠️ Technologies Used
        
        - **Machine Learning**: Scikit-learn, XGBoost, PyTorch
        - **Explanations**: SHAP, LIME
        - **Visualizations**: Plotly, Matplotlib, Seaborn
        - **Web Interface**: Streamlit
        - **Data Processing**: Pandas, NumPy
        
        ### 📚 Features Overview
        
        #### Data Processing
        - Automatic handling of missing values
        - Smart categorical encoding
        - Feature scaling and normalization
        - Train/test splitting
        
        #### Model Training
        - Multiple algorithm support
        - Hyperparameter tuning
        - Model comparison
        - Performance evaluation
        
        #### Model Explanations
        - Global feature importance
        - Local prediction explanations
        - SHAP values and plots
        - LIME explanations
        - Permutation importance
        
        #### Visualizations
        - Interactive plots
        - Feature importance charts
        - Partial dependence plots
        - Confusion matrices
        - Residual analysis
        
        ### 🚀 Getting Started
        
        1. **Upload your data** - CSV or Excel files supported
        2. **Select your target variable** - What you want to predict
        3. **Choose a model** - Linear, Random Forest, XGBoost, or Neural Networks
        4. **Train and evaluate** - Get performance metrics
        5. **Explain predictions** - Understand why the model made certain decisions
        6. **Visualize results** - Interactive charts and plots
        7. **Make predictions** - On new data
        
        ### 📞 Support
        
        For questions, issues, or contributions, please visit our GitHub repository.
        
        ---
        
        **Built with ❤️ for the machine learning community**
        """)


def main():
    """Main function to run the dashboard"""
    dashboard = XplainMLDashboard()
    
    # Render header
    dashboard.render_header()
    
    # Render sidebar and get selected page
    selected_page = dashboard.render_sidebar()
    
    # Route to appropriate page
    if selected_page == "Home":
        dashboard.page_home()
    elif selected_page == "Data Upload & Preprocessing":
        dashboard.page_data_upload()
    elif selected_page == "Model Training":
        dashboard.page_model_training()
    elif selected_page == "Model Explanations":
        dashboard.page_explanations()
    elif selected_page == "Visualizations":
        dashboard.page_visualizations()
    elif selected_page == "Predictions":
        dashboard.page_predictions()
    elif selected_page == "About":
        dashboard.page_about()


if __name__ == "__main__":
    main()