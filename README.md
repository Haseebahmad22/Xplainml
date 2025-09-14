# XplainML - Interpretable Machine Learning for Tabular Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

XplainML is a comprehensive Python tool designed to make machine learning predictions for tabular datasets interpretable and explainable. It focuses on helping users understand model decisions, feature importance, and the reasoning behind predictions, especially for complex models like deep learning or ensemble methods.

## 🎯 Purpose & Motivation

While machine learning models can achieve high accuracy, many models (e.g., Random Forest, XGBoost, Neural Networks) are often seen as "black boxes." XplainML addresses this by:

- 🔍 Providing insights into feature importance
- 📊 Explaining individual predictions  
- 🤝 Helping users trust and debug ML models
- 📈 Making AI decisions transparent and accountable

This project is particularly useful in finance, healthcare, and business analytics, where understanding the rationale behind a prediction is as important as the prediction itself.

## ✨ Core Features

### 📂 Data Loading & Preprocessing
- Accepts tabular datasets (CSV, Excel, or Pandas DataFrame)
- Handles missing values, categorical encoding, and feature scaling
- Automatic train/test splitting with stratification support
- Smart data type detection and preprocessing

### 🤖 Model Training
- **Linear Models**: Linear Regression, Logistic Regression
- **Tree-Based Models**: Random Forest, XGBoost
- **Deep Learning Models**: PyTorch-based Neural Networks
- **Hyperparameter Tuning**: Automated grid search and random search
- **Model Comparison**: Compare multiple algorithms automatically

### 🎯 Prediction
- Make predictions on unseen data
- Support for both regression and classification tasks
- Batch prediction capabilities
- Confidence scoring and probability estimates

### 🔍 Interpretability & Explainability
- **SHAP**: Model-agnostic explanations with Shapley values
- **LIME**: Local interpretable model-agnostic explanations
- **Permutation Importance**: Feature importance through permutation
- **Global and Local Explanations**: Understand both overall model behavior and individual predictions
- **Feature Interactions**: Analyze how features interact with each other

### 📊 Visualization
- Interactive feature importance plots
- Partial Dependence Plots (PDP) for feature effect visualization
- SHAP summary plots and waterfall charts
- Confusion matrices and performance metrics
- Residual analysis for regression tasks
- Interactive dashboards using Plotly

### 💻 User Interaction
- **CLI Interface**: Command-line tool for dataset input, model selection, and explanation options
- **Web Dashboard**: Interactive Streamlit-based web interface
- **Jupyter Notebook Support**: Easy integration with data science workflows

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Haseebahmad22/Xplainml.git
cd Xplainml
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Basic Usage

#### CLI Interface
```bash
# Basic usage with Random Forest and SHAP explanations
python xplainml.py --mode cli --dataset data.csv --target target_column --model random_forest --explain shap

# Compare multiple models
python xplainml.py --mode cli --dataset data.csv --target target_column --compare-models

# Hyperparameter tuning
python xplainml.py --mode cli --dataset data.csv --target target_column --model xgboost --tune

# Generate sample data for testing
python xplainml.py --mode cli --generate-sample --samples 1000
```

#### Web Dashboard
```bash
# Launch interactive Streamlit dashboard
python xplainml.py --mode web --port 8501

# Or directly run the frontend
streamlit run frontend/app.py
```

#### Web Dashboard
```bash
# Launch the interactive web dashboard
streamlit run dashboard/app.py
```

#### Python API
```python
from src.data_preprocessing import DataPreprocessor
from src.models import MLModel
from src.explainer import ModelExplainer
from src.visualizer import ModelVisualizer

# Load and preprocess data
preprocessor = DataPreprocessor()
data = preprocessor.preprocess_pipeline('data.csv', 'target_column')

# Train model
model = MLModel('random_forest', data['task_type'])
model.fit(data['X_train'], data['y_train'])

# Generate explanations
explainer = ModelExplainer(model, data['X_train'], data['y_train'])
explanations = explainer.explain_global()

# Create visualizations
visualizer = ModelVisualizer(model, explainer)
visualizer.plot_feature_importance(explanations['shap']['feature_importance'])
```

## 📁 Project Structure

```
Xplainml/
├── backend/                      # Core ML and explanation modules
│   ├── data_preprocessing.py     # Data loading and preprocessing
│   ├── models.py                 # ML model implementations  
│   ├── prediction.py             # Prediction functionality
│   ├── explainer.py              # Model explanation methods
│   ├── visualizer.py             # Visualization tools
│   ├── xplainml.py               # CLI interface
│   └── __init__.py               # Package initialization
├── frontend/                     # Web interface
│   └── app.py                    # Streamlit dashboard
├── notebooks/                    # Jupyter notebooks
│   └── xplainml_demo.ipynb      # Demo notebook
├── data/                         # Data directory
├── tests/                        # Unit tests
├── output/                       # Generated results and plots
├── requirements.txt              # Python dependencies
├── xplainml.py                   # Main entry point
└── README.md                     # This file
```

## 📊 Supported Models

| Model Type | Classification | Regression | Hyperparameter Tuning | Feature Importance |
|------------|----------------|------------|----------------------|-------------------|
| Linear Models | ✅ | ✅ | ✅ | ✅ |
| Random Forest | ✅ | ✅ | ✅ | ✅ |
| XGBoost | ✅ | ✅ | ✅ | ✅ |
| Neural Networks | ✅ | ✅ | ✅ | ❌ |

## 🔍 Explanation Methods

| Method | Global Explanations | Local Explanations | Model Agnostic | Visualization |
|--------|-------------------|-------------------|----------------|---------------|
| SHAP | ✅ | ✅ | ✅ | ✅ |
| LIME | ❌ | ✅ | ✅ | ✅ |
| Permutation Importance | ✅ | ❌ | ✅ | ✅ |
| Built-in Importance | ✅ | ❌ | ❌ | ✅ |

## 📈 Example Use Cases

### 1. Healthcare: Medical Diagnosis
```python
# Predict disease based on symptoms and lab results
# Explain which symptoms are most important for diagnosis
python xplainml.py --dataset medical_data.csv --target diagnosis --model xgboost --explain shap --output html
```

### 2. Finance: Credit Scoring
```python
# Predict loan default risk
# Understand which financial factors drive decisions
python xplainml.py --dataset credit_data.csv --target default --model random_forest --tune --explain all
```

### 3. Marketing: Customer Segmentation
```python
# Predict customer lifetime value
# Identify key customer characteristics
python xplainml.py --dataset customer_data.csv --target clv --model neural_network --explain shap
```

## 🛠️ Advanced Features

### Custom Model Configuration
```python
# Neural network with custom architecture
model = MLModel('neural_network', 'classification',
                hidden_sizes=[128, 64, 32], 
                dropout_rate=0.3,
                learning_rate=0.001)
```

### Batch Predictions
```python
predictor = Predictor(model, preprocessor)
predictions = predictor.predict_batch('new_data.csv', 'predictions.csv')
```

### Interactive Explanations
```python
# Compare different explanation methods
comparison = explainer.compare_explanations(sample, methods=['shap', 'lime'])
```

## 📋 Requirements

- **Python**: 3.8 or higher
- **Core Libraries**: pandas, numpy, scikit-learn
- **Deep Learning**: torch, torchvision
- **Explanations**: shap, lime
- **Visualizations**: matplotlib, seaborn, plotly
- **Web Interface**: streamlit
- **Additional**: xgboost, openpyxl, click, joblib

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/Haseebahmad22/Xplainml.git
cd Xplainml

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Start development server
streamlit run dashboard/app.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SHAP**: For providing excellent model explanation capabilities
- **LIME**: For local interpretable explanations
- **Streamlit**: For the amazing web framework
- **Scikit-learn**: For the robust ML foundation
- **Plotly**: For interactive visualizations

## 📞 Support

- 📧 **Email**: [your-email@example.com]
- 🐛 **Issues**: [GitHub Issues](https://github.com/Haseebahmad22/Xplainml/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Haseebahmad22/Xplainml/discussions)

## 📚 Documentation

For detailed documentation, examples, and tutorials, visit our [Documentation Site](https://haseebahmad22.github.io/Xplainml/).

## 🎉 Star History

If you find XplainML useful, please consider giving it a star! ⭐

---

**Built with ❤️ for the machine learning community**

*Making AI interpretable, one model at a time.*