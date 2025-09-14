from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
import json
import pickle
import tempfile
import shutil
from datetime import datetime

# Add the backend directory to the path
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Import XplainML modules
try:
    from backend.data_preprocessing import DataPreprocessor, create_sample_data
    from backend.models import MLModel, ModelTuner
    from backend.explainer import ModelExplainer
    from backend.visualizer import ModelVisualizer
    from backend.prediction import Predictor
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    print("Make sure the backend directory is properly set up")

# Create FastAPI app
app = FastAPI(
    title="XplainML API",
    description="Backend API for XplainML - Interpretable Machine Learning Platform",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for sessions (in production, use Redis or database)
sessions: Dict[str, Any] = {}
models: Dict[str, Any] = {}

# Helper functions
def create_session_id() -> str:
    return str(uuid.uuid4())

def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            'id': session_id,
            'created_at': datetime.now().isoformat(),
            'data_loaded': False,
            'model_trained': False,
            'explanations_generated': False,
            'current_step': 1
        }
    return sessions[session_id]

@app.get("/")
async def root():
    return {"message": "XplainML API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Session Management
@app.post("/sessions")
async def create_session():
    session_id = create_session_id()
    session = get_session(session_id)
    return {"session_id": session_id, "session": session}

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    session = get_session(session_id)
    return {"session": session}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    # Clean up associated models
    for model_id in list(models.keys()):
        if models[model_id].get('session_id') == session_id:
            del models[model_id]
    return {"message": "Session deleted successfully"}

# Data Upload and Processing
@app.post("/data/upload")
async def upload_data(
    file: UploadFile = File(...),
    session_id: str = None
):
    if not session_id:
        session_id = create_session_id()
    
    session = get_session(session_id)
    
    try:
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, 'wb') as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load and analyze the data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please use CSV or Excel files.")
        
        # Analyze data
        data_summary = {
            'filename': file.filename,
            'originalShape': df.shape,
            'featureNames': df.columns.tolist(),
            'dataTypes': df.dtypes.astype(str).to_dict(),
            'missingValues': df.isnull().sum().to_dict(),
            'sampleData': df.head(10).to_dict('records'),
            'taskType': 'auto'  # Will be determined during preprocessing
        }
        
        # Store in session
        session['data_summary'] = data_summary
        session['temp_file_path'] = temp_file_path
        session['data_loaded'] = True
        session['current_step'] = 2
        
        # Clean up temp directory (keep file for preprocessing)
        # shutil.rmtree(temp_dir)
        
        return {
            "session_id": session_id,
            "data_summary": data_summary,
            "message": "Data uploaded successfully"
        }
        
    except Exception as e:
        # Clean up on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/data/preprocess")
async def preprocess_data(
    session_id: str,
    target_column: str,
    missing_strategy: str = 'mean',
    encoding_type: str = 'auto',
    scaling_type: str = 'standard',
    test_size: float = 0.2
):
    session = get_session(session_id)
    
    if not session.get('data_loaded'):
        raise HTTPException(status_code=400, detail="No data uploaded for this session")
    
    try:
        temp_file_path = session.get('temp_file_path')
        if not temp_file_path or not os.path.exists(temp_file_path):
            raise HTTPException(status_code=400, detail="Uploaded file not found")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Preprocess the data
        processed_data = preprocessor.preprocess_pipeline(
            file_path=temp_file_path,
            target_column=target_column,
            test_size=test_size,
            missing_strategy=missing_strategy,
            encoding_type=encoding_type,
            scaling_type=scaling_type,
            random_state=42
        )
        
        # Update session
        session['preprocessed_data'] = {
            'originalShape': processed_data['original_shape'],
            'featureNames': processed_data['feature_names'],
            'taskType': processed_data['task_type'],
            'trainShape': processed_data['X_train'].shape,
            'testShape': processed_data['X_test'].shape,
            'targetColumn': target_column,
            'preprocessorSettings': {
                'missingStrategy': missing_strategy,
                'encodingType': encoding_type,
                'scalingType': scaling_type,
                'testSize': test_size
            }
        }
        
        # Store preprocessor and data for later use
        session['preprocessor'] = preprocessor
        session['processed_data_full'] = processed_data
        session['current_step'] = 2
        
        return {
            "message": "Data preprocessed successfully",
            "preprocessed_data": session['preprocessed_data']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing data: {str(e)}")

# Sample Data Generation
@app.post("/data/generate-sample")
async def generate_sample_data(
    session_id: str = None,
    n_samples: int = 1000,
    task_type: str = 'classification'
):
    if not session_id:
        session_id = create_session_id()
    
    session = get_session(session_id)
    
    try:
        # Create temp directory for sample data
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, 'sample_data.csv')
        
        # Generate sample data
        df = create_sample_data(temp_file_path, n_samples=n_samples)
        
        # Analyze the generated data
        data_summary = {
            'filename': 'sample_data.csv',
            'originalShape': df.shape,
            'featureNames': df.columns.tolist(),
            'dataTypes': df.dtypes.astype(str).to_dict(),
            'missingValues': df.isnull().sum().to_dict(),
            'sampleData': df.head(10).to_dict('records'),
            'taskType': 'classification'
        }
        
        # Store in session
        session['data_summary'] = data_summary
        session['temp_file_path'] = temp_file_path
        session['data_loaded'] = True
        session['current_step'] = 2
        
        return {
            "session_id": session_id,
            "data_summary": data_summary,
            "message": "Sample data generated successfully"
        }
        
    except Exception as e:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error generating sample data: {str(e)}")

# Model Training Endpoints
@app.post("/models/train")
async def train_model(
    session_id: str,
    model_type: str,
    use_tuning: bool = False,
    cv_folds: int = 5,
    custom_parameters: Optional[Dict[str, Any]] = None
):
    session = get_session(session_id)
    
    if not session.get('processed_data_full'):
        raise HTTPException(status_code=400, detail="No preprocessed data available")
    
    try:
        processed_data = session['processed_data_full']
        
        # Create model ID
        model_id = str(uuid.uuid4())
        
        if use_tuning:
            # Hyperparameter tuning
            tuner = ModelTuner(model_type, processed_data['task_type'])
            model = tuner.tune(
                processed_data['X_train'], 
                processed_data['y_train'],
                cv=cv_folds
            )
            best_params = tuner.best_params
            best_score = tuner.best_score
        else:
            # Regular training
            model = MLModel(model_type, processed_data['task_type'])
            if custom_parameters:
                model.set_params(**custom_parameters)
            model.fit(processed_data['X_train'], processed_data['y_train'])
            best_params = None
            best_score = None
        
        # Evaluate model
        train_metrics = model.evaluate(processed_data['X_train'], processed_data['y_train'])
        test_metrics = model.evaluate(processed_data['X_test'], processed_data['y_test'])
        
        # Store model
        model_info = {
            'id': model_id,
            'session_id': session_id,
            'type': model_type,
            'model': model,
            'metrics': {
                'train': train_metrics,
                'test': test_metrics
            },
            'best_params': best_params,
            'best_score': best_score,
            'training_time': 0,  # TODO: Add timing
            'created_at': datetime.now().isoformat()
        }
        
        models[model_id] = model_info
        
        # Update session
        session['current_model_id'] = model_id
        session['model_trained'] = True
        session['current_step'] = 3
        
        # Return model info without the actual model object
        response_model = {k: v for k, v in model_info.items() if k != 'model'}
        
        return {
            "message": "Model trained successfully",
            "model": response_model
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = models[model_id]
    # Return model info without the actual model object
    response_model = {k: v for k, v in model_info.items() if k != 'model'}
    return {"model": response_model}

@app.get("/sessions/{session_id}/models")
async def get_session_models(session_id: str):
    session_models = [
        {k: v for k, v in model_info.items() if k != 'model'}
        for model_info in models.values()
        if model_info.get('session_id') == session_id
    ]
    return {"models": session_models}

# Model Explanation Endpoints
@app.post("/explanations/generate")
async def generate_explanations(
    session_id: str,
    model_id: str,
    methods: List[str] = ['shap', 'permutation'],
    num_samples: int = 5
):
    session = get_session(session_id)
    
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not session.get('processed_data_full'):
        raise HTTPException(status_code=400, detail="No preprocessed data available")
    
    try:
        model_info = models[model_id]
        model = model_info['model']
        processed_data = session['processed_data_full']
        
        # Initialize explainer
        explainer = ModelExplainer(
            model=model,
            X_train=processed_data['X_train'],
            y_train=processed_data['y_train'],
            feature_names=processed_data['feature_names']
        )
        
        # Generate explanations
        explanations = {}
        
        if 'shap' in methods:
            try:
                explanations['shap'] = explainer.explain_global(method='shap')
            except Exception as e:
                print(f"SHAP explanation failed: {str(e)}")
        
        if 'lime' in methods:
            try:
                explanations['lime'] = explainer.explain_global(method='lime')
            except Exception as e:
                print(f"LIME explanation failed: {str(e)}")
        
        if 'permutation' in methods:
            try:
                explanations['permutation'] = explainer.explain_global(method='permutation')
            except Exception as e:
                print(f"Permutation explanation failed: {str(e)}")
        
        # Model feature importance
        importance = model.get_feature_importance()
        if importance:
            explanations['builtin'] = {'feature_importance': importance}
        
        # Generate local explanations
        local_explanations = []
        for i in range(min(num_samples, len(processed_data['X_test']))):
            sample = processed_data['X_test'].iloc[i]
            try:
                local_exp = explainer.explain_local(sample, method='lime')
                local_explanations.append({
                    'sample_index': i,
                    'explanation': local_exp
                })
            except Exception as e:
                print(f"Local explanation failed for sample {i+1}: {str(e)}")
        
        # Store explanations in session
        session['global_explanations'] = explanations
        session['local_explanations'] = local_explanations
        session['explanations_generated'] = True
        session['current_step'] = 4
        
        return {
            "message": "Explanations generated successfully",
            "global_explanations": explanations,
            "local_explanations": local_explanations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanations: {str(e)}")

@app.get("/sessions/{session_id}/explanations")
async def get_explanations(session_id: str):
    session = get_session(session_id)
    
    return {
        "global_explanations": session.get('global_explanations', {}),
        "local_explanations": session.get('local_explanations', [])
    }

# Prediction Endpoints
@app.post("/predictions/single")
async def predict_single(
    session_id: str,
    model_id: str,
    features: Dict[str, Any]
):
    session = get_session(session_id)
    
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not session.get('processed_data_full'):
        raise HTTPException(status_code=400, detail="No preprocessed data available")
    
    try:
        model_info = models[model_id]
        model = model_info['model']
        processed_data = session['processed_data_full']
        
        # Create prediction input
        input_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Get prediction probabilities if classification
        probabilities = None
        if hasattr(model, 'predict_proba') and processed_data['task_type'] == 'classification':
            probs = model.predict_proba(input_df)[0]
            probabilities = probs.tolist()
        
        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "features": features
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.post("/predictions/batch")
async def predict_batch(
    session_id: str,
    model_id: str,
    data: List[Dict[str, Any]]
):
    session = get_session(session_id)
    
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not session.get('processed_data_full'):
        raise HTTPException(status_code=400, detail="No preprocessed data available")
    
    try:
        model_info = models[model_id]
        model = model_info['model']
        processed_data = session['processed_data_full']
        
        # Create prediction input
        input_df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(input_df).tolist()
        
        # Get prediction probabilities if classification
        probabilities = None
        if hasattr(model, 'predict_proba') and processed_data['task_type'] == 'classification':
            probs = model.predict_proba(input_df)
            probabilities = probs.tolist()
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "count": len(predictions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making batch predictions: {str(e)}")

@app.post("/predictions/file")
async def predict_file(
    session_id: str,
    model_id: str,
    file: UploadFile = File(...)
):
    session = get_session(session_id)
    
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not session.get('processed_data_full'):
        raise HTTPException(status_code=400, detail="No preprocessed data available")
    
    try:
        model_info = models[model_id]
        model = model_info['model']
        processed_data = session['processed_data_full']
        
        # Save uploaded file temporarily
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, 'wb') as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Check features match
        expected_features = set(processed_data['feature_names'])
        provided_features = set(df.columns)
        
        missing_features = expected_features - provided_features
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing features: {list(missing_features)}"
            )
        
        # Prepare data (only use expected features)
        prediction_data = df[list(expected_features)]
        
        # Make predictions
        predictions = model.predict(prediction_data).tolist()
        
        # Get prediction probabilities if classification
        probabilities = None
        if hasattr(model, 'predict_proba') and processed_data['task_type'] == 'classification':
            probs = model.predict_proba(prediction_data)
            probabilities = probs.tolist()
        
        # Add predictions to original dataframe
        result_df = df.copy()
        result_df['Prediction'] = predictions
        
        # Add probabilities if available
        if probabilities:
            for i, prob_row in enumerate(probabilities):
                for j, prob in enumerate(prob_row):
                    result_df[f'Probability_Class_{j}'] = [row[j] for row in probabilities]
        
        # Clean up temp file
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "results": result_df.to_dict('records'),
            "count": len(predictions),
            "filename": file.filename
        }
        
    except Exception as e:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error processing file predictions: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)