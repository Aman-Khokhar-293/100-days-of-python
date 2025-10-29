# FastAPI server for model deployment and predictions

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from ai_automl_engine import AutoMLEngine
import pickle
import io
from typing import List, Dict
import uvicorn


app = FastAPI(
    title="AI AutoML API",
    description="REST API for automated machine learning",
    version="1.0.0"
)

trained_models = {}  # store trained models in memory


class PredictionRequest(BaseModel):
    data: List[Dict[str, float]]
    model_id: str = "default"


class TrainingResponse(BaseModel):
    model_id: str
    task_type: str
    best_model: str
    best_score: float
    message: str


@app.get("/")
async def root():
    return {
        "message": "🚀 AI AutoML API",
        "version": "1.0.0",
        "endpoints": {
            "train": "/train",
            "predict": "/predict",
            "models": "/models",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(trained_models)
    }


@app.post("/train", response_model=TrainingResponse)
async def train_model(
    file: UploadFile = File(...),
    target_column: str = "target",
    task_type: str = "auto",
    model_id: str = "default"
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found in dataset"
            )
        
        # train the model
        automl = AutoMLEngine(task_type=task_type)
        X, y = automl.preprocess_data(df, target_column)
        automl.train_models(X, y)
        
        trained_models[model_id] = {
            'engine': automl,
            'feature_columns': list(X.columns)
        }
        
        summary = automl.get_model_summary()
        
        return TrainingResponse(
            model_id=model_id,
            task_type=summary['task_type'],
            best_model=summary['best_model'],
            best_score=summary['best_score'],
            message="Model trained successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        if request.model_id not in trained_models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_id}' not found"
            )
        
        model_info = trained_models[request.model_id]
        automl = model_info['engine']
        df = pd.DataFrame(request.data)
        
        expected_cols = set(model_info['feature_columns'])
        provided_cols = set(df.columns)
        
        if not expected_cols.issubset(provided_cols):
            missing = expected_cols - provided_cols
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {missing}"
            )
        
        predictions = automl.predict(df[model_info['feature_columns']])
        
        return {
            "model_id": request.model_id,
            "predictions": predictions.tolist(),
            "count": len(predictions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    models_info = []
    
    for model_id, model_info in trained_models.items():
        summary = model_info['engine'].get_model_summary()
        models_info.append({
            "model_id": model_id,
            "task_type": summary['task_type'],
            "best_model": summary['best_model'],
            "best_score": summary['best_score'],
            "feature_count": len(model_info['feature_columns'])
        })
    
    return {
        "models": models_info,
        "total": len(models_info)
    }


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    if model_id not in trained_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )
    
    del trained_models[model_id]
    
    return {
        "message": f"Model '{model_id}' deleted successfully"
    }


@app.get("/models/{model_id}/summary")
async def get_model_summary(model_id: str):
    if model_id not in trained_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )
    
    model_info = trained_models[model_id]
    summary = model_info['engine'].get_model_summary()
    
    feature_importance = None
    if summary['feature_importance'] is not None:
        feature_importance = summary['feature_importance'].to_dict('records')
    
    model_results = {}
    for name, result in summary['all_results'].items():
        if summary['task_type'] == 'classification':
            model_results[name] = {'accuracy': result['accuracy']}
        else:
            model_results[name] = {
                'r2_score': result['r2_score'],
                'mse': result['mse']
            }
    
    return {
        "model_id": model_id,
        "task_type": summary['task_type'],
        "best_model": summary['best_model'],
        "best_score": summary['best_score'],
        "feature_count": len(model_info['feature_columns']),
        "features": model_info['feature_columns'],
        "feature_importance": feature_importance,
        "all_models": model_results
    }


if __name__ == "__main__":
    print("🚀 Starting AI AutoML API Server...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Interactive API: http://localhost:8000/redoc")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
