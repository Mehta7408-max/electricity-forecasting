"""
FastAPI Server - REST API for GNN-based electricity price predictions

Endpoints:
- GET /health - Health check
- GET /predict/test - Test set predictions
- GET /predict/future?hours=24 - Future forecast
- GET /cheapest?hours=24&top=3 - Find cheapest hours
- GET /metrics - Model performance metrics
- POST /retrain - Trigger model retraining
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd

from gnn_predict import GNNPredictor, evaluate_predictions
from gnn_pipeline import run_full_pipeline
from gnn_config import DEFAULT_ZONE


# ────────────────────────────────────────────────────────────────
# Initialize FastAPI
# ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GNN Electricity Price Forecasting API",
    description="REST API for electricity price predictions using Graph Neural Networks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────
# Pydantic Models
# ────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    timestamp: str
    predicted_price: float


class CheapestHourResponse(BaseModel):
    timestamp: str
    predicted_price: float
    rank: int


class MetricsResponse(BaseModel):
    mae: float
    rmse: float
    r2: float
    mape: float
    num_samples: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool


class RetrainRequest(BaseModel):
    price_zone: Optional[str] = DEFAULT_ZONE
    model_type: Optional[str] = 'GCN'


# ────────────────────────────────────────────────────────────────
# Global predictor instance
# ────────────────────────────────────────────────────────────────
try:
    predictor = GNNPredictor()
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️  Failed to load model: {e}")
    predictor = None
    MODEL_LOADED = False


# ────────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "GNN Electricity Price Forecasting API",
        "version": "1.0.0",
        "description": "Graph Neural Network based electricity price predictions for Denmark",
        "endpoints": {
            "health": "/health",
            "predict_test": "/predict/test",
            "predict_future": "/predict/future?hours=24",
            "cheapest_hours": "/cheapest?hours=24&top=3",
            "metrics": "/metrics",
            "retrain": "/retrain (POST)"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and model health status."""
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=MODEL_LOADED
    )


@app.get("/predict/test", tags=["Predictions"])
async def predict_test_set():
    """
    Get predictions on the test set.
    
    Returns:
        List of predictions with actual values for comparison.
    """
    if not MODEL_LOADED or predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        test_predictions = predictor.predict_test_set()
        
        results = []
        for _, row in test_predictions.iterrows():
            results.append({
                "timestamp": row['timestamp'].isoformat(),
                "actual": float(row['actual']),
                "predicted": float(row['predicted']),
                "error": float(row['error']),
                "abs_error": float(row['abs_error'])
            })
        
        return {
            "num_predictions": len(results),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict/future", response_model=List[PredictionResponse], tags=["Predictions"])
async def predict_future(hours: int = 24):
    """
    Predict future electricity prices.
    
    Args:
        hours: Number of hours to predict (1-168).
    
    Returns:
        List of future price predictions.
    """
    if not MODEL_LOADED or predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if hours < 1 or hours > 168:
        raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
    
    try:
        future_predictions = predictor.predict_future(num_hours=hours)
        
        results = []
        for _, row in future_predictions.iterrows():
            results.append(PredictionResponse(
                timestamp=row['timestamp'].isoformat(),
                predicted_price=float(row['predicted_price'])
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/cheapest", response_model=List[CheapestHourResponse], tags=["Analysis"])
async def get_cheapest_hours(hours: int = 24, top: int = 3):
    """
    Find the cheapest hours in the forecast.
    
    Args:
        hours: Forecast horizon (1-168).
        top: Number of cheapest hours to return (1-24).
    
    Returns:
        List of cheapest hours with prices.
    """
    if not MODEL_LOADED or predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if hours < 1 or hours > 168:
        raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
    
    if top < 1 or top > 24:
        raise HTTPException(status_code=400, detail="Top must be between 1 and 24")
    
    try:
        future_predictions = predictor.predict_future(num_hours=hours)
        cheapest = predictor.get_cheapest_hours(future_predictions, top_n=top)
        
        results = []
        for idx, row in cheapest.iterrows():
            results.append(CheapestHourResponse(
                timestamp=row['timestamp'].isoformat(),
                predicted_price=float(row['predicted_price']),
                rank=idx + 1
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/expensive", tags=["Analysis"])
async def get_expensive_hours(hours: int = 24, top: int = 3):
    """
    Find the most expensive hours in the forecast.
    
    Args:
        hours: Forecast horizon (1-168).
        top: Number of most expensive hours to return (1-24).
    
    Returns:
        List of most expensive hours with prices.
    """
    if not MODEL_LOADED or predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if hours < 1 or hours > 168:
        raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
    
    if top < 1 or top > 24:
        raise HTTPException(status_code=400, detail="Top must be between 1 and 24")
    
    try:
        future_predictions = predictor.predict_future(num_hours=hours)
        expensive = predictor.get_most_expensive_hours(future_predictions, top_n=top)
        
        results = []
        for idx, row in expensive.iterrows():
            results.append({
                "timestamp": row['timestamp'].isoformat(),
                "predicted_price": float(row['predicted_price']),
                "rank": idx + 1
            })
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse, tags=["Evaluation"])
async def get_metrics():
    """
    Get model performance metrics on test set.
    
    Returns:
        Performance metrics (MAE, RMSE, R², MAPE).
    """
    if not MODEL_LOADED or predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        test_predictions = predictor.predict_test_set()
        metrics = evaluate_predictions(test_predictions)
        
        return MetricsResponse(
            mae=metrics['MAE'],
            rmse=metrics['RMSE'],
            r2=metrics['R2'],
            mape=metrics['MAPE'],
            num_samples=metrics['num_samples']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics calculation failed: {str(e)}")


@app.post("/retrain", tags=["Training"])
async def retrain_model(request: RetrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger model retraining in the background.
    
    Args:
        request: Retraining configuration.
    
    Returns:
        Status message.
    """
    try:
        # Add retraining task to background
        background_tasks.add_task(
            run_full_pipeline,
            price_zone=request.price_zone,
            model_type=request.model_type,
            skip_ingestion=False,
            skip_training=False,
            visualize=False
        )
        
        return {
            "status": "accepted",
            "message": "Retraining started in background",
            "price_zone": request.price_zone,
            "model_type": request.model_type
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {str(e)}")


@app.get("/stats", tags=["Analysis"])
async def get_statistics(hours: int = 24):
    """
    Get statistical summary of forecast prices.
    
    Args:
        hours: Forecast horizon (1-168).
    
    Returns:
        Price statistics (mean, min, max, std).
    """
    if not MODEL_LOADED or predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if hours < 1 or hours > 168:
        raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
    
    try:
        future_predictions = predictor.predict_future(num_hours=hours)
        prices = future_predictions['predicted_price']
        
        return {
            "mean": float(prices.mean()),
            "median": float(prices.median()),
            "min": float(prices.min()),
            "max": float(prices.max()),
            "std": float(prices.std()),
            "range": float(prices.max() - prices.min()),
            "num_hours": hours
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics calculation failed: {str(e)}")


# ────────────────────────────────────────────────────────────────
# Run server
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting GNN Electricity Price Forecasting API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )