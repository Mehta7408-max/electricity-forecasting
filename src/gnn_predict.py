"""
Prediction Module - Make electricity price predictions using trained GNN model
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from gnn_models import create_model
from gnn_config import GNN_CONFIG, DEVICE, ARTIFACTS_DIR, FORECAST_HORIZON
from gnn_graph_builder import TemporalGraphBuilder, load_graph


class GNNPredictor:
    """Make predictions with trained GNN model."""
    
    def __init__(self, model_path=None, graph_path=None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint.
            graph_path: Path to graph data.
        """
        self.model_path = model_path or ARTIFACTS_DIR / 'best_model.pt'
        self.graph_path = graph_path
        
        # Load graph data
        if graph_path:
            self.data = load_graph(graph_path)
        else:
            self.data = load_graph()
        
        # Create and load model
        num_features = self.data.x.shape[1]
        self.model = create_model(
            GNN_CONFIG['conv_type'], 
            num_features, 
            GNN_CONFIG
        ).to(DEVICE)
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Loaded model from {self.model_path}")
    
    @torch.no_grad()
    def predict_all(self):
        """
        Predict prices for all nodes in the graph.
        
        Returns:
            Dictionary with predictions and actuals.
        """
        self.model.eval()
        
        # Move data to device
        data = self.data.to(DEVICE)
        
        # Get predictions
        predictions = self.model(data.x, data.edge_index)
        
        # Convert to numpy
        preds = predictions.cpu().numpy()
        actuals = data.y.cpu().numpy()
        
        return {
            'predictions': preds,
            'actuals': actuals,
            'timestamps': self.data.timestamps
        }
    
    def predict_test_set(self):
        """
        Predict on test set only.
        
        Returns:
            DataFrame with predictions and actuals.
        """
        results = self.predict_all()
        
        # Filter test set
        test_mask = self.data.test_mask.cpu().numpy()
        
        df = pd.DataFrame({
            'timestamp': results['timestamps'][test_mask],
            'actual': results['actuals'][test_mask],
            'predicted': results['predictions'][test_mask]
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['error'] = df['actual'] - df['predicted']
        df['abs_error'] = np.abs(df['error'])
        df['pct_error'] = (df['error'] / df['actual']) * 100
        
        return df
    
    def predict_future(self, num_hours=FORECAST_HORIZON):
        """
        Predict future prices (next N hours).
        
        Note: This is a simplified version. For true future prediction,
        you would need to iteratively update the graph with new predictions.
        
        Args:
            num_hours: Number of hours to predict ahead.
        
        Returns:
            DataFrame with future predictions.
        """
        print(f"\n🔮 Predicting next {num_hours} hours...")
        
        # Get all predictions
        results = self.predict_all()
        
        # Get last num_hours predictions (as proxy for future)
        # In production, you'd build a new graph with predicted values
        last_indices = list(range(-num_hours, 0))
        
        last_timestamp = pd.to_datetime(results['timestamps'][-1])
        
        future_timestamps = [
            last_timestamp + timedelta(hours=i+1) 
            for i in range(num_hours)
        ]
        
        # Use last predictions as future forecast
        future_preds = results['predictions'][last_indices]
        
        df = pd.DataFrame({
            'timestamp': future_timestamps,
            'predicted_price': future_preds
        })
        
        return df
    
    def get_cheapest_hours(self, predictions_df, top_n=3):
        """
        Find the cheapest hours from predictions.
        
        Args:
            predictions_df: DataFrame with predictions.
            top_n: Number of cheapest hours to return.
        
        Returns:
            DataFrame with cheapest hours.
        """
        if 'predicted_price' in predictions_df.columns:
            price_col = 'predicted_price'
        else:
            price_col = 'predicted'
        
        cheapest = predictions_df.nsmallest(top_n, price_col)
        
        return cheapest[['timestamp', price_col]].reset_index(drop=True)
    
    def get_most_expensive_hours(self, predictions_df, top_n=3):
        """
        Find the most expensive hours from predictions.
        
        Args:
            predictions_df: DataFrame with predictions.
            top_n: Number of most expensive hours to return.
        
        Returns:
            DataFrame with most expensive hours.
        """
        if 'predicted_price' in predictions_df.columns:
            price_col = 'predicted_price'
        else:
            price_col = 'predicted'
        
        most_expensive = predictions_df.nlargest(top_n, price_col)
        
        return most_expensive[['timestamp', price_col]].reset_index(drop=True)


def evaluate_predictions(predictions_df):
    """
    Evaluate prediction quality.
    
    Args:
        predictions_df: DataFrame with 'actual' and 'predicted' columns.
    
    Returns:
        Dictionary with metrics.
    """
    if 'actual' not in predictions_df.columns:
        print("⚠️  No actual values for evaluation")
        return None
    
    actual = predictions_df['actual'].values
    predicted = predictions_df['predicted'].values
    
    # Compute metrics
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'num_samples': len(actual)
    }
    
    print("\n📊 Prediction Metrics:")
    print(f"   MAE:  {mae:.6f} DKK/kWh")
    print(f"   RMSE: {rmse:.6f} DKK/kWh")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R²:   {r2:.6f}")
    
    return metrics


if __name__ == "__main__":
    # Create predictor
    predictor = GNNPredictor()
    
    # Predict on test set
    print("\n📈 Making predictions on test set...")
    test_predictions = predictor.predict_test_set()
    
    print(f"\nTest set size: {len(test_predictions)}")
    print("\nSample predictions:")
    print(test_predictions.head(10))
    
    # Evaluate
    metrics = evaluate_predictions(test_predictions)
    
    # Find cheapest hours
    print("\n💰 Cheapest hours in test set:")
    cheapest = predictor.get_cheapest_hours(test_predictions, top_n=5)
    print(cheapest)
    
    # Predict future
    future_predictions = predictor.predict_future(num_hours=24)
    print("\n🔮 Next 24 hours forecast:")
    print(future_predictions)
    
    print("\n💰 Cheapest hours in next 24 hours:")
    future_cheapest = predictor.get_cheapest_hours(future_predictions, top_n=5)
    print(future_cheapest)