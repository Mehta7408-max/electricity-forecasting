"""
Heterogeneous GNN Prediction Module - Make predictions with trained hetero model
"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from hetero_models import create_hetero_model
from hetero_config import HETERO_GNN_CONFIG, DEVICE, ARTIFACTS_DIR, FORECAST_HORIZON


class HeteroGNNPredictor:
    """Make predictions with trained heterogeneous GNN model."""
    
    def __init__(self, model_path=None, graph_path=None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint.
            graph_path: Path to heterogeneous graph data.
        """
        self.model_path = model_path or ARTIFACTS_DIR / 'best_hetero_model.pt'
        
        # Load graph data
        if graph_path:
            self.data = torch.load(graph_path)
        else:
            from hetero_config import GRAPH_DIR
            graph_path = GRAPH_DIR / "hetero_temporal_graph.pt"
            self.data = torch.load(graph_path)
        
        # Get metadata
        metadata = (self.data.node_types, self.data.edge_types)
        
        # Create and load model
        self.model = create_hetero_model(
            HETERO_GNN_CONFIG.get('model_type', 'SimpleHeteroGNN'),
            metadata,
            HETERO_GNN_CONFIG
        ).to(DEVICE)
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Loaded heterogeneous model from {self.model_path}")
    
    @torch.no_grad()
    def predict_all(self):
        """
        Predict prices for all hour nodes in the graph.
        
        Returns:
            Dictionary with predictions and actuals.
        """
        self.model.eval()
        
        # Move data to device
        data = self.data.to(DEVICE)
        
        # Get predictions
        predictions = self.model(data.x_dict, data.edge_index_dict)
        
        # Convert to numpy
        preds = predictions.cpu().numpy()
        actuals = data['hour'].y.cpu().numpy()
        
        return {
            'predictions': preds,
            'actuals': actuals,
            'timestamps': self.data['hour'].timestamps
        }
    
    def predict_test_set(self):
        """
        Predict on test set only.
        
        Returns:
            DataFrame with predictions and actuals.
        """
        results = self.predict_all()
        
        # Filter test set
        test_mask = self.data['hour'].test_mask.cpu().numpy()
        
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
        
        Note: This is a simplified version using test set predictions.
        For true future prediction, you would need to:
        1. Fetch latest data
        2. Create new nodes
        3. Extend the graph
        4. Run the model
        
        Args:
            num_hours: Number of hours to predict ahead.
        
        Returns:
            DataFrame with future predictions.
        """
        print(f"\n🔮 Predicting next {num_hours} hours...")
        
        # Get all predictions
        results = self.predict_all()
        
        # Get last num_hours predictions (as proxy for future)
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
        """Find the cheapest hours from predictions."""
        if 'predicted_price' in predictions_df.columns:
            price_col = 'predicted_price'
        else:
            price_col = 'predicted'
        
        cheapest = predictions_df.nsmallest(top_n, price_col)
        return cheapest[['timestamp', price_col]].reset_index(drop=True)
    
    def get_most_expensive_hours(self, predictions_df, top_n=3):
        """Find the most expensive hours from predictions."""
        if 'predicted_price' in predictions_df.columns:
            price_col = 'predicted_price'
        else:
            price_col = 'predicted'
        
        most_expensive = predictions_df.nlargest(top_n, price_col)
        return most_expensive[['timestamp', price_col]].reset_index(drop=True)


def evaluate_hetero_predictions(predictions_df):
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
    
    print("\n📊 Heterogeneous GNN Prediction Metrics:")
    print(f"   MAE:  {mae:.6f} DKK/kWh")
    print(f"   RMSE: {rmse:.6f} DKK/kWh")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R²:   {r2:.6f}")
    
    return metrics


if __name__ == "__main__":
    # Create predictor
    predictor = HeteroGNNPredictor()
    
    # Predict on test set
    print("\n📈 Making predictions on test set...")
    test_predictions = predictor.predict_test_set()
    
    print(f"\nTest set size: {len(test_predictions)}")
    print("\nSample predictions:")
    print(test_predictions.head(10))
    
    # Evaluate
    metrics = evaluate_hetero_predictions(test_predictions)
    
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