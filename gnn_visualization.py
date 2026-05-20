"""
Visualization Module - Plot GNN predictions, training history, and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from gnn_config import ARTIFACTS_DIR


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/MAE over epochs.
    
    Args:
        history: Dictionary with training history.
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[1].plot(epochs, history['train_mae'], label='Train MAE', linewidth=2)
    axes[1].plot(epochs, history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE (DKK/kWh)', fontsize=12)
    axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved training history plot to {save_path}")
    
    plt.show()


def plot_predictions_vs_actual(predictions_df, save_path=None):
    """
    Plot predicted vs actual prices.
    
    Args:
        predictions_df: DataFrame with 'actual' and 'predicted' columns.
        save_path: Path to save the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Time series plot
    if 'timestamp' in predictions_df.columns:
        axes[0].plot(predictions_df['timestamp'], predictions_df['actual'], 
                    label='Actual', alpha=0.7, linewidth=1.5)
        axes[0].plot(predictions_df['timestamp'], predictions_df['predicted'], 
                    label='Predicted', alpha=0.7, linewidth=1.5)
        axes[0].set_xlabel('Timestamp', fontsize=12)
    else:
        axes[0].plot(predictions_df['actual'], label='Actual', alpha=0.7, linewidth=1.5)
        axes[0].plot(predictions_df['predicted'], label='Predicted', alpha=0.7, linewidth=1.5)
        axes[0].set_xlabel('Sample Index', fontsize=12)
    
    axes[0].set_ylabel('Price (DKK/kWh)', fontsize=12)
    axes[0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(predictions_df['actual'], predictions_df['predicted'], 
                   alpha=0.5, s=20)
    
    # Add diagonal line
    min_val = min(predictions_df['actual'].min(), predictions_df['predicted'].min())
    max_val = max(predictions_df['actual'].max(), predictions_df['predicted'].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
    
    axes[1].set_xlabel('Actual Price (DKK/kWh)', fontsize=12)
    axes[1].set_ylabel('Predicted Price (DKK/kWh)', fontsize=12)
    axes[1].set_title('Prediction Scatter Plot', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved predictions plot to {save_path}")
    
    plt.show()


def plot_error_distribution(predictions_df, save_path=None):
    """
    Plot error distribution.
    
    Args:
        predictions_df: DataFrame with 'actual' and 'predicted' columns.
        save_path: Path to save the plot.
    """
    errors = predictions_df['actual'] - predictions_df['predicted']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error (DKK/kWh)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot
    axes[1].boxplot(errors, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].axhline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_ylabel('Prediction Error (DKK/kWh)', fontsize=12)
    axes[1].set_title('Error Box Plot', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved error distribution plot to {save_path}")
    
    plt.show()
    
    # Print error statistics
    print("\n📊 Error Statistics:")
    print(f"   Mean Error: {errors.mean():.6f} DKK/kWh")
    print(f"   Std Error: {errors.std():.6f} DKK/kWh")
    print(f"   Min Error: {errors.min():.6f} DKK/kWh")
    print(f"   Max Error: {errors.max():.6f} DKK/kWh")


def plot_hourly_performance(predictions_df, save_path=None):
    """
    Plot model performance by hour of day.
    
    Args:
        predictions_df: DataFrame with predictions and timestamp.
        save_path: Path to save the plot.
    """
    if 'timestamp' not in predictions_df.columns:
        print("⚠️  Timestamp column not found")
        return
    
    # Extract hour
    df = predictions_df.copy()
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['error'] = df['actual'] - df['predicted']
    df['abs_error'] = np.abs(df['error'])
    
    # Group by hour
    hourly_stats = df.groupby('hour').agg({
        'abs_error': ['mean', 'std'],
        'error': 'mean'
    }).reset_index()
    
    hourly_stats.columns = ['hour', 'mae', 'std', 'bias']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # MAE by hour
    axes[0].bar(hourly_stats['hour'], hourly_stats['mae'], 
               alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Hour of Day', fontsize=12)
    axes[0].set_ylabel('Mean Absolute Error (DKK/kWh)', fontsize=12)
    axes[0].set_title('MAE by Hour of Day', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Bias by hour
    axes[1].bar(hourly_stats['hour'], hourly_stats['bias'], 
               alpha=0.7, edgecolor='black', color='orange')
    axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Hour of Day', fontsize=12)
    axes[1].set_ylabel('Mean Error (DKK/kWh)', fontsize=12)
    axes[1].set_title('Prediction Bias by Hour of Day', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved hourly performance plot to {save_path}")
    
    plt.show()


def plot_future_forecast(future_df, save_path=None):
    """
    Plot future price forecast.
    
    Args:
        future_df: DataFrame with future predictions.
        save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(future_df['timestamp'], future_df['predicted_price'], 
           marker='o', linewidth=2, markersize=6, alpha=0.7)
    
    # Highlight cheapest and most expensive hours
    cheapest_idx = future_df['predicted_price'].idxmin()
    most_expensive_idx = future_df['predicted_price'].idxmax()
    
    ax.scatter(future_df.loc[cheapest_idx, 'timestamp'], 
              future_df.loc[cheapest_idx, 'predicted_price'],
              color='green', s=200, marker='*', 
              label='Cheapest Hour', zorder=5)
    
    ax.scatter(future_df.loc[most_expensive_idx, 'timestamp'], 
              future_df.loc[most_expensive_idx, 'predicted_price'],
              color='red', s=200, marker='*', 
              label='Most Expensive Hour', zorder=5)
    
    ax.set_xlabel('Timestamp', fontsize=12)
    ax.set_ylabel('Predicted Price (DKK/kWh)', fontsize=12)
    ax.set_title('24-Hour Price Forecast', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved forecast plot to {save_path}")
    
    plt.show()


def plot_model_comparison(results_dict, save_path=None):
    """
    Compare performance of different GNN models.
    
    Args:
        results_dict: Dictionary with model names as keys and metrics as values.
        save_path: Path to save the plot.
    """
    models = list(results_dict.keys())
    metrics = ['test_mae', 'test_rmse', 'test_r2', 'test_mape']
    metric_names = ['MAE (DKK/kWh)', 'RMSE (DKK/kWh)', 'R²', 'MAPE (%)']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [results_dict[model][metric] for model in models]
        
        bars = axes[idx].bar(models, values, alpha=0.7, edgecolor='black')
        
        # Color bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        axes[idx].set_ylabel(metric_name, fontsize=12)
        axes[idx].set_title(f'{metric_name} Comparison', fontsize=13, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved model comparison plot to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    import json
    
    # Load training history
    history_path = ARTIFACTS_DIR / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        plot_training_history(history, save_path=ARTIFACTS_DIR / 'training_history.png')