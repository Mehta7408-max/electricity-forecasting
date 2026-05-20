"""
Main Pipeline - Complete workflow for GNN-based electricity price forecasting

This script runs the entire pipeline:
1. Data ingestion (fetch prices and weather)
2. Feature engineering
3. Graph construction
4. Model training
5. Prediction and evaluation
6. Visualization
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from gnn_config import ARTIFACTS_DIR, GNN_CONFIG, DEFAULT_ZONE
from gnn_data_ingestion import run_ingestion
from gnn_feature_engineering import prepare_data_for_gnn
from gnn_graph_builder import build_and_save_graph, load_graph
from gnn_train import train_gnn_model
from gnn_predict import GNNPredictor, evaluate_predictions
from gnn_visualization import (
    plot_training_history, 
    plot_predictions_vs_actual,
    plot_error_distribution,
    plot_hourly_performance,
    plot_future_forecast
)


def run_full_pipeline(
    price_zone=DEFAULT_ZONE,
    model_type='GCN',
    skip_ingestion=False,
    skip_training=False,
    visualize=True
):
    """
    Run the complete GNN pipeline.
    
    Args:
        price_zone: Price zone to forecast (DK1 or DK2).
        model_type: Type of GNN model (GCN, GAT, GraphSAGE, GIN).
        skip_ingestion: Skip data ingestion if data already exists.
        skip_training: Skip training if model already exists.
        visualize: Generate visualization plots.
    
    Returns:
        Dictionary with results and metrics.
    """
    print("="*70)
    print("🚀 GNN-BASED ELECTRICITY PRICE FORECASTING PIPELINE")
    print("="*70)
    print(f"Price Zone: {price_zone}")
    print(f"Model Type: {model_type}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    
    # ────────────────────────────────────────────────────────────────
    # STEP 1: Data Ingestion
    # ────────────────────────────────────────────────────────────────
    if not skip_ingestion:
        print("\n" + "="*70)
        print("STEP 1: DATA INGESTION")
        print("="*70)
        
        prices_df = run_ingestion()
        results['ingestion'] = {
            'num_records': len(prices_df),
            'date_range': {
                'start': str(prices_df['hour_utc'].min()),
                'end': str(prices_df['hour_utc'].max())
            }
        }
    else:
        print("\n⏭️  Skipping data ingestion")
    
    # ────────────────────────────────────────────────────────────────
    # STEP 2: Feature Engineering
    # ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    
    df = prepare_data_for_gnn(price_zone)
    
    if df is None:
        print("❌ Feature engineering failed. Exiting.")
        return None
    
    results['features'] = {
        'num_samples': len(df),
        'num_features': len(df.columns),
        'date_range': {
            'start': str(df['hour_utc'].min()),
            'end': str(df['hour_utc'].max())
        }
    }
    
    # ────────────────────────────────────────────────────────────────
    # STEP 3: Graph Construction
    # ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 3: GRAPH CONSTRUCTION")
    print("="*70)
    
    graph_data = build_and_save_graph(df)
    
    results['graph'] = {
        'num_nodes': graph_data.num_nodes,
        'num_edges': graph_data.edge_index.shape[1],
        'num_features': graph_data.x.shape[1],
        'train_nodes': graph_data.train_mask.sum().item(),
        'val_nodes': graph_data.val_mask.sum().item(),
        'test_nodes': graph_data.test_mask.sum().item()
    }
    
    # ────────────────────────────────────────────────────────────────
    # STEP 4: Model Training
    # ────────────────────────────────────────────────────────────────
    if not skip_training:
        print("\n" + "="*70)
        print("STEP 4: MODEL TRAINING")
        print("="*70)
        
        model, metrics, history = train_gnn_model(
            graph_data, 
            model_type=model_type,
            config=GNN_CONFIG
        )
        
        results['training'] = {
            'model_type': model_type,
            'metrics': metrics,
            'best_epoch': len(history['train_loss']),
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
    else:
        print("\n⏭️  Skipping model training")
        history = None
    
    # ────────────────────────────────────────────────────────────────
    # STEP 5: Prediction & Evaluation
    # ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 5: PREDICTION & EVALUATION")
    print("="*70)
    
    predictor = GNNPredictor()
    
    # Test set predictions
    test_predictions = predictor.predict_test_set()
    test_metrics = evaluate_predictions(test_predictions)
    
    results['evaluation'] = test_metrics
    
    # Future forecast
    future_predictions = predictor.predict_future(num_hours=24)
    cheapest_hours = predictor.get_cheapest_hours(future_predictions, top_n=3)
    most_expensive_hours = predictor.get_most_expensive_hours(future_predictions, top_n=3)
    
    results['forecast'] = {
        'next_24h': future_predictions.to_dict('records'),
        'cheapest_hours': cheapest_hours.to_dict('records'),
        'most_expensive_hours': most_expensive_hours.to_dict('records')
    }
    
    print("\n💰 CHEAPEST HOURS (Next 24h):")
    print(cheapest_hours)
    
    print("\n💸 MOST EXPENSIVE HOURS (Next 24h):")
    print(most_expensive_hours)
    
    # ────────────────────────────────────────────────────────────────
    # STEP 6: Visualization
    # ────────────────────────────────────────────────────────────────
    if visualize and history is not None:
        print("\n" + "="*70)
        print("STEP 6: VISUALIZATION")
        print("="*70)
        
        # Training history
        plot_training_history(
            history, 
            save_path=ARTIFACTS_DIR / 'training_history.png'
        )
        
        # Predictions vs actual
        plot_predictions_vs_actual(
            test_predictions,
            save_path=ARTIFACTS_DIR / 'predictions_vs_actual.png'
        )
        
        # Error distribution
        plot_error_distribution(
            test_predictions,
            save_path=ARTIFACTS_DIR / 'error_distribution.png'
        )
        
        # Hourly performance
        plot_hourly_performance(
            test_predictions,
            save_path=ARTIFACTS_DIR / 'hourly_performance.png'
        )
        
        # Future forecast
        plot_future_forecast(
            future_predictions,
            save_path=ARTIFACTS_DIR / 'future_forecast.png'
        )
    
    # ────────────────────────────────────────────────────────────────
    # Save Results
    # ────────────────────────────────────────────────────────────────
    results['timestamp'] = datetime.now().isoformat()
    results['config'] = {
        'price_zone': price_zone,
        'model_type': model_type,
        'gnn_config': GNN_CONFIG
    }
    
    results_path = ARTIFACTS_DIR / 'pipeline_results.json'
    with open(results_path, 'w') as f:
        # Convert non-serializable objects
        results_serializable = json.loads(
            json.dumps(results, default=str, indent=2)
        )
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n💾 Saved pipeline results to {results_path}")
    
    # ────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("📊 PIPELINE SUMMARY")
    print("="*70)
    
    if 'evaluation' in results:
        print(f"Test MAE:  {results['evaluation']['MAE']:.6f} DKK/kWh")
        print(f"Test RMSE: {results['evaluation']['RMSE']:.6f} DKK/kWh")
        print(f"Test R²:   {results['evaluation']['R2']:.6f}")
        print(f"Test MAPE: {results['evaluation']['MAPE']:.2f}%")
    
    print("\n✅ Pipeline completed successfully!")
    print("="*70)
    
    return results


def compare_models(price_zone=DEFAULT_ZONE):
    """
    Compare different GNN architectures.
    
    Args:
        price_zone: Price zone to forecast.
    
    Returns:
        Dictionary with comparison results.
    """
    print("\n" + "="*70)
    print("🔍 COMPARING GNN MODELS")
    print("="*70)
    
    # Load graph
    graph_data = load_graph()
    
    model_types = ['GCN', 'GAT', 'GraphSAGE', 'GIN']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Training {model_type}...")
        print(f"{'='*70}")
        
        try:
            model, metrics, history = train_gnn_model(
                graph_data,
                model_type=model_type,
                config=GNN_CONFIG
            )
            
            results[model_type] = metrics
            
        except Exception as e:
            print(f"❌ Failed to train {model_type}: {e}")
            continue
    
    # Save comparison results
    comparison_path = ARTIFACTS_DIR / 'model_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved comparison results to {comparison_path}")
    
    # Print comparison table
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)
    
    print(f"{'Model':<15} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'MAPE (%)':<12}")
    print("-" * 70)
    
    for model_type, metrics in results.items():
        print(f"{model_type:<15} "
              f"{metrics['test_mae']:<12.6f} "
              f"{metrics['test_rmse']:<12.6f} "
              f"{metrics['test_r2']:<12.6f} "
              f"{metrics['test_mape']:<12.2f}")
    
    # Visualize comparison
    from gnn_visualization import plot_model_comparison
    plot_model_comparison(
        results,
        save_path=ARTIFACTS_DIR / 'model_comparison.png'
    )
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='GNN-based electricity price forecasting pipeline'
    )
    
    parser.add_argument(
        '--zone',
        type=str,
        default=DEFAULT_ZONE,
        choices=['DK1', 'DK2'],
        help='Price zone to forecast'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='GCN',
        choices=['GCN', 'GAT', 'GraphSAGE', 'GIN'],
        help='GNN model type'
    )
    
    parser.add_argument(
        '--skip-ingestion',
        action='store_true',
        help='Skip data ingestion step'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training step'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip visualization step'
    )
    
    parser.add_argument(
        '--compare-models',
        action='store_true',
        help='Compare different GNN architectures'
    )
    
    args = parser.parse_args()
    
    if args.compare_models:
        compare_models(price_zone=args.zone)
    else:
        run_full_pipeline(
            price_zone=args.zone,
            model_type=args.model,
            skip_ingestion=args.skip_ingestion,
            skip_training=args.skip_training,
            visualize=not args.no_visualize
        )