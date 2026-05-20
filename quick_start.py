"""
Quick Start Example - Simple script to test the GNN forecasting system

This script demonstrates basic usage:
1. Fetch data
2. Build graph
3. Train model
4. Make predictions
"""

import warnings
warnings.filterwarnings('ignore')

from gnn_data_ingestion import run_ingestion
from gnn_feature_engineering import prepare_data_for_gnn
from gnn_graph_builder import build_and_save_graph
from gnn_train import train_gnn_model
from gnn_predict import GNNPredictor, evaluate_predictions


def quick_start():
    """Run a quick demonstration of the GNN forecasting system."""
    
    print("="*70)
    print("🚀 QUICK START: GNN ELECTRICITY PRICE FORECASTING")
    print("="*70)
    
    # ────────────────────────────────────────────────────────────────
    # 1. Fetch Data
    # ────────────────────────────────────────────────────────────────
    print("\n📡 Step 1: Fetching data (this may take a few minutes)...")
    try:
        prices_df = run_ingestion(days_back=30)  # Fetch 30 days for quick demo
        print(f"✅ Fetched {len(prices_df)} price records")
    except Exception as e:
        print(f"❌ Data fetching failed: {e}")
        return
    
    # ────────────────────────────────────────────────────────────────
    # 2. Prepare Features
    # ────────────────────────────────────────────────────────────────
    print("\n🔧 Step 2: Preparing features...")
    try:
        df = prepare_data_for_gnn(price_zone='DK1')
        if df is None:
            print("❌ Feature preparation failed")
            return
        print(f"✅ Prepared {len(df)} samples with features")
    except Exception as e:
        print(f"❌ Feature preparation failed: {e}")
        return
    
    # ────────────────────────────────────────────────────────────────
    # 3. Build Graph
    # ────────────────────────────────────────────────────────────────
    print("\n🔗 Step 3: Building temporal graph...")
    try:
        graph_data = build_and_save_graph(df)
        print(f"✅ Built graph with {graph_data.num_nodes} nodes and {graph_data.edge_index.shape[1]} edges")
    except Exception as e:
        print(f"❌ Graph building failed: {e}")
        return
    
    # ────────────────────────────────────────────────────────────────
    # 4. Train Model (Quick training with fewer epochs)
    # ────────────────────────────────────────────────────────────────
    print("\n🎓 Step 4: Training GNN model (this will take a few minutes)...")
    try:
        # Use smaller config for quick demo
        quick_config = {
            'hidden_channels': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'conv_type': 'GCN',
            'learning_rate': 0.001,
            'weight_decay': 5e-4,
            'num_epochs': 30,  # Fewer epochs for quick demo
            'early_stopping_patience': 10,
        }
        
        model, metrics, history = train_gnn_model(
            graph_data, 
            model_type='GCN',
            config=quick_config
        )
        
        print(f"✅ Training complete!")
        print(f"   Test MAE: {metrics['test_mae']:.4f} DKK/kWh")
        print(f"   Test R²: {metrics['test_r2']:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # ────────────────────────────────────────────────────────────────
    # 5. Make Predictions
    # ────────────────────────────────────────────────────────────────
    print("\n🔮 Step 5: Making predictions...")
    try:
        predictor = GNNPredictor()
        
        # Predict test set
        test_predictions = predictor.predict_test_set()
        print(f"✅ Generated {len(test_predictions)} test predictions")
        
        # Predict future
        future_predictions = predictor.predict_future(num_hours=24)
        print(f"✅ Generated 24-hour forecast")
        
        # Find cheapest hours
        cheapest = predictor.get_cheapest_hours(future_predictions, top_n=3)
        
        print("\n" + "="*70)
        print("💰 CHEAPEST HOURS (Next 24h):")
        print("="*70)
        for idx, row in cheapest.iterrows():
            print(f"   {row['timestamp'].strftime('%Y-%m-%d %H:%M')} - "
                  f"{row['predicted_price']:.4f} DKK/kWh")
        
        # Statistics
        print("\n" + "="*70)
        print("📊 PRICE STATISTICS (Next 24h):")
        print("="*70)
        print(f"   Average: {future_predictions['predicted_price'].mean():.4f} DKK/kWh")
        print(f"   Minimum: {future_predictions['predicted_price'].min():.4f} DKK/kWh")
        print(f"   Maximum: {future_predictions['predicted_price'].max():.4f} DKK/kWh")
        print(f"   Std Dev: {future_predictions['predicted_price'].std():.4f} DKK/kWh")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    
    # ────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("✅ QUICK START COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run full pipeline: python gnn_pipeline.py")
    print("2. Compare models: python gnn_pipeline.py --compare-models")
    print("3. Customize config in gnn_config.py")
    print("4. Explore visualizations in artifacts/ directory")
    print("="*70)


if __name__ == "__main__":
    quick_start()