# electricity-forecasting
# GNN-Based Electricity Price Forecasting for Denmark

A complete Graph Neural Network (GNN) system for forecasting electricity prices in Denmark using temporal graph structures and real-time data from free APIs.

## 🌟 Features

- **Homogeneous Temporal Graph**: Each hour is a node with multi-hop temporal connections
- **Multiple GNN Architectures**: GCN, GAT, GraphSAGE, and GIN
- **Real-time Data**: Fetches electricity prices and weather data from free APIs
- **Complete Pipeline**: From data ingestion to prediction and visualization
- **No API Keys Required**: Uses free public APIs (Energy-Charts and Open-Meteo)

## 📊 Graph Structure

### Nodes
- **Type**: Hourly timesteps
- **Features** (19 dimensions):
  - Temporal: hour, day_of_week, month, is_weekend
  - Lag features: 1h, 2h, 24h, 48h, 168h price lags
  - Rolling statistics: 6h, 12h, 24h mean and std
  - Weather: temperature, wind speed, cloud cover, humidity

### Edges
- **Temporal 1h**: Connect consecutive hours (t → t+1)
- **Temporal 24h**: Connect same hour previous day (t → t+24)
- **Temporal 168h**: Connect same hour previous week (t → t+168)
- **Temporal 2h & 6h**: Additional temporal connections
- **Bidirectional**: All edges are bidirectional

## 🏗️ Architecture

```
Data Sources (Energy-Charts + Open-Meteo)
            ↓
    SQLite Database
            ↓
  Feature Engineering
            ↓
   Graph Construction
            ↓
   GNN Model Training
    (GCN/GAT/SAGE/GIN)
            ↓
 Predictions & Forecasting
            ↓
    Visualization
```

## 📦 Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd gnn-electricity-forecast
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch Geometric
```bash
# CPU version
pip install torch-geometric

# For GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 🚀 Quick Start

### Run Complete Pipeline
```bash
python gnn_pipeline.py --zone DK1 --model GCN
```

### Step-by-Step Execution

#### 1. Data Ingestion (90 days of historical data)
```bash
python gnn_data_ingestion.py
```

#### 2. Build Graph
```bash
python gnn_graph_builder.py
```

#### 3. Train Model
```bash
python gnn_train.py
```

#### 4. Make Predictions
```bash
python gnn_predict.py
```

### Compare Models
```bash
python gnn_pipeline.py --compare-models --zone DK1
```

## 📁 Project Structure

```
gnn-electricity-forecast/
├── gnn_config.py              # Configuration settings
├── gnn_database.py            # Database operations
├── gnn_data_ingestion.py      # Fetch data from APIs
├── gnn_feature_engineering.py # Create node features
├── gnn_graph_builder.py       # Construct temporal graph
├── gnn_models.py              # GNN architectures
├── gnn_train.py               # Training pipeline
├── gnn_predict.py             # Prediction and forecasting
├── gnn_visualization.py       # Plotting functions
├── gnn_pipeline.py            # Main pipeline script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/                      # Data directory (auto-created)
│   ├── energy.db             # SQLite database
│   └── graphs/               # Saved graph objects
└── artifacts/                 # Model outputs (auto-created)
    ├── best_model.pt         # Best trained model
    ├── training_history.json # Training metrics
    ├── pipeline_results.json # Pipeline results
    └── *.png                 # Visualization plots
```

## 🔧 Configuration

Edit `gnn_config.py` to customize:

```python
# Model architecture
GNN_CONFIG = {
    "hidden_channels": 128,    # Hidden layer size
    "num_layers": 3,           # Number of GNN layers
    "dropout": 0.2,            # Dropout rate
    "conv_type": "GCN",        # GCN, GAT, GraphSAGE, GIN
    "num_heads": 4,            # For GAT only
    
    # Training
    "learning_rate": 0.001,
    "weight_decay": 5e-4,
    "num_epochs": 100,
    "early_stopping_patience": 15,
}

# Graph edges
EDGE_TYPES = {
    "temporal_1h": 1,
    "temporal_24h": 24,
    "temporal_168h": 168,
    "temporal_2h": 2,
    "temporal_6h": 6,
}
```

## 📈 Model Performance

### Expected Metrics
- **MAE**: ~0.04-0.08 DKK/kWh
- **RMSE**: ~0.06-0.12 DKK/kWh
- **R²**: 0.90-0.97
- **MAPE**: 10-25%

### Model Comparison
Different architectures offer trade-offs:
- **GCN**: Fast, good baseline performance
- **GAT**: Attention mechanism, better for complex patterns
- **GraphSAGE**: Sampling-based, scales well
- **GIN**: Most expressive, best for capturing graph structure

## 🎯 Use Cases

### 1. Price Forecasting
```python
from gnn_predict import GNNPredictor

predictor = GNNPredictor()
future_forecast = predictor.predict_future(num_hours=24)
print(future_forecast)
```

### 2. Find Cheapest Hours
```python
cheapest = predictor.get_cheapest_hours(future_forecast, top_n=3)
print(f"Best time to charge EV: {cheapest}")
```

### 3. Model Evaluation
```python
test_predictions = predictor.predict_test_set()
from gnn_predict import evaluate_predictions
metrics = evaluate_predictions(test_predictions)
```

## 📊 Visualization

The pipeline generates several plots:
- **Training History**: Loss and MAE curves
- **Predictions vs Actual**: Time series comparison
- **Error Distribution**: Histogram and box plot
- **Hourly Performance**: MAE by hour of day
- **Future Forecast**: 24-hour ahead prediction
- **Model Comparison**: Compare GNN architectures

## 🔄 Data Sources

### 1. Energy-Charts API (Fraunhofer ISE)
- **URL**: https://api.energy-charts.info/
- **Data**: Spot electricity prices for DK1/DK2
- **Format**: EUR/MWh (converted to DKK/kWh)
- **Update**: Hourly
- **Cost**: Free, no API key

### 2. Open-Meteo
- **URL**: https://open-meteo.com/
- **Data**: Weather (temperature, wind, cloud cover, humidity)
- **Format**: Hourly time series
- **Coverage**: Historical archive + 2-day forecast
- **Cost**: Free, no API key

## 🧠 GNN Models

### Graph Convolutional Network (GCN)
```python
# Simple and efficient
# Uses mean aggregation
model = create_model('GCN', num_features, config)
```

### Graph Attention Network (GAT)
```python
# Learns edge importance
# Multi-head attention mechanism
model = create_model('GAT', num_features, config)
```

### GraphSAGE
```python
# Sampling-based approach
# Scales to large graphs
model = create_model('GraphSAGE', num_features, config)
```

### Graph Isomorphism Network (GIN)
```python
# Most expressive GNN
# Best for structural patterns
model = create_model('GIN', num_features, config)
```

## 🛠️ Command Line Arguments

```bash
python gnn_pipeline.py --help

Options:
  --zone {DK1,DK2}              Price zone (default: DK1)
  --model {GCN,GAT,GraphSAGE,GIN}  GNN architecture (default: GCN)
  --skip-ingestion              Skip data fetch if already present
  --skip-training               Skip training if model exists
  --no-visualize                Don't generate plots
  --compare-models              Train and compare all architectures
```

## 🔬 Advanced Usage

### Custom Graph Structure
```python
from gnn_graph_builder import TemporalGraphBuilder

builder = TemporalGraphBuilder()
graph = builder.build_graph(df, train_mask_ratio=0.7)

# Modify edge types
EDGE_TYPES = {
    "temporal_1h": 1,
    "temporal_12h": 12,
    "temporal_24h": 24,
}
```

### Hyperparameter Tuning
```python
from gnn_train import train_gnn_model

config = {
    'hidden_channels': 256,
    'num_layers': 4,
    'dropout': 0.3,
    'learning_rate': 0.0005,
    'num_epochs': 150,
}

model, metrics, history = train_gnn_model(
    graph_data, 
    model_type='GAT',
    config=config
)
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{gnn_electricity_forecast,
  title={GNN-Based Electricity Price Forecasting for Denmark},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/gnn-electricity-forecast}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Energy-Charts API** by Fraunhofer ISE
- **Open-Meteo** for weather data
- **PyTorch Geometric** team for the GNN library
- **Nord Pool** for electricity market data

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]

## 🔮 Future Enhancements

- [ ] Add attention visualization
- [ ] Implement multi-step ahead forecasting
- [ ] Add heterogeneous graph (multiple node types)
- [ ] Real-time streaming predictions
- [ ] REST API for predictions
- [ ] Web dashboard with Streamlit
- [ ] Docker containerization
- [ ] Add more weather features
- [ ] Include demand forecasting
- [ ] Multi-zone joint modeling

## 📚 References

1. Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks
2. Veličković et al. (2018). Graph Attention Networks
3. Hamilton et al. (2017). Inductive Representation Learning on Large Graphs
4. Xu et al. (2019). How Powerful are Graph Neural Networks?