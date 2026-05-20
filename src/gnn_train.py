"""
Training Module - Train GNN models for electricity price forecasting
"""
import pickle
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from gnn_models import create_model
from gnn_config import GNN_CONFIG, DEVICE, ARTIFACTS_DIR, GRAPH_DIR


class GNNTrainer:
    """Trainer for GNN models."""
    
    def __init__(self, model, data, config=None):
        """
        Initialize trainer.
        
        Args:
            model: GNN model instance.
            data: PyTorch Geometric Data object.
            config: Training configuration dictionary.
        """
        self.model = model.to(DEVICE)
        self.data = data.to(DEVICE)
        self.config = config or GNN_CONFIG
        
        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(self.data.x, self.data.edge_index)
        
        # Compute loss on training nodes
        loss = F.mse_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute MAE
        with torch.no_grad():
            mae = F.l1_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        
        return loss.item(), mae.item()
    
    @torch.no_grad()
    def evaluate(self, mask):
        """Evaluate model on given mask."""
        self.model.eval()
        
        out = self.model(self.data.x, self.data.edge_index)
        
        loss = F.mse_loss(out[mask], self.data.y[mask])
        mae = F.l1_loss(out[mask], self.data.y[mask])
        
        return loss.item(), mae.item()
    
    def train(self, num_epochs=None, early_stopping_patience=None):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train.
            early_stopping_patience: Patience for early stopping.
        
        Returns:
            Dictionary with training history.
        """
        num_epochs = num_epochs or self.config['num_epochs']
        early_stopping_patience = early_stopping_patience or self.config['early_stopping_patience']
        
        print(f"\n🚀 Starting training for {num_epochs} epochs...")
        print(f"Device: {DEVICE}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_mae = self.train_epoch()
            
            # Validate
            val_loss, val_mae = self.evaluate(self.data.val_mask)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                # Save best model
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train MAE: {train_mae:.4f} | "
                      f"Val MAE: {val_mae:.4f}")
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                break
        
        print(f"\n✅ Training complete!")
        print(f"Best epoch: {self.best_epoch}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def test(self):
        print("\n📊 Evaluating on test set...")
        
        # Load best model
        self.load_checkpoint('best_model.pt')
        
        # Load the target scaler to reverse normalization
        scaler_path = GRAPH_DIR / "scaler.pkl"
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            target_scaler = scalers['target_scaler']
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            
            # Extract test predictions and targets (still in scaled space)
            y_pred_scaled = out[self.data.test_mask].cpu().numpy().reshape(-1, 1)
            y_true_scaled = self.data.y[self.data.test_mask].cpu().numpy().reshape(-1, 1)
            
            # Inverse transform back to raw DKK/kWh prices
            y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
            y_true = target_scaler.inverse_transform(y_true_scaled).flatten()
            
            # Calculate standard baseline metrics on real prices
            mae = np.mean(np.abs(y_pred - y_true))
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            
            # R² Score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            # Robust SMAPE calculation to handle zero/negative pricing safely
            smape = np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
        metrics = {
            'test_mae': float(mae),
            'test_rmse': float(rmse),
            'test_r2': float(r2),
            'test_mape': float(smape)
        }
        
        print(f"\n📈 Real-Scale Test Results:")
        print(f"   MAE:  {mae:.4f} DKK/kWh")
        print(f"   RMSE: {rmse:.4f} DKK/kWh")
        print(f"   R²:   {r2:.4f}")
        print(f"   SMAPE: {smape:.2f}%")
        
        return metrics
         
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint_path = ARTIFACTS_DIR / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }, checkpoint_path)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint_path = ARTIFACTS_DIR / filename
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
    
    def save_history(self, filename='training_history.json'):
        """Save training history to JSON."""
        history_path = ARTIFACTS_DIR / filename
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"💾 Saved training history to {history_path}")


def train_gnn_model(data, model_type='GCN', config=None):
    """
    Train a GNN model.
    
    Args:
        data: PyTorch Geometric Data object.
        model_type: Type of GNN model.
        config: Training configuration.
    
    Returns:
        Trained model and metrics.
    """
    config = config or GNN_CONFIG
    
    # Create model
    num_features = data.x.shape[1]
    model = create_model(model_type, num_features, config)
    
    print(f"\n🔧 Created {model_type} model")
    print(f"   Input features: {num_features}")
    print(f"   Hidden channels: {config['hidden_channels']}")
    print(f"   Number of layers: {config['num_layers']}")
    print(f"   Dropout: {config['dropout']}")
    
    # Create trainer
    trainer = GNNTrainer(model, data, config)
    
    # Train
    history = trainer.train()
    
    # Test
    metrics = trainer.test()
    
    # Save history
    trainer.save_history()
    
    return model, metrics, history


if __name__ == "__main__":
    from gnn_graph_builder import load_graph
    
    # Load graph
    print("📂 Loading graph...")
    data = load_graph()
    
    # Train model
    model, metrics, history = train_gnn_model(data, model_type='GCN')
    
    print("\n✅ Training pipeline complete!")