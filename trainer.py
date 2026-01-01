import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple
import visualization
from model import FocalLoss

logger = logging.getLogger("CryptoPrediction")

class Trainer:
    """
    Handles the training and validation loop for the model.
    """
    def __init__(self, model: nn.Module, config, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        # Use BCEWithLogitsLoss with pos_weight to handle class imbalance/bias
        # pos_weight < 1 penalizes the positive class (Up) less, effectively prioritizing the negative class (Down)
        # We set pos_weight=0.5 to make "Down" samples 2x more important than "Up" samples
        pos_weight = torch.tensor([0.5]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor):
        """
        Executes the training loop with early stopping.
        """
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        logger.info("Starting training...")
        
        best_val_loss = float('inf')
        counter = 0
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            epoch_loss = 0
            correct_train = 0
            total_train = 0
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y.unsqueeze(1))
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate training accuracy
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                correct_train += (predicted == batch_y.unsqueeze(1)).sum().item()
                total_train += batch_y.size(0)
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_acc = correct_train / total_train
            
            # Validation
            val_loss, val_acc = self.evaluate(X_test, y_test)
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f'Epoch [{epoch+1}/{self.config.EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}')
            
            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # Save best model
                self.model.save(self.config.MODEL_FILE)
            else:
                counter += 1
                if counter >= self.config.PATIENCE:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        logger.info("Training completed.")
        self.plot_loss()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """
        Evaluates the model on a given dataset.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y.unsqueeze(1)).item()
            
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            correct = (predicted == y.unsqueeze(1)).sum().item()
            acc = correct / len(y)
            
        return loss, acc

    def plot_loss(self):
        """
        Plots the training and validation loss.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        horizon_str = "Day" if self.config.PREDICTION_HORIZON == 1 else f"{self.config.PREDICTION_HORIZON} Days"
        plt.title(f'Model Loss (Horizon: {horizon_str})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        visualization.save_plot(self.config.OUTPUT_DIR, 'training_loss.png')
        plt.close()
