"""
Crypto Prediction - Training Loop
=================================

This module manages the model training process, including the training loop,
validation, early stopping, and checkpoint saving.
"""

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
        
        # Default fallback (overridden in train method for Class Weights)
        self.criterion = nn.BCEWithLogitsLoss()
            
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor, y_test: torch.Tensor, 
              class_weights: torch.Tensor = None, sample_weights: torch.Tensor = None):
        """
        Executes the training loop with early stopping.
        Args:
           class_weights (torch.Tensor, optional): Weights for each class (to handle imbalance).
           sample_weights (torch.Tensor, optional): Weights for each sample (e.g. profit weighting).
        """
        if sample_weights is not None:
             train_data = TensorDataset(X_train, y_train, sample_weights)
        else:
             train_data = TensorDataset(X_train, y_train)
             
        train_loader = DataLoader(train_data, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        # Update Criterion with Class Weights if provided
        reduction_mode = 'none' if sample_weights is not None else 'mean'
        
        if class_weights is not None:
            # BCEWithLogitsLoss combines Sigmoid and BCE. 
            # pos_weight should be length 1 (weight for positive class '1')
            # If class_weights has 2 elements [weight_0, weight_1], we can use pos_weight = weight_1 / weight_0
            pos_weight = class_weights[1] / class_weights[0]
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction_mode)
            logger.info(f"Using Weighted BCE Loss. Pos Class Weight: {pos_weight.item():.4f}, Reduction: {reduction_mode}")
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction=reduction_mode)
        
        logger.info(f"Starting training ({self.config.MODEL_TYPE})...")
        
        best_val_loss = float('inf')
        counter = 0
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            epoch_loss = 0
            correct_train = 0
            total_train = 0
            
            for batch in train_loader:
                batch_X = batch[0]
                batch_y = batch[1]
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Calculate Loss
                if sample_weights is not None:
                    batch_w = batch[2]
                    raw_loss = self.criterion(outputs, batch_y.unsqueeze(1))
                    # Weighted Loss = raw_loss * sample_weight
                    loss = (raw_loss * batch_w.unsqueeze(1)).mean()
                else:
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
                if hasattr(self.model, 'save'):
                    self.model.save(self.config.MODEL_FILE)
                else:
                    torch.save(self.model.state_dict(), self.config.MODEL_FILE)
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
            
            # Loss is handled by self.criterion which might be weighted now
            loss_obj = self.criterion(outputs, y.unsqueeze(1))
            
            # Handle reduction='none' case (used when profit weights are active in training)
            if loss_obj.dim() > 0:
                 loss = loss_obj.mean().item()
            else:
                 loss = loss_obj.item()
            
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
        
        visualization.save_plot(self.config.OUTPUT_DIR, f'training_loss_classification.png')
        plt.close()
