"""
Crypto Prediction - Model Architecture
======================================

This module defines the Deep Learning architecture (LSTM) used for price direction prediction.
"""

import torch
import torch.nn as nn
import os
import logging
import numpy as np

logger = logging.getLogger("CryptoPrediction")

class CryptoLSTMClassifier(nn.Module):
    """
    Simple LSTM classifier for cryptocurrency price direction prediction.
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2):
        """
        Initializes the Simple LSTM.
        """
        super(CryptoLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Explicit Dropout Layer
        self.dropout = nn.Dropout(dropout)
        
        # Final Fully Connected Layer
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass Simple LSTM.
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass LSTM
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Use only the last time step feature
        out = out[:, -1, :]
        
        # Apply Dropout
        out = self.dropout(out)
        
        # Pass to linear layer
        out = self.fc(out)
        return out

    def save(self, filepath: str):
        """
        Saves the model state dictionary to a file.
        
        Args:
            filepath (str): Path to save the model.
        """
        torch.save(self.state_dict(), filepath)
        logger.debug(f"Model saved to {filepath}")
            
    def load(self, filepath: str, device: torch.device):
        """
        Loads the model state dictionary from a file.
        
        Args:
            filepath (str): Path to load the model from.
            device (torch.device): Device to load the model onto.
        """
        if os.path.exists(filepath):
            self.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
            self.to(device)
            logger.info(f"Model loaded from {filepath}")
        else:
            logger.warning(f"Model file not found: {filepath}")

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss) # pt is the probability of being right
        
        # Apply class weighting (alpha)
        # alpha_t = alpha if target=1, else (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CryptoLinearClassifier(nn.Module):
    """
    Simple Linear Classifier (Logistic Regression equivalent via SGD).
    """
    def __init__(self, input_size: int, seq_length: int, **kwargs):
        super(CryptoLinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size * seq_length, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq, Feat) -> Flatten -> (Batch, Seq*Feat)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class RandomForestWrapper:
    """
    Wraps a sklearn Random Forest to mimic a PyTorch model for the Evaluator.
    """
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type

    def __call__(self, x):
        # x is tensor (Batch, Seq, Feat)
        x_np = x.cpu().numpy()
        # Flatten: (Batch, Seq*Feat)
        x_flat = x_np.reshape(x_np.shape[0], -1)
        
        # Return probs of class 1
        out = self.model.predict_proba(x_flat)[:, 1]
             
        # Evaluator expects tensor output
        return torch.tensor(out, dtype=torch.float32).to(x.device)
    
    def eval(self):
        pass
    
    def train(self):
        pass
    
    def to(self, device):
        return self
