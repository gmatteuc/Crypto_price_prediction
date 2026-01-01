"""
Crypto Prediction - Model Architecture
======================================

This module defines the Deep Learning architecture (LSTM) and custom loss functions
(Directional Huber Loss) used for price direction prediction.
"""

import torch
import torch.nn as nn
import os
import logging

logger = logging.getLogger("CryptoPrediction")

class CryptoLSTMClassifier(nn.Module):
    """
    LSTM-based classifier for cryptocurrency price direction prediction.
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2):
        """
        Initializes the LSTM Classifier.
        
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units in LSTM.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout probability.
        """
        super(CryptoLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Handle dropout warning for single layer LSTM
        # PyTorch ignores dropout if num_layers=1, but warns about it.
        # We set it to 0 here and apply it manually after the LSTM.
        lstm_dropout = dropout if num_layers > 1 else 0
        
        # GRU Layer (Switched from LSTM for better performance on noise)
        # batch_first=True means input is (batch, seq, features)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=lstm_dropout)
        
        # Explicit Dropout Layer (applied after LSTM)
        self.dropout = nn.Dropout(dropout)
        
        # Final Fully Connected Layer (Output Logits)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1).
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass GRU
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.gru(x, h0)
        
        # Take only the output of the last time step
        # out[:, -1, :] shape: (batch_size, hidden_size)
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

class CryptoLSTMRegressor(nn.Module):
    """
    LSTM-based regressor for cryptocurrency price return prediction.
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super(CryptoLSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        lstm_dropout = dropout if num_layers > 1 else 0
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=lstm_dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
