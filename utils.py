"""
Crypto Prediction - Utility Functions
=====================================

This module provides helper functions for logging setup, random seed seeding,
and other common tasks shared across the project.
"""

import random
import numpy as np
import torch
import logging
import sys
import os

def setup_logging(log_file: str = "app.log") -> logging.Logger:
    """
    Sets up the logging configuration.
    
    Args:
        log_file (str): Path to the log file.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger("CryptoPrediction")
    
    # Check if handlers already exist to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger

def set_seeds(seed: int = 42):
    """
    Sets random seeds for reproducibility.
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seeds set to {seed}")

def ensure_directories(paths: list):
    """
    Ensures that the specified directories exist.
    
    Args:
        paths (list): List of directory paths.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
