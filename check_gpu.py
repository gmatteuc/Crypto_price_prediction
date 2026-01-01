"""
Crypto Prediction - GPU Check Utility
=====================================

This script verifies the availability of CUDA-enabled GPUs and checks
the PyTorch installation status for hardware acceleration.
"""

import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available.")
    print("This is likely because the installed PyTorch version is CPU-only or CUDA drivers are missing.")
