"""
Crypto Prediction - Visualization Module
========================================

This module contains functions for generating plots and charts to visualize
training progress, model predictions, and strategy performance.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
import os
import logging

logger = logging.getLogger("CryptoPrediction")

def set_plot_style():
    """
    Sets the matplotlib and seaborn style for the project using Purple and Orange tones.
    
    Returns:
        tuple: (selected_palette, div_cmap, gradient_cmap)
    """
    plt.style.use('ggplot')
    
    # Custom Purple and Orange color cycle
    # 1. Deep Purple (Main)
    # 2. Dark Orange (Contrast)
    # 3. Medium Purple (Secondary)
    # 4. Orange (Secondary Contrast)
    # 5. Light Purple
    # 6. Light Orange
    selected_colors = [
        "#4B0082",  # Indigo/Deep Purple
        "#FF8C00",  # Dark Orange
        "#9370DB",  # Medium Purple
        "#FFA500",  # Orange
        "#BA55D3",  # Medium Orchid
        "#FFD700",  # Gold
        "#D8BFD8",  # Thistle (Light Purple)
        "#FFE4B5",  # Moccasin
    ]
    
    selected_color_cycle = cycler('color', selected_colors)
    selected_palette = sns.color_palette(selected_colors)
    
    mpl.rcParams['axes.prop_cycle'] = selected_color_cycle
    sns.set_palette(selected_palette)
    
    # Custom colormaps
    # Diverging: Purple -> White -> Orange
    div_cmap = LinearSegmentedColormap.from_list(
        "po_div", ["#4B0082", "#FFFFFF", "#FF8C00"], N=256
    )
    
    # Gradient: Purple to Orange
    gradient_cmap = LinearSegmentedColormap.from_list(
        "po_gradient", ["#4B0082", "#FF8C00"], N=256
    )
    
    # Gradient: Purple only (for sequential)
    purple_cmap = LinearSegmentedColormap.from_list(
        "purple_grad", ["#E6E6FA", "#4B0082"], N=256
    )
    
    return selected_palette, div_cmap, gradient_cmap, purple_cmap

def save_plot(save_path, filename):
    """
    Helper to save plots to a directory.
    """
    if not save_path:
        return
    
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {full_path}")
    plt.show()
