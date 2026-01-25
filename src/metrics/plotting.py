"""
Shared plotting utilities for metric visualization.
"""

import matplotlib.pyplot as plt
from typing import Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TrainingInfo:
    """Information about symmetry penalty used during training."""
    penalty_type: Optional[str] = None
    layers: Optional[List[int]] = None
    lambda_sym: float = 0.0
    
    def format_subtitle(self) -> Optional[str]:
        """Format training info as subtitle string, or None if no penalty was used."""
        if self.penalty_type and self.lambda_sym > 0 and self.layers:
            layers_str = str(self.layers).replace(' ', '')
            return f'Model trained with {self.penalty_type} Penalty (layers={layers_str}, λ={self.lambda_sym})'
        return None


def plot_metric_vs_layer(
    values: Dict[str, float],
    metric_name: str,
    save_path: Path = None,
    color: str = 'steelblue',
    ylabel: str = None,
    oracle_value: float = None,
    training_info: TrainingInfo = None,
    log_eps: float = 1e-6,
):
    """
    Generic function to plot a metric as a function of layer depth.
    
    Args:
        values: Dictionary mapping layer names to metric values.
        metric_name: Name of the metric (used in title).
        save_path: Optional path to save the plot.
        color: Bar color for the metric.
        ylabel: Y-axis label (defaults to metric_name).
        oracle_value: Optional oracle value to show as additional bar.
        training_info: Optional training penalty information for subtitle.
        log_eps: Epsilon for log scale (minimum value).
    """
    layers = list(values.keys())
    vals = list(values.values())
    
    # Add oracle bar if provided
    if oracle_value is not None:
        layers = layers + ['oracle']
        vals = vals + [oracle_value]
        colors = [color] * (len(layers) - 1) + ['green']
    else:
        colors = [color] * len(layers)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(layers))
    
    # Left: linear scale
    axes[0].bar(x, vals, color=colors, edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layers, rotation=45, ha='right')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel(ylabel or metric_name)
    axes[0].set_ylim(bottom=0)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_title('Linear')
    
    # Right: log scale
    log_vals = [max(v, log_eps) for v in vals]
    axes[1].bar(x, log_vals, color=colors, edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers, rotation=45, ha='right')
    axes[1].set_xlabel('Layer')
    axes[1].set_yscale('log')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_title('Log')
    
    # Set overall title with optional training info subtitle
    title = f'{metric_name} by Layer'
    if training_info:
        subtitle = training_info.format_subtitle()
        if subtitle:
            title += f': {subtitle}'
    fig.suptitle(title)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
