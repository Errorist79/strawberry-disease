"""
Visualization utilities for training results.

Provides functions to plot training curves, confusion matrices, etc.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(
    metrics_file: Path,
    output_path: Optional[Path] = None,
    metrics_to_plot: Optional[List[str]] = None,
):
    """
    Plot training curves from metrics file.

    Args:
        metrics_file: Path to metrics JSON file
        output_path: Path to save plot
        metrics_to_plot: List of metric names to plot
    """
    import json

    # Load metrics
    with open(metrics_file, "r") as f:
        metrics_history = json.load(f)

    if not metrics_history:
        print("No metrics to plot")
        return

    # Default metrics
    if metrics_to_plot is None:
        metrics_to_plot = ["map50", "map50_95", "precision", "recall"]

    # Extract data
    epochs = [m.get("epoch", idx) for idx, m in enumerate(metrics_history)]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metrics_to_plot):
        if idx >= len(axes):
            break

        values = [m.get(metric_name, 0) for m in metrics_history]

        axes[idx].plot(epochs, values, marker="o", linewidth=2)
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel(metric_name.upper())
        axes[idx].set_title(f"{metric_name.upper()} vs Epoch")
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Training curves saved: {output_path}")
    else:
        plt.show()


def create_confusion_matrix(
    results_dir: Path,
    output_path: Optional[Path] = None,
):
    """
    Create confusion matrix from YOLO results.

    Args:
        results_dir: Path to YOLO results directory
        output_path: Path to save plot
    """
    # Look for confusion matrix in results
    cm_path = results_dir / "confusion_matrix.png"

    if cm_path.exists():
        print(f"Confusion matrix already exists: {cm_path}")
        if output_path and output_path != cm_path:
            import shutil

            shutil.copy(cm_path, output_path)
    else:
        print(f"No confusion matrix found in {results_dir}")
