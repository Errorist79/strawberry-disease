"""
Custom metrics logger callback.

Logs detailed metrics during training for better monitoring.
"""

import json
from pathlib import Path
from typing import Dict, Optional


class MetricsLogger:
    """Log detailed metrics during training."""

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory to save logs (default: runs/metrics)
        """
        self.log_dir = Path(log_dir) if log_dir else Path("runs/metrics")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = []
        self.current_epoch = 0

    def on_train_epoch_end(self, trainer):
        """Called at end of training epoch."""
        # Extract metrics from trainer
        metrics = {}

        if hasattr(trainer, "metrics"):
            metrics = trainer.metrics

        # Add epoch number
        metrics["epoch"] = self.current_epoch
        self.metrics_history.append(metrics)

        # Save to file
        self._save_metrics()

        self.current_epoch += 1

    def _save_metrics(self):
        """Save metrics to JSON file."""
        log_file = self.log_dir / "metrics_history.json"

        with open(log_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
