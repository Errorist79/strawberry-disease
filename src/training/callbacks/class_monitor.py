"""
Class performance monitoring callback.

Monitors per-class performance to detect overfitting or poor performance
on specific classes.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class ClassPerformanceMonitor:
    """Monitor per-class performance during training."""

    def __init__(
        self,
        class_names: List[str],
        log_dir: Optional[Path] = None,
        alert_threshold: float = 0.3,
    ):
        """
        Initialize class performance monitor.

        Args:
            class_names: List of class names
            log_dir: Directory to save logs
            alert_threshold: Alert if class performance drops below this
        """
        self.class_names = class_names
        self.log_dir = Path(log_dir) if log_dir else Path("runs/class_metrics")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.alert_threshold = alert_threshold

        self.class_history = {name: [] for name in class_names}
        self.current_epoch = 0

    def on_val_end(self, trainer):
        """Called at end of validation."""
        # Extract per-class metrics if available
        if hasattr(trainer, "metrics") and hasattr(trainer.metrics, "per_class"):
            per_class = trainer.metrics.per_class

            for idx, class_name in enumerate(self.class_names):
                if idx < len(per_class):
                    metrics = {
                        "epoch": self.current_epoch,
                        "map50": float(per_class[idx].get("map50", 0)),
                        "precision": float(per_class[idx].get("precision", 0)),
                        "recall": float(per_class[idx].get("recall", 0)),
                    }

                    self.class_history[class_name].append(metrics)

                    # Check for alerts
                    if metrics["map50"] < self.alert_threshold:
                        print(
                            f"⚠️  Alert: {class_name} performance below threshold "
                            f"(mAP50: {metrics['map50']:.3f})"
                        )

        self._save_metrics()
        self.current_epoch += 1

    def _save_metrics(self):
        """Save class metrics to JSON."""
        log_file = self.log_dir / "class_performance.json"

        with open(log_file, "w") as f:
            json.dump(self.class_history, f, indent=2)
