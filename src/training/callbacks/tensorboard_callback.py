"""TensorBoard callback for YOLOv8 training."""

from pathlib import Path
from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter


class TensorBoardCallback:
    """
    TensorBoard logging callback for YOLOv8 training.

    Logs training metrics to TensorBoard in real-time.
    """

    def __init__(self, log_dir: str = None):
        """
        Initialize TensorBoard callback.

        Args:
            log_dir: Directory for TensorBoard logs (default: runs/detect/tensorboard)
        """
        self.log_dir = log_dir or "runs/detect/tensorboard"
        self.writer = None

    def on_train_start(self, trainer):
        """Called when training starts."""
        # Create TensorBoard writer
        log_path = Path(trainer.save_dir) / "tensorboard"
        log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_path))
        print(f"📊 TensorBoard logging enabled: {log_path}")
        print(f"   Run: tensorboard --logdir {log_path.parent} --port 6006")

    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch."""
        if self.writer is None:
            return

        epoch = trainer.epoch
        metrics = trainer.metrics

        # Training losses
        if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
            self.writer.add_scalar('Loss/train_box', float(trainer.loss_items[0]), epoch)
            self.writer.add_scalar('Loss/train_cls', float(trainer.loss_items[1]), epoch)
            self.writer.add_scalar('Loss/train_dfl', float(trainer.loss_items[2]), epoch)

        # Validation losses (if available)
        if hasattr(metrics, 'box_loss'):
            self.writer.add_scalar('Loss/val_box', float(metrics.box_loss), epoch)
        if hasattr(metrics, 'cls_loss'):
            self.writer.add_scalar('Loss/val_cls', float(metrics.cls_loss), epoch)
        if hasattr(metrics, 'dfl_loss'):
            self.writer.add_scalar('Loss/val_dfl', float(metrics.dfl_loss), epoch)

        # Metrics
        if hasattr(metrics, 'box'):
            self.writer.add_scalar('Metrics/mAP50', float(metrics.box.map50), epoch)
            self.writer.add_scalar('Metrics/mAP50-95', float(metrics.box.map), epoch)
            self.writer.add_scalar('Metrics/precision', float(metrics.box.mp), epoch)
            self.writer.add_scalar('Metrics/recall', float(metrics.box.mr), epoch)

        # Learning rate
        if hasattr(trainer, 'optimizer'):
            for i, param_group in enumerate(trainer.optimizer.param_groups):
                self.writer.add_scalar(f'LearningRate/pg{i}', param_group['lr'], epoch)

        # Flush to disk
        self.writer.flush()

    def on_train_end(self, trainer):
        """Called when training ends."""
        if self.writer is not None:
            self.writer.close()
            print(f"✅ TensorBoard logs saved")

    def on_val_end(self, trainer):
        """Called after validation."""
        # Validation metrics are logged in on_train_epoch_end
        pass
