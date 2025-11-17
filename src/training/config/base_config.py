"""
Base configuration classes for training.

Provides dataclass-based configuration for models, data, and training parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class ModelConfig:
    """Model configuration."""

    model_size: str = "l"  # n, s, m, l, x
    pretrained: bool = True
    input_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    def __post_init__(self):
        """Validate configuration."""
        valid_sizes = ["n", "s", "m", "l", "x"]
        if self.model_size not in valid_sizes:
            raise ValueError(
                f"model_size must be one of {valid_sizes}, got {self.model_size}"
            )

        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                f"confidence_threshold must be between 0 and 1, got {self.confidence_threshold}"
            )

        if not 0 <= self.iou_threshold <= 1:
            raise ValueError(
                f"iou_threshold must be between 0 and 1, got {self.iou_threshold}"
            )

    @property
    def model_name(self) -> str:
        """Get full model name."""
        return f"yolov8{self.model_size}"


@dataclass
class DataConfig:
    """Data configuration."""

    dataset_yaml: Union[str, Path]
    train_split: str = "train"
    val_split: str = "val"
    test_split: Optional[str] = "test"
    class_weights: Optional[Dict[str, float]] = None
    oversample_classes: Optional[Dict[str, int]] = None  # class -> multiplier
    cache: str = "disk"  # ram, disk, or False
    workers: int = 8

    def __post_init__(self):
        """Validate and convert paths."""
        self.dataset_yaml = Path(self.dataset_yaml)
        if not self.dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {self.dataset_yaml}")

        valid_cache = ["ram", "disk", False, "False"]
        if self.cache not in valid_cache:
            raise ValueError(f"cache must be one of {valid_cache}, got {self.cache}")


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic training parameters
    epochs: int = 200
    batch_size: int = -1  # -1 for auto
    device: Union[str, int, List[int]] = "cpu"

    # Optimization parameters
    optimizer: str = "auto"  # auto, SGD, Adam, AdamW
    lr0: float = 0.01  # Initial learning rate
    lrf: float = 0.01  # Final learning rate (lr0 * lrf)
    momentum: float = 0.937
    weight_decay: float = 0.001  # Increased from default 0.0005
    warmup_epochs: int = 5  # Increased from default 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # Regularization
    dropout: float = 0.3  # Added dropout

    # Early stopping and checkpointing
    patience: int = 30  # Early stopping patience
    save_period: int = 10  # Save checkpoint every N epochs

    # Mixed precision
    amp: bool = True  # Automatic Mixed Precision

    # Output configuration
    project: str = "runs/detect"
    name: str = "strawberry_disease"
    exist_ok: bool = True
    verbose: bool = True

    # Additional YOLO-specific parameters
    close_mosaic: int = 10  # Disable mosaic augmentation in last N epochs

    def __post_init__(self):
        """Validate configuration."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")

        if self.lr0 <= 0:
            raise ValueError(f"lr0 must be positive, got {self.lr0}")

        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")

    def to_yolo_args(self) -> Dict:
        """Convert to YOLO training arguments."""
        return {
            "epochs": self.epochs,
            "batch": self.batch_size,
            "device": self.device,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            "dropout": self.dropout,
            "patience": self.patience,
            "save_period": self.save_period,
            "amp": self.amp,
            "project": self.project,
            "name": self.name,
            "exist_ok": self.exist_ok,
            "verbose": self.verbose,
            "close_mosaic": self.close_mosaic,
        }
