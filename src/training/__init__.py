"""
Modular training framework for strawberry disease detection.

This package provides a flexible and extensible training system for YOLOv8 models
with support for ensemble learning, advanced augmentation, and class balancing.
"""

from .config.base_config import TrainingConfig, ModelConfig, DataConfig
from .config.augmentation_config import (
    AugmentationConfig,
    StandardAugmentation,
    AggressiveAugmentation,
)
from .models.yolo_trainer import YOLOTrainer
from .models.ensemble_trainer import EnsembleTrainer

__all__ = [
    "TrainingConfig",
    "ModelConfig",
    "DataConfig",
    "AugmentationConfig",
    "StandardAugmentation",
    "AggressiveAugmentation",
    "YOLOTrainer",
    "EnsembleTrainer",
]

__version__ = "1.0.0"
