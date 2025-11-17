"""Configuration classes for training."""

from .base_config import TrainingConfig, ModelConfig, DataConfig
from .augmentation_config import (
    AugmentationConfig,
    StandardAugmentation,
    AggressiveAugmentation,
)
from .training_presets import TrainingPresets, get_preset

__all__ = [
    "TrainingConfig",
    "ModelConfig",
    "DataConfig",
    "AugmentationConfig",
    "StandardAugmentation",
    "AggressiveAugmentation",
    "TrainingPresets",
    "get_preset",
]
