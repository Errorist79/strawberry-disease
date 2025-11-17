"""
Training presets for common scenarios.

Provides pre-configured training setups for different use cases.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .augmentation_config import (
    AugmentationConfig,
    AggressiveAugmentation,
    StandardAugmentation,
)
from .base_config import DataConfig, ModelConfig, TrainingConfig


@dataclass
class PresetConfig:
    """Complete configuration preset."""

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    augmentation: AugmentationConfig
    description: str = ""


class TrainingPresets:
    """Collection of training presets."""

    @staticmethod
    def quick_test(dataset_yaml: str) -> PresetConfig:
        """
        Quick test configuration.

        Fast training for testing pipeline, not for production.
        """
        return PresetConfig(
            model=ModelConfig(model_size="n", input_size=416),
            data=DataConfig(dataset_yaml=dataset_yaml, cache="ram", workers=4),
            training=TrainingConfig(
                epochs=10,
                batch_size=32,
                patience=5,
                lr0=0.01,
                warmup_epochs=1,
            ),
            augmentation=StandardAugmentation(),
            description="Quick test run with minimal epochs",
        )

    @staticmethod
    def standard_training(
        dataset_yaml: str,
        model_size: str = "l",
        device: str = "0",
    ) -> PresetConfig:
        """
        Standard training configuration.

        Balanced approach suitable for most cases.
        """
        return PresetConfig(
            model=ModelConfig(model_size=model_size, input_size=640),
            data=DataConfig(dataset_yaml=dataset_yaml, cache="disk", workers=8),
            training=TrainingConfig(
                epochs=200,
                batch_size=-1,  # Auto
                device=device,
                patience=30,
                lr0=0.01,
                lrf=0.01,
                weight_decay=0.001,
                warmup_epochs=5,
                dropout=0.3,
            ),
            augmentation=StandardAugmentation(),
            description="Standard training with moderate augmentation",
        )

    @staticmethod
    def anti_overfitting(
        dataset_yaml: str,
        model_size: str = "l",
        device: str = "0",
        class_weights: Optional[Dict[str, float]] = None,
    ) -> PresetConfig:
        """
        Anti-overfitting configuration.

        Based on analyze.md recommendations. Uses aggressive augmentation
        and strong regularization to combat overfitting.

        Recommended class_weights for strawberry dataset:
        {
            'anthracnose_fruit_rot': 3.0,
            'powdery_mildew_fruit': 3.0,
            'blossom_blight': 2.0,
        }
        """
        # Default class weights if not provided
        if class_weights is None:
            class_weights = {
                "anthracnose_fruit_rot": 3.0,
                "powdery_mildew_fruit": 3.0,
                "blossom_blight": 2.0,
            }

        return PresetConfig(
            model=ModelConfig(model_size=model_size, input_size=640),
            data=DataConfig(
                dataset_yaml=dataset_yaml,
                class_weights=class_weights,
                cache="disk",
                workers=8,
            ),
            training=TrainingConfig(
                epochs=200,
                batch_size=-1,  # Auto-batch
                device=device,
                patience=30,
                lr0=0.01,
                lrf=0.001,  # Lower final LR
                weight_decay=0.001,  # Stronger regularization
                warmup_epochs=5,
                dropout=0.3,  # High dropout
                close_mosaic=10,  # Disable mosaic in last 10 epochs
            ),
            augmentation=AggressiveAugmentation(),
            description="Aggressive augmentation + regularization to prevent overfitting",
        )

    @staticmethod
    def fine_tuning(
        dataset_yaml: str,
        pretrained_weights: str,
        model_size: str = "l",
        device: str = "0",
    ) -> PresetConfig:
        """
        Fine-tuning configuration.

        For fine-tuning a pre-trained model on new data.
        Uses lower learning rate and fewer epochs.
        """
        return PresetConfig(
            model=ModelConfig(model_size=model_size, input_size=640),
            data=DataConfig(dataset_yaml=dataset_yaml, cache="disk", workers=8),
            training=TrainingConfig(
                epochs=50,
                batch_size=-1,
                device=device,
                patience=15,
                lr0=0.001,  # Lower LR for fine-tuning
                lrf=0.0001,
                weight_decay=0.0005,
                warmup_epochs=3,
                dropout=0.2,
            ),
            augmentation=StandardAugmentation(),
            description=f"Fine-tuning from {pretrained_weights}",
        )

    @staticmethod
    def ensemble_member(
        dataset_yaml: str,
        model_size: str = "l",
        augmentation_level: str = "standard",
        device: str = "0",
        class_weights: Optional[Dict[str, float]] = None,
    ) -> PresetConfig:
        """
        Configuration for ensemble member.

        Creates diverse models for ensemble by varying augmentation.

        Args:
            dataset_yaml: Path to dataset configuration
            model_size: Model size (n, s, m, l, x)
            augmentation_level: Augmentation intensity (minimal, standard, aggressive)
            device: Device to use
            class_weights: Optional class weights for imbalanced data
        """
        # Select augmentation based on level
        if augmentation_level == "aggressive":
            augmentation = AggressiveAugmentation()
        elif augmentation_level == "minimal":
            from .augmentation_config import MinimalAugmentation

            augmentation = MinimalAugmentation()
        else:
            augmentation = StandardAugmentation()

        return PresetConfig(
            model=ModelConfig(model_size=model_size, input_size=640),
            data=DataConfig(
                dataset_yaml=dataset_yaml,
                class_weights=class_weights,
                cache="disk",
                workers=8,
            ),
            training=TrainingConfig(
                epochs=200,
                batch_size=-1,
                device=device,
                patience=30,
                lr0=0.01,
                lrf=0.01,
                weight_decay=0.001,
                warmup_epochs=5,
                dropout=0.3,
            ),
            augmentation=augmentation,
            description=f"Ensemble member: {model_size} with {augmentation_level} augmentation",
        )

    @staticmethod
    def balanced_oversampled(
        dataset_yaml: str,
        model_size: str = "l",
        device: str = "0",
        class_weights: Optional[Dict[str, float]] = None,
    ) -> PresetConfig:
        """
        Optimized configuration for oversampled dataset with augmentation.

        This preset is designed to work with datasets oversampled using the
        new augmentation-based oversampling strategy (target_balance=0.65).

        Uses moderate augmentation during training since the dataset copies
        already have augmentation applied, combined with class weights to
        handle remaining imbalance.

        Recommended class_weights for strawberry dataset after oversampling:
        {
            'healthy_flower': 8.0,     # Very underrepresented even after 5x
            'healthy_leaf': 4.0,       # Underrepresented after 5x
            'healthy_fruit': 6.0,      # Underrepresented after 5x
            'anthracnose_fruit_rot': 2.0,  # Moderate after 3x
        }

        Args:
            dataset_yaml: Path to oversampled dataset YAML
            model_size: Model size (n, s, m, l, x)
            device: Device to use
            class_weights: Optional class weights. If None, uses recommended defaults.
        """
        # Default class weights optimized for oversampled dataset
        if class_weights is None:
            class_weights = {
                "healthy_flower": 8.0,
                "healthy_leaf": 4.0,
                "healthy_fruit": 6.0,
                "anthracnose_fruit_rot": 2.0,
            }

        return PresetConfig(
            model=ModelConfig(model_size=model_size, input_size=640),
            data=DataConfig(
                dataset_yaml=dataset_yaml,
                class_weights=class_weights,
                cache="disk",
                workers=8,
            ),
            training=TrainingConfig(
                epochs=200,
                batch_size=-1,
                device=device,
                patience=30,
                lr0=0.01,
                lrf=0.01,
                weight_decay=0.001,  # Moderate regularization
                warmup_epochs=5,
                dropout=0.3,
                close_mosaic=15,  # Disable mosaic earlier to see real augmentation
            ),
            augmentation=StandardAugmentation(),  # Moderate - dataset already augmented
            description="Optimized for augmentation-based oversampled dataset with class weights",
        )


def get_preset(
    preset_name: str,
    dataset_yaml: str,
    model_size: str = "l",
    device: str = "0",
    **kwargs,
) -> PresetConfig:
    """
    Get a training preset by name.

    Args:
        preset_name: Name of preset (quick_test, standard, anti_overfitting, fine_tuning, ensemble)
        dataset_yaml: Path to dataset YAML
        model_size: Model size
        device: Device to use
        **kwargs: Additional preset-specific arguments

    Returns:
        PresetConfig instance

    Raises:
        ValueError: If preset name is not recognized
    """
    presets = {
        "quick_test": TrainingPresets.quick_test,
        "standard": TrainingPresets.standard_training,
        "anti_overfitting": TrainingPresets.anti_overfitting,
        "fine_tuning": TrainingPresets.fine_tuning,
        "ensemble": TrainingPresets.ensemble_member,
        "balanced_oversampled": TrainingPresets.balanced_oversampled,
    }

    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(presets.keys())}"
        )

    preset_func = presets[preset_name]

    # Build arguments
    args = {"dataset_yaml": dataset_yaml}

    # Add common arguments if accepted by preset
    import inspect

    sig = inspect.signature(preset_func)
    if "model_size" in sig.parameters:
        args["model_size"] = model_size
    if "device" in sig.parameters:
        args["device"] = device

    # Add any additional kwargs
    args.update(kwargs)

    return preset_func(**args)
