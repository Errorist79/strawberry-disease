#!/usr/bin/env python3
"""
Training examples demonstrating different usage patterns.

These examples show how to use the modular training framework
for various scenarios.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config.augmentation_config import (
    AggressiveAugmentation,
    StandardAugmentation,
    get_augmentation_preset,
)
from src.training.config.base_config import DataConfig, ModelConfig, TrainingConfig
from src.training.config.training_presets import get_preset
from src.training.data.dataset_loader import DatasetLoader, validate_dataset
from src.training.data.oversampling import create_oversampled_dataset
from src.training.models.ensemble_trainer import EnsembleTrainer
from src.training.models.yolo_trainer import YOLOTrainer


def example_1_quick_test():
    """Example 1: Quick test with minimal training."""
    print("\n" + "=" * 70)
    print("Example 1: Quick Test (10 epochs, nano model)")
    print("=" * 70)

    # Use quick_test preset
    preset = get_preset(
        preset_name="quick_test",
        dataset_yaml="data/processed/yolo_dataset/dataset.yaml",
    )

    trainer = YOLOTrainer(
        model_config=preset.model,
        data_config=preset.data,
        training_config=preset.training,
        augmentation_config=preset.augmentation,
    )

    # Train
    results = trainer.train()

    print(f"\n✅ Quick test complete! Model saved to: {trainer.best_model_path}")


def example_2_standard_training():
    """Example 2: Standard training with moderate settings."""
    print("\n" + "=" * 70)
    print("Example 2: Standard Training (YOLOv8l, 200 epochs)")
    print("=" * 70)

    # Manual configuration
    model_config = ModelConfig(
        model_size="l",
        input_size=640,
    )

    data_config = DataConfig(
        dataset_yaml="data/processed/yolo_dataset/dataset.yaml",
        cache="disk",
        workers=8,
    )

    training_config = TrainingConfig(
        epochs=200,
        batch_size=-1,  # Auto
        device="0",
        patience=30,
    )

    augmentation_config = StandardAugmentation()

    # Create trainer
    trainer = YOLOTrainer(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        augmentation_config=augmentation_config,
    )

    # Train
    results = trainer.train()

    # Validate
    val_results = trainer.validate(split="val")
    test_results = trainer.validate(split="test")

    print(f"\n✅ Training complete!")
    print(f"Validation mAP50: {val_results['map50']:.4f}")
    print(f"Test mAP50: {test_results['map50']:.4f}")


def example_3_anti_overfitting():
    """Example 3: Anti-overfitting configuration (recommended)."""
    print("\n" + "=" * 70)
    print("Example 3: Anti-Overfitting (Aggressive aug + regularization)")
    print("=" * 70)

    # Use anti_overfitting preset
    preset = get_preset(
        preset_name="anti_overfitting",
        dataset_yaml="data/processed/yolo_dataset/dataset.yaml",
        model_size="l",
        device="0",
        class_weights={
            "anthracnose_fruit_rot": 3.0,
            "powdery_mildew_fruit": 3.0,
            "blossom_blight": 2.0,
        },
    )

    print(f"Preset: {preset.description}")
    print(f"Augmentation: {preset.augmentation.__class__.__name__}")
    print(f"Class weights: {preset.data.class_weights}")

    trainer = YOLOTrainer(
        model_config=preset.model,
        data_config=preset.data,
        training_config=preset.training,
        augmentation_config=preset.augmentation,
    )

    # Train
    results = trainer.train()

    # Validate both splits
    val_results = trainer.validate(split="val")
    test_results = trainer.validate(split="test")

    # Calculate val-test gap
    gap_map50 = abs(val_results["map50"] - test_results["map50"])
    gap_map50_95 = abs(val_results["map50_95"] - test_results["map50_95"])

    print(f"\n📊 Results:")
    print(f"Validation mAP50: {val_results['map50']:.4f}")
    print(f"Test mAP50: {test_results['map50']:.4f}")
    print(f"Val-Test gap (mAP50): {gap_map50:.4f}")
    print(f"Val-Test gap (mAP50-95): {gap_map50_95:.4f}")

    if gap_map50 < 0.05:  # Less than 5% gap
        print("✅ Low overfitting detected!")
    else:
        print("⚠️  Consider more aggressive regularization")


def example_4_with_oversampling():
    """Example 4: Training with oversampled dataset."""
    print("\n" + "=" * 70)
    print("Example 4: Training with Oversampling")
    print("=" * 70)

    # First, create oversampled dataset
    print("Creating oversampled dataset...")
    oversample_config = {
        "anthracnose_fruit_rot": 3,  # 3x multiplier
        "powdery_mildew_fruit": 3,
    }

    oversampled_yaml = create_oversampled_dataset(
        dataset_yaml=Path("data/processed/yolo_dataset/dataset.yaml"),
        oversample_config=oversample_config,
        output_dir=Path("data/processed/oversampled"),
        splits=["train"],
    )

    print(f"✅ Oversampled dataset created: {oversampled_yaml}")

    # Now train with oversampled dataset
    preset = get_preset(
        preset_name="anti_overfitting",
        dataset_yaml=str(oversampled_yaml),
        model_size="l",
    )

    trainer = YOLOTrainer(
        model_config=preset.model,
        data_config=preset.data,
        training_config=preset.training,
        augmentation_config=preset.augmentation,
    )

    results = trainer.train()

    print(f"\n✅ Training complete with oversampled data!")


def example_5_ensemble():
    """Example 5: Ensemble training (best accuracy)."""
    print("\n" + "=" * 70)
    print("Example 5: Ensemble Training (3 models)")
    print("=" * 70)

    # Dataset config (shared by all models)
    data_config = DataConfig(
        dataset_yaml="data/processed/yolo_dataset/dataset.yaml",
        class_weights={
            "anthracnose_fruit_rot": 3.0,
            "powdery_mildew_fruit": 3.0,
        },
        cache="disk",
    )

    # Define 3 models with different configurations
    models = [
        # Model 1: YOLOv8l + Aggressive augmentation
        (
            ModelConfig(model_size="l", input_size=640),
            TrainingConfig(epochs=200, device="0", patience=30),
            AggressiveAugmentation(),
        ),
        # Model 2: YOLOv8l + Standard augmentation
        (
            ModelConfig(model_size="l", input_size=640),
            TrainingConfig(epochs=200, device="0", patience=30),
            StandardAugmentation(),
        ),
        # Model 3: YOLOv8m + Standard augmentation (lighter model)
        (
            ModelConfig(model_size="m", input_size=640),
            TrainingConfig(epochs=200, device="0", patience=30),
            StandardAugmentation(),
        ),
    ]

    # Create ensemble trainer
    ensemble = EnsembleTrainer(
        models=models,
        data_config=data_config,
        output_dir=Path("runs/ensemble"),
        ensemble_name="strawberry_final",
    )

    # Train all models
    print("Training ensemble (this will take a while)...")
    results = ensemble.train_all(parallel=False)

    # Validate ensemble
    val_results = ensemble.validate_ensemble(split="val")
    test_results = ensemble.validate_ensemble(split="test")

    print(f"\n📊 Ensemble Results:")
    print(f"Validation mAP50: {val_results['ensemble_metrics']['weighted_map50']:.4f}")
    print(f"Test mAP50: {test_results['ensemble_metrics']['weighted_map50']:.4f}")

    print(f"\n✅ Ensemble training complete!")
    print(f"Models saved to: {ensemble.ensemble_dir}")


def example_6_dataset_validation():
    """Example 6: Dataset validation and analysis."""
    print("\n" + "=" * 70)
    print("Example 6: Dataset Validation and Analysis")
    print("=" * 70)

    dataset_yaml = Path("data/processed/yolo_dataset/dataset.yaml")

    # Validate dataset
    is_valid = validate_dataset(dataset_yaml, verbose=True)

    if not is_valid:
        print("❌ Dataset has errors, fix them before training!")
        return

    # Load dataset for analysis
    loader = DatasetLoader(dataset_yaml)

    # Calculate recommended class weights
    print("\n📊 Calculating recommended class weights...")
    class_weights = loader.get_class_weights(method="inverse", power=1.0)

    print("\nRecommended class weights:")
    for class_name, weight in sorted(
        class_weights.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {class_name}: {weight:.2f}")

    # Get validation results
    validation = loader.validate()

    if "train" in validation["stats"]:
        class_dist = validation["stats"]["train"]["class_distribution"]
        total = validation["stats"]["train"]["total_instances"]

        print(f"\nClass distribution (training set):")
        for class_name, count in sorted(
            class_dist.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = 100 * count / total
            print(f"  {class_name}: {count} ({percentage:.1f}%)")

            # Recommend oversampling if very low
            if percentage < 2.0:
                multiplier = int(2.0 / percentage)
                print(f"    ⚠️  Consider oversampling {multiplier}x")


def example_7_custom_augmentation():
    """Example 7: Using custom augmentation configuration."""
    print("\n" + "=" * 70)
    print("Example 7: Custom Augmentation")
    print("=" * 70)

    from dataclasses import dataclass

    from src.training.config.augmentation_config import AugmentationConfig

    # Define custom augmentation
    @dataclass
    class GreenhouseAugmentation(AugmentationConfig):
        """Custom augmentation for greenhouse conditions."""

        # Heavy HSV for varied lighting
        hsv_h: float = 0.08
        hsv_s: float = 0.4
        hsv_v: float = 0.2

        # Moderate geometric
        degrees: float = 20.0
        translate: float = 0.15
        scale: float = 0.6

        # Standard advanced
        mosaic: float = 1.0
        mixup: float = 0.1
        copy_paste: float = 0.2

    # Use custom augmentation
    trainer = YOLOTrainer(
        model_config=ModelConfig(model_size="l"),
        data_config=DataConfig(
            dataset_yaml="data/processed/yolo_dataset/dataset.yaml"
        ),
        training_config=TrainingConfig(epochs=200),
        augmentation_config=GreenhouseAugmentation(),
    )

    results = trainer.train()

    print(f"✅ Training complete with custom augmentation!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training examples")
    parser.add_argument(
        "example",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Example number to run",
    )

    args = parser.parse_args()

    examples = {
        1: example_1_quick_test,
        2: example_2_standard_training,
        3: example_3_anti_overfitting,
        4: example_4_with_oversampling,
        5: example_5_ensemble,
        6: example_6_dataset_validation,
        7: example_7_custom_augmentation,
    }

    examples[args.example]()

    print("\n" + "=" * 70)
    print(f"Example {args.example} completed!")
    print("=" * 70 + "\n")
