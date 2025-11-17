#!/usr/bin/env python3
"""
Train ensemble of YOLOv8 models for strawberry disease detection.

This script trains multiple models with different configurations and combines
them for improved accuracy. Based on recommendations from docs/analyze.md.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config.augmentation_config import (
    AggressiveAugmentation,
    StandardAugmentation,
)
from src.training.config.base_config import DataConfig, ModelConfig, TrainingConfig
from src.training.data.dataset_loader import validate_dataset
from src.training.data.oversampling import create_oversampled_dataset
from src.training.models.ensemble_trainer import EnsembleTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train ensemble of YOLOv8 models for strawberry disease detection"
    )

    # Dataset arguments
    parser.add_argument(
        "--data-yaml",
        type=Path,
        required=True,
        help="Path to dataset.yaml file",
    )
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Validate dataset before training",
    )

    # Model arguments
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["l", "l", "m"],
        help="Model sizes for ensemble (e.g., l l m)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        nargs="+",
        default=["aggressive", "standard", "standard"],
        help="Augmentation levels for each model (aggressive, standard)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 for auto)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device (cpu, 0, 0,1,2,3)",
    )

    # Ensemble arguments
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train models in parallel (requires multiple GPUs)",
    )
    parser.add_argument(
        "--ensemble-name",
        type=str,
        default="strawberry_ensemble",
        help="Name for ensemble",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/ensemble"),
        help="Output directory",
    )

    # Class balancing arguments
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class weights for imbalanced classes",
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Create oversampled dataset for underrepresented classes",
    )
    parser.add_argument(
        "--oversample-output",
        type=Path,
        default=Path("data/processed/oversampled"),
        help="Output directory for oversampled dataset",
    )

    args = parser.parse_args()

    # Validate number of models matches augmentation configs
    if len(args.models) != len(args.augmentation):
        print("Error: Number of models must match number of augmentation configs")
        print(f"  Models: {len(args.models)} ({args.models})")
        print(f"  Augmentation: {len(args.augmentation)} ({args.augmentation})")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("Ensemble Training Configuration")
    print(f"{'='*70}")
    print(f"Dataset: {args.data_yaml}")
    print(f"Models: {args.models}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Parallel: {args.parallel}")
    print(f"{'='*70}\n")

    # Validate dataset
    if args.validate_data:
        print("Validating dataset...")
        if not validate_dataset(args.data_yaml, verbose=True):
            print("❌ Dataset validation failed!")
            sys.exit(1)
        print("✅ Dataset is valid\n")

    # Handle dataset preparation
    dataset_yaml = args.data_yaml

    # Create oversampled dataset if requested
    if args.oversample:
        print("Creating oversampled dataset for underrepresented classes...")
        oversample_config = {
            "anthracnose_fruit_rot": 3,  # 3x multiplier
            "powdery_mildew_fruit": 3,  # 3x multiplier
        }

        dataset_yaml = create_oversampled_dataset(
            dataset_yaml=args.data_yaml,
            oversample_config=oversample_config,
            output_dir=args.oversample_output,
            splits=["train"],  # Only oversample training set
        )
        print(f"✅ Oversampled dataset created: {dataset_yaml}\n")

    # Setup class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = {
            "anthracnose_fruit_rot": 3.0,
            "powdery_mildew_fruit": 3.0,
            "blossom_blight": 2.0,
        }
        print(f"Using class weights: {class_weights}\n")

    # Create data config
    data_config = DataConfig(
        dataset_yaml=dataset_yaml,
        class_weights=class_weights,
        cache="disk",
        workers=8,
    )

    # Create model configurations
    model_configs = []
    for model_size, aug_level in zip(args.models, args.augmentation):
        # Model config
        model_cfg = ModelConfig(
            model_size=model_size,
            input_size=args.imgsz,
        )

        # Training config
        training_cfg = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch,
            device=args.device,
            patience=30,
            lr0=0.01,
            lrf=0.001,  # Lower final LR for better convergence
            weight_decay=0.001,
            warmup_epochs=5,
            dropout=0.3,
            label_smoothing=0.1,
            project=str(args.output_dir),
        )

        # Augmentation config
        if aug_level.lower() == "aggressive":
            aug_cfg = AggressiveAugmentation()
        else:
            aug_cfg = StandardAugmentation()

        model_configs.append((model_cfg, training_cfg, aug_cfg))

    # Create ensemble trainer
    ensemble_trainer = EnsembleTrainer(
        models=model_configs,
        data_config=data_config,
        output_dir=args.output_dir,
        ensemble_name=args.ensemble_name,
    )

    # Train ensemble
    print("Starting ensemble training...\n")
    training_results = ensemble_trainer.train_all(parallel=args.parallel)

    # Validate ensemble
    print("\nValidating ensemble on validation set...")
    val_results = ensemble_trainer.validate_ensemble(split="val")

    # Try to validate on test set if available
    try:
        print("\nValidating ensemble on test set...")
        test_results = ensemble_trainer.validate_ensemble(split="test")
    except Exception as e:
        print(f"Note: Test set validation skipped ({e})")
        test_results = None

    # Print final summary
    print(f"\n{'='*70}")
    print("✅ ENSEMBLE TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nEnsemble: {args.ensemble_name}")
    print(f"Number of models: {len(args.models)}")
    print(f"Output directory: {ensemble_trainer.ensemble_dir}")
    print(f"\nValidation Results:")
    print(
        f"  Weighted mAP50: {val_results['ensemble_metrics']['weighted_map50']:.4f}"
    )
    print(
        f"  Weighted mAP50-95: {val_results['ensemble_metrics']['weighted_map50_95']:.4f}"
    )
    print(
        f"  Weighted Precision: {val_results['ensemble_metrics']['weighted_precision']:.4f}"
    )
    print(
        f"  Weighted Recall: {val_results['ensemble_metrics']['weighted_recall']:.4f}"
    )

    if test_results:
        print(f"\nTest Results:")
        print(
            f"  Weighted mAP50: {test_results['ensemble_metrics']['weighted_map50']:.4f}"
        )
        print(
            f"  Weighted mAP50-95: {test_results['ensemble_metrics']['weighted_map50_95']:.4f}"
        )

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
