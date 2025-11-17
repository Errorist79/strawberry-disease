#!/usr/bin/env python3
"""
Train YOLOv8 model on strawberry disease detection dataset.

This script now uses the modular training framework while maintaining
backward compatibility with the original interface.

For ensemble training, use train_ensemble.py instead.
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

# Add src to path for modular training framework
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config.augmentation_config import (
    AggressiveAugmentation,
    StandardAugmentation,
    get_augmentation_preset,
)
from src.training.config.base_config import DataConfig, ModelConfig, TrainingConfig
from src.training.config.training_presets import get_preset
from src.training.data.dataset_loader import validate_dataset
from src.training.models.yolo_trainer import YOLOTrainer


def prepare_dataset(data_dir: Path, output_dir: Path) -> Path:
    """
    Prepare dataset in YOLO format.

    Args:
        data_dir: Raw dataset directory (can be data/raw or data/processed/yolo_dataset)
        output_dir: Output directory for YOLO-formatted dataset

    Returns:
        Path to dataset.yaml file
    """
    print("Preparing dataset...")

    # Create YOLO dataset structure
    yolo_dir = output_dir / "yolo_dataset"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    # First check if data/processed/yolo_dataset exists (converted dataset)
    processed_dataset = Path("data/processed/yolo_dataset/data.yaml")
    if processed_dataset.exists():
        print(f"Found converted dataset: {processed_dataset}")
        return processed_dataset

    # If dataset is already in YOLO format, find the data.yaml
    yaml_files = list(data_dir.rglob("data.yaml")) + list(data_dir.rglob("dataset.yaml"))

    if yaml_files:
        # Dataset already in YOLO format
        dataset_yaml = yaml_files[0]
        print(f"Found existing dataset config: {dataset_yaml}")

        # Read and potentially modify paths
        with open(dataset_yaml, "r") as f:
            config = yaml.safe_load(f)

        # Update paths to be absolute
        base_path = dataset_yaml.parent
        if "path" not in config:
            config["path"] = str(base_path)

        # Save updated config
        output_yaml = yolo_dir / "dataset.yaml"
        with open(output_yaml, "w") as f:
            yaml.dump(config, f)

        print(f"Dataset config saved to: {output_yaml}")
        return output_yaml

    else:
        print("No YOLO format dataset found. Please organize dataset manually.")
        print("Expected structure:")
        print("  data/raw/")
        print("    ├── train/")
        print("    │   ├── images/")
        print("    │   └── labels/")
        print("    ├── val/")
        print("    │   ├── images/")
        print("    │   └── labels/")
        print("    └── data.yaml")
        sys.exit(1)


def train_yolo(
    dataset_yaml: Path,
    model_size: str = "n",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    output_dir: Path = None,
) -> Path:
    """
    Train YOLOv8 model.

    Args:
        dataset_yaml: Path to dataset.yaml
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
        device: Device to use (cpu, 0, 0,1,2,3)
        output_dir: Output directory for trained models

    Returns:
        Path to best model weights
    """
    print(f"\n=== Training YOLOv8{model_size} ===")
    print(f"Dataset: {dataset_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print()

    # Initialize model
    model = YOLO(f"yolov8{model_size}.pt")

    # Train
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(output_dir) if output_dir else "runs/detect",
        name="strawberry_disease",
        patience=15,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        cache='disk',  # Use disk cache to save GPU memory
        exist_ok=True,
        workers=16,  # More workers for CPU training
        amp=False,  # Disable AMP for CPU
    )

    # Get path to best model
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n=== Training Complete ===")
    print(f"Best model saved to: {best_model_path}")

    return best_model_path


def evaluate_model(model_path: Path, dataset_yaml: Path) -> dict:
    """
    Evaluate trained model on validation set.

    Args:
        model_path: Path to model weights
        dataset_yaml: Path to dataset.yaml

    Returns:
        Evaluation metrics
    """
    print(f"\n=== Evaluating Model ===")

    model = YOLO(str(model_path))
    metrics = model.val(data=str(dataset_yaml))

    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for strawberry disease detection"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw dataset directory (deprecated, use --data-yaml)",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        help="Path to dataset.yaml (recommended)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="l",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size (-1 for auto, but not supported with multi-GPU). For multi-GPU, use a multiple of GPU count (e.g., --batch 32 for 2 GPUs = 16 per GPU)")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu, 0, 0,1,2,3) - default: cpu for compatibility"
    )
    parser.add_argument(
        "--weights-output",
        type=Path,
        default=Path("models/weights"),
        help="Output directory for final model weights",
    )

    # New modular options
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick_test", "standard", "anti_overfitting", "fine_tuning", "balanced_oversampled"],
        help="Use training preset (overrides other settings). For multi-GPU training, also specify --batch to override preset's AutoBatch setting.",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        choices=["minimal", "standard", "aggressive"],
        default="aggressive",
        help="Augmentation level (default: aggressive for anti-overfitting)",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class weights for imbalanced data",
    )
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Validate dataset before training",
    )
    parser.add_argument(
        "--legacy-mode",
        action="store_true",
        help="Use legacy training (bypasses modular framework)",
    )

    args = parser.parse_args()

    # Check if --batch was explicitly provided on command line
    import sys
    batch_explicitly_set = '--batch' in sys.argv

    # Determine dataset YAML path
    if args.data_yaml:
        dataset_yaml = args.data_yaml
    else:
        # Legacy mode: prepare dataset
        dataset_yaml = prepare_dataset(args.data_dir, args.output_dir)

    # Validate dataset if requested
    if args.validate_data:
        print("Validating dataset...")
        if not validate_dataset(dataset_yaml, verbose=True):
            print("❌ Dataset validation failed!")
            sys.exit(1)
        print("✅ Dataset is valid\n")

    # Legacy mode: use old training function
    if args.legacy_mode:
        print("Using legacy training mode...\n")
        best_model_path = train_yolo(
            dataset_yaml=dataset_yaml,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

        # Evaluate model
        metrics = evaluate_model(best_model_path, dataset_yaml)

        # Copy best model to final location
        args.weights_output.mkdir(parents=True, exist_ok=True)
        final_model_path = args.weights_output / "best.pt"
        shutil.copy(best_model_path, final_model_path)

        print(f"\n=== Training Complete ===")
        print(f"Final model saved to: {final_model_path}")
        print(f"Metrics: {metrics}")
        return

    # New modular mode
    print("Using modular training framework...\n")

    # Use preset if specified
    if args.preset:
        print(f"Loading preset: {args.preset}")
        preset_config = get_preset(
            preset_name=args.preset,
            dataset_yaml=str(dataset_yaml),
            model_size=args.model_size,
            device=args.device,
        )

        model_config = preset_config.model
        training_config = preset_config.training
        augmentation_config = preset_config.augmentation
        data_config = preset_config.data

        print(f"Preset description: {preset_config.description}\n")

        # Override batch size if explicitly specified on command line
        # This is critical for multi-GPU training where AutoBatch (batch=-1) is not supported
        if batch_explicitly_set:
            print(f"⚙️  Overriding preset batch_size ({training_config.batch_size}) with --batch {args.batch}")
            training_config.batch_size = args.batch
        elif training_config.batch_size == -1 and ',' in str(args.device):
            # Preset uses AutoBatch but multi-GPU detected
            gpu_count = len(str(args.device).split(','))
            suggested_batch = 16 * gpu_count  # 16 per GPU is a good starting point
            print(f"⚠️  WARNING: Preset uses AutoBatch (batch=-1) which is incompatible with multi-GPU training!")
            print(f"   Detected {gpu_count} GPUs in device: {args.device}")
            print(f"   Please specify --batch with a multiple of {gpu_count}")
            print(f"   Suggested: --batch {suggested_batch} (or {suggested_batch//2}, {suggested_batch*2})")
            print(f"\nTo fix, re-run with:")
            print(f"  --batch {suggested_batch}")
            import sys
            sys.exit(1)

    else:
        # Build configuration from arguments
        model_config = ModelConfig(
            model_size=args.model_size,
            input_size=args.imgsz,
        )

        # Setup class weights if requested
        class_weights = None
        if args.use_class_weights:
            class_weights = {
                "anthracnose_fruit_rot": 3.0,
                "powdery_mildew_fruit": 3.0,
                "blossom_blight": 2.0,
            }

        data_config = DataConfig(
            dataset_yaml=dataset_yaml,
            class_weights=class_weights,
            cache="disk",
            workers=16,
        )

        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch,
            device=args.device,
            patience=30,
            lr0=0.01,
            lrf=0.001,
            weight_decay=0.001,
            warmup_epochs=5,
            dropout=0.3,
        )

        augmentation_config = get_augmentation_preset(args.augmentation)

    # Validate batch size for multi-GPU training
    if ',' in str(training_config.device):
        gpu_count = len(str(training_config.device).split(','))
        batch_size = training_config.batch_size

        if batch_size == -1:
            # This shouldn't happen after our checks above, but just in case
            print(f"❌ ERROR: AutoBatch (batch=-1) is not supported with multi-GPU training!")
            print(f"   Please specify --batch with a multiple of {gpu_count}")
            import sys
            sys.exit(1)
        elif batch_size % gpu_count != 0:
            print(f"⚠️  WARNING: Batch size {batch_size} is not evenly divisible by GPU count {gpu_count}")
            print(f"   This may cause uneven GPU utilization or errors during training.")
            print(f"   Recommended batch sizes: {', '.join([str(gpu_count * i) for i in range(4, 12, 2)])}")
            print(f"\n   Continue anyway? (y/N): ", end='')
            response = input().strip().lower()
            if response != 'y':
                print("Training cancelled. Please adjust --batch to a multiple of GPU count.")
                import sys
                sys.exit(1)
        else:
            per_gpu_batch = batch_size // gpu_count
            print(f"✅ Multi-GPU training: {gpu_count} GPUs, batch size {batch_size} ({per_gpu_batch} per GPU)\n")

    # Create trainer
    trainer = YOLOTrainer(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        augmentation_config=augmentation_config,
    )

    # Train
    training_results = trainer.train()

    # Validate
    val_results = trainer.validate(split="val")

    # Try test set if available
    try:
        test_results = trainer.validate(split="test")
        print("\nTest set results:")
        print(f"  mAP50: {test_results['map50']:.4f}")
        print(f"  mAP50-95: {test_results['map50_95']:.4f}")
    except Exception as e:
        print(f"Note: Test set validation skipped ({e})")

    # Copy best model to final location
    args.weights_output.mkdir(parents=True, exist_ok=True)
    final_model_path = args.weights_output / "best.pt"
    shutil.copy(trainer.best_model_path, final_model_path)

    # Save configuration
    config_path = args.weights_output / "training_config.json"
    trainer.save_config(config_path)

    print(f"\n{'='*70}")
    print("✅ Training Complete!")
    print(f"{'='*70}")
    print(f"Final model: {final_model_path}")
    print(f"Configuration: {config_path}")
    print(f"Validation mAP50: {val_results['map50']:.4f}")
    print(f"Validation mAP50-95: {val_results['map50_95']:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
