#!/usr/bin/env python3
"""
Train YOLOv8 model on strawberry disease detection dataset.

This script:
1. Prepares the dataset in YOLO format
2. Creates train/val splits if needed
3. Trains YOLOv8 model
4. Evaluates and saves the best model
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


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
        patience=10,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        cache=False,  # Don't cache images (memory intensive)
        exist_ok=True,
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
    parser = argparse.ArgumentParser(description="Train YOLOv8 for strawberry disease detection")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Raw dataset directory",
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
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, 0, 0,1,2,3)")
    parser.add_argument(
        "--weights-output",
        type=Path,
        default=Path("models/weights"),
        help="Output directory for final model weights",
    )

    args = parser.parse_args()

    # Prepare dataset
    dataset_yaml = prepare_dataset(args.data_dir, args.output_dir)

    # Train model
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


if __name__ == "__main__":
    main()
