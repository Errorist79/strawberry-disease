"""
YOLO model trainer.

Provides a high-level wrapper for training single YOLO models with custom configurations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from ultralytics import YOLO

from ..config.augmentation_config import AugmentationConfig
from ..config.base_config import DataConfig, ModelConfig, TrainingConfig
from ..data.augmentation import apply_augmentation


class YOLOTrainer:
    """High-level YOLO model trainer."""

    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig,
        augmentation_config: Optional[AugmentationConfig] = None,
        callbacks: Optional[List] = None,
    ):
        """
        Initialize YOLO trainer.

        Args:
            model_config: Model configuration
            data_config: Data configuration
            training_config: Training configuration
            augmentation_config: Optional augmentation configuration
            callbacks: Optional list of callbacks
        """
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.augmentation_config = augmentation_config
        self.callbacks = callbacks or []

        # Initialize model
        if model_config.pretrained:
            model_path = f"{model_config.model_name}.pt"
        else:
            model_path = f"{model_config.model_name}.yaml"

        self.model = YOLO(model_path)

        # Training results
        self.results = None
        self.best_model_path = None

    def train(
        self, resume: bool = False, pretrained_weights: Optional[Path] = None
    ) -> Dict:
        """
        Train the model.

        Args:
            resume: Whether to resume from last checkpoint
            pretrained_weights: Optional path to pretrained weights

        Returns:
            Dictionary with training results and metrics
        """
        print(f"\n{'='*70}")
        print(f"Training {self.model_config.model_name.upper()}")
        print(f"{'='*70}")
        print(f"Dataset: {self.data_config.dataset_yaml}")
        print(f"Epochs: {self.training_config.epochs}")
        print(f"Image size: {self.model_config.input_size}")
        print(f"Batch size: {self.training_config.batch_size}")
        print(f"Device: {self.training_config.device}")
        print(f"{'='*70}\n")

        # Load pretrained weights if provided
        if pretrained_weights:
            print(f"Loading pretrained weights: {pretrained_weights}")
            self.model = YOLO(str(pretrained_weights))

        # Prepare training arguments
        train_args = self._prepare_training_args()

        # Add resume flag
        if resume:
            train_args["resume"] = True

        # Register callbacks
        for callback in self.callbacks:
            self.model.add_callback(callback)

        # Train model
        print("Starting training...")
        self.results = self.model.train(**train_args)

        # Get best model path
        self.best_model_path = Path(self.results.save_dir) / "weights" / "best.pt"

        print(f"\n{'='*70}")
        print(f"✅ Training Complete!")
        print(f"Best model: {self.best_model_path}")
        print(f"{'='*70}\n")

        # Return summary
        return self._get_training_summary()

    def _prepare_training_args(self) -> Dict:
        """Prepare arguments for YOLO training."""
        # Start with base training config
        train_args = self.training_config.to_yolo_args()

        # Add model-specific args
        train_args["imgsz"] = self.model_config.input_size
        train_args["data"] = str(self.data_config.dataset_yaml)

        # Add data-specific args
        train_args["cache"] = self.data_config.cache
        train_args["workers"] = self.data_config.workers

        # Apply augmentation if provided
        if self.augmentation_config:
            train_args = apply_augmentation(train_args, self.augmentation_config)

        # Add class weights if provided
        # Note: YOLO doesn't support class weights directly in train(),
        # but we can log them for reference
        if self.data_config.class_weights:
            print(f"\nClass weights: {self.data_config.class_weights}")
            # Store in training args for custom callbacks to use
            train_args["_class_weights"] = self.data_config.class_weights

        return train_args

    def _get_training_summary(self) -> Dict:
        """Get training summary with metrics."""
        if not self.results:
            return {}

        summary = {
            "model": self.model_config.model_name,
            "best_model_path": str(self.best_model_path),
            "save_dir": str(self.results.save_dir),
            "epochs_trained": len(self.results.results_dict) if hasattr(self.results, 'results_dict') else self.training_config.epochs,
        }

        # Add final metrics if available
        if hasattr(self.results, "results_dict"):
            metrics = self.results.results_dict
            summary["metrics"] = {
                "map50": float(metrics.get("metrics/mAP50(B)", 0)),
                "map50_95": float(metrics.get("metrics/mAP50-95(B)", 0)),
                "precision": float(metrics.get("metrics/precision(B)", 0)),
                "recall": float(metrics.get("metrics/recall(B)", 0)),
            }

        return summary

    def validate(self, split: str = "val") -> Dict:
        """
        Validate the model.

        Args:
            split: Split to validate on (val or test)

        Returns:
            Validation metrics
        """
        if not self.best_model_path or not self.best_model_path.exists():
            raise ValueError("No trained model found. Train the model first.")

        print(f"\n{'='*70}")
        print(f"Validating on {split} split")
        print(f"{'='*70}\n")

        # Load best model
        model = YOLO(str(self.best_model_path))

        # Validate
        metrics = model.val(
            data=str(self.data_config.dataset_yaml),
            split=split,
            imgsz=self.model_config.input_size,
            conf=self.model_config.confidence_threshold,
            iou=self.model_config.iou_threshold,
        )

        # Extract metrics
        results = {
            "split": split,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }

        # Per-class metrics if available
        if hasattr(metrics.box, "maps"):
            class_names = self.data_config.class_names if hasattr(self.data_config, "class_names") else []
            if class_names:
                results["per_class_map50"] = {
                    class_names[i]: float(metrics.box.maps[i])
                    for i in range(len(class_names))
                }

        print(f"\n{'='*70}")
        print(f"Validation Results ({split}):")
        print(f"  mAP50: {results['map50']:.4f}")
        print(f"  mAP50-95: {results['map50_95']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"{'='*70}\n")

        return results

    def export(
        self,
        format: str = "onnx",
        output_path: Optional[Path] = None,
        **export_kwargs,
    ) -> Path:
        """
        Export trained model to different format.

        Args:
            format: Export format (onnx, torchscript, coreml, etc.)
            output_path: Optional custom output path
            **export_kwargs: Additional export arguments

        Returns:
            Path to exported model
        """
        if not self.best_model_path or not self.best_model_path.exists():
            raise ValueError("No trained model found. Train the model first.")

        print(f"\n{'='*70}")
        print(f"Exporting model to {format.upper()}")
        print(f"{'='*70}\n")

        # Load best model
        model = YOLO(str(self.best_model_path))

        # Export
        export_path = model.export(
            format=format, imgsz=self.model_config.input_size, **export_kwargs
        )

        # Copy to custom location if specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy(export_path, output_path)
            export_path = output_path

        print(f"\n✅ Model exported: {export_path}\n")

        return Path(export_path)

    def save_config(self, output_path: Path):
        """
        Save training configuration to JSON.

        Args:
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config = {
            "model": {
                "model_name": self.model_config.model_name,
                "input_size": self.model_config.input_size,
                "confidence_threshold": self.model_config.confidence_threshold,
                "iou_threshold": self.model_config.iou_threshold,
            },
            "training": self.training_config.to_yolo_args(),
            "data": {
                "dataset_yaml": str(self.data_config.dataset_yaml),
                "class_weights": self.data_config.class_weights,
            },
        }

        if self.augmentation_config:
            config["augmentation"] = self.augmentation_config.to_dict()

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Configuration saved: {output_path}")
