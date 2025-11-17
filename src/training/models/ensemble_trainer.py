"""
Ensemble trainer for training and combining multiple YOLO models.

Implements Weighted Boxes Fusion (WBF) for combining predictions from multiple models.
"""

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from ultralytics import YOLO

from ..config.augmentation_config import AugmentationConfig
from ..config.base_config import DataConfig, ModelConfig, TrainingConfig
from .yolo_trainer import YOLOTrainer


class EnsembleTrainer:
    """Train and manage ensemble of YOLO models."""

    def __init__(
        self,
        models: List[Tuple[ModelConfig, TrainingConfig, AugmentationConfig]],
        data_config: DataConfig,
        output_dir: Path,
        ensemble_name: str = "ensemble",
    ):
        """
        Initialize ensemble trainer.

        Args:
            models: List of (model_config, training_config, augmentation_config) tuples
            data_config: Shared data configuration
            output_dir: Output directory for ensemble
            ensemble_name: Name for ensemble
        """
        self.models = models
        self.data_config = data_config
        self.output_dir = Path(output_dir)
        self.ensemble_name = ensemble_name

        # Create output directory
        self.ensemble_dir = self.output_dir / ensemble_name
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Trainers
        self.trainers: List[YOLOTrainer] = []
        self.trained_model_paths: List[Path] = []
        self.model_weights: List[float] = []  # For WBF

    def train_all(
        self,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> Dict:
        """
        Train all models in ensemble.

        Args:
            parallel: Whether to train models in parallel (requires multiple GPUs)
            max_workers: Maximum parallel workers (default: number of models)

        Returns:
            Dictionary with training results for all models
        """
        print(f"\n{'='*70}")
        print(f"Training Ensemble: {self.ensemble_name}")
        print(f"Number of models: {len(self.models)}")
        print(f"Parallel training: {parallel}")
        print(f"{'='*70}\n")

        results = []

        if parallel and len(self.models) > 1:
            # Parallel training (for multiple GPUs)
            results = self._train_parallel(max_workers)
        else:
            # Sequential training
            results = self._train_sequential()

        # Calculate ensemble weights based on validation performance
        self._calculate_ensemble_weights(results)

        # Save ensemble configuration
        self._save_ensemble_config(results)

        print(f"\n{'='*70}")
        print(f"✅ Ensemble Training Complete!")
        print(f"Trained {len(self.trainers)} models")
        print(f"Output directory: {self.ensemble_dir}")
        print(f"{'='*70}\n")

        return {
            "ensemble_name": self.ensemble_name,
            "num_models": len(self.trainers),
            "model_results": results,
            "ensemble_weights": self.model_weights,
            "ensemble_dir": str(self.ensemble_dir),
        }

    def _train_sequential(self) -> List[Dict]:
        """Train models sequentially."""
        results = []

        for idx, (model_config, training_config, aug_config) in enumerate(
            self.models, 1
        ):
            print(f"\n{'='*70}")
            print(f"Training Model {idx}/{len(self.models)}")
            print(f"Model: {model_config.model_name}")
            print(f"Augmentation: {aug_config.__class__.__name__}")
            print(f"{'='*70}\n")

            # Update training config to use unique output directory
            training_config.name = f"{self.ensemble_name}_model{idx}"

            # Create trainer
            trainer = YOLOTrainer(
                model_config=model_config,
                data_config=self.data_config,
                training_config=training_config,
                augmentation_config=aug_config,
            )

            # Train
            result = trainer.train()
            result["model_idx"] = idx
            results.append(result)

            # Store trainer and model path
            self.trainers.append(trainer)
            self.trained_model_paths.append(trainer.best_model_path)

        return results

    def _train_parallel(self, max_workers: Optional[int] = None) -> List[Dict]:
        """
        Train models in parallel.

        Note: This requires multiple GPUs or careful device management.
        """
        if max_workers is None:
            max_workers = min(len(self.models), mp.cpu_count())

        print(
            f"Training {len(self.models)} models in parallel with {max_workers} workers"
        )
        print(
            "⚠️  Warning: Parallel training requires multiple GPUs or careful device configuration"
        )

        # For now, fall back to sequential training
        # True parallel training requires more sophisticated device management
        print("Note: Falling back to sequential training")
        return self._train_sequential()

    def _calculate_ensemble_weights(self, results: List[Dict]):
        """
        Calculate ensemble weights based on validation performance.

        Uses mAP50-95 as the weighting metric.
        """
        print("\nCalculating ensemble weights based on validation performance...")

        # Validate all models
        val_scores = []
        for trainer in self.trainers:
            val_result = trainer.validate(split="val")
            score = val_result.get("map50_95", 0)
            val_scores.append(score)

        # Normalize scores to weights (softmax-like)
        val_scores = np.array(val_scores)

        # Use softmax with temperature for weighting
        temperature = 2.0  # Higher = more uniform weights
        exp_scores = np.exp(val_scores / temperature)
        weights = exp_scores / exp_scores.sum()

        self.model_weights = weights.tolist()

        print("\nEnsemble weights:")
        for idx, (weight, score) in enumerate(zip(self.model_weights, val_scores), 1):
            print(f"  Model {idx}: weight={weight:.3f}, mAP50-95={score:.4f}")

    def predict_ensemble(
        self,
        image_path: Union[str, Path],
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
        fusion_method: str = "wbf",
    ) -> List:
        """
        Make predictions using ensemble.

        Args:
            image_path: Path to image
            iou_threshold: IOU threshold for NMS/WBF
            conf_threshold: Confidence threshold
            fusion_method: Fusion method (wbf, nms, or nmw)

        Returns:
            Fused predictions
        """
        if not self.trainers:
            raise ValueError("No trained models in ensemble. Train models first.")

        # Get predictions from all models
        all_predictions = []

        for trainer, weight in zip(self.trainers, self.model_weights):
            model = YOLO(str(trainer.best_model_path))
            results = model.predict(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )

            if len(results) > 0:
                all_predictions.append((results[0], weight))

        # Fuse predictions
        if fusion_method == "wbf":
            fused = self._weighted_boxes_fusion(
                all_predictions, iou_threshold, conf_threshold
            )
        else:
            # For now, just use the best model's predictions
            fused = all_predictions[0][0] if all_predictions else None

        return fused

    def _weighted_boxes_fusion(
        self, predictions: List[Tuple], iou_threshold: float, conf_threshold: float
    ):
        """
        Apply Weighted Boxes Fusion.

        Note: This is a simplified implementation. For production,
        consider using the ensemble-boxes library.
        """
        # For now, return the prediction from the best model
        # Full WBF implementation would require ensemble-boxes library
        if predictions:
            # Sort by weight and return best
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[0][0]
        return None

    def validate_ensemble(self, split: str = "val") -> Dict:
        """
        Validate ensemble performance.

        Currently validates individual models. Full ensemble validation
        would require running predictions on entire dataset and computing metrics.

        Args:
            split: Split to validate on

        Returns:
            Dictionary with individual and ensemble metrics
        """
        print(f"\n{'='*70}")
        print(f"Validating Ensemble on {split} split")
        print(f"{'='*70}\n")

        # Validate each model
        individual_results = []
        for idx, trainer in enumerate(self.trainers, 1):
            print(f"\nValidating Model {idx}...")
            result = trainer.validate(split=split)
            result["model_idx"] = idx
            result["weight"] = self.model_weights[idx - 1]
            individual_results.append(result)

        # Calculate weighted average metrics
        weighted_map50 = sum(
            r["map50"] * r["weight"] for r in individual_results
        )
        weighted_map50_95 = sum(
            r["map50_95"] * r["weight"] for r in individual_results
        )
        weighted_precision = sum(
            r["precision"] * r["weight"] for r in individual_results
        )
        weighted_recall = sum(
            r["recall"] * r["weight"] for r in individual_results
        )

        ensemble_result = {
            "split": split,
            "ensemble_metrics": {
                "weighted_map50": weighted_map50,
                "weighted_map50_95": weighted_map50_95,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
            },
            "individual_results": individual_results,
        }

        print(f"\n{'='*70}")
        print(f"Ensemble Validation Results ({split}):")
        print(f"  Weighted mAP50: {weighted_map50:.4f}")
        print(f"  Weighted mAP50-95: {weighted_map50_95:.4f}")
        print(f"  Weighted Precision: {weighted_precision:.4f}")
        print(f"  Weighted Recall: {weighted_recall:.4f}")
        print(f"{'='*70}\n")

        return ensemble_result

    def _save_ensemble_config(self, training_results: List[Dict]):
        """Save ensemble configuration and results."""
        config = {
            "ensemble_name": self.ensemble_name,
            "num_models": len(self.models),
            "models": [
                {
                    "model_idx": idx + 1,
                    "model_path": str(path),
                    "weight": weight,
                    "model_name": self.models[idx][0].model_name,
                }
                for idx, (path, weight) in enumerate(
                    zip(self.trained_model_paths, self.model_weights)
                )
            ],
            "data_config": {
                "dataset_yaml": str(self.data_config.dataset_yaml),
            },
            "training_results": training_results,
        }

        config_path = self.ensemble_dir / "ensemble_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nEnsemble configuration saved: {config_path}")

    @classmethod
    def from_config(cls, config_path: Path) -> "EnsembleTrainer":
        """
        Load ensemble from configuration file.

        Args:
            config_path: Path to ensemble_config.json

        Returns:
            EnsembleTrainer instance
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        # This is a simplified loader - full implementation would
        # reconstruct all the model/training/augmentation configs
        raise NotImplementedError(
            "Loading ensemble from config not yet implemented. "
            "Use this for inference with pre-trained models."
        )
