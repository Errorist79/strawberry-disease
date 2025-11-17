"""
Class weight callback for handling class imbalance.

Applies class weights to BCE loss function via Ultralytics callback system.
This is the official workaround recommended by Ultralytics for handling class imbalance.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import yaml


class ClassWeightCallback:
    """Apply class weights to BCE loss via callback.

    This callback modifies the BCE (Binary Cross Entropy) loss function
    to use pos_weight parameter, which allows different weights for each class.

    Example:
        >>> class_weights = {
        ...     'anthracnose_fruit_rot': 3.0,
        ...     'powdery_mildew_fruit': 3.0,
        ...     'blossom_blight': 2.0,
        ... }
        >>> callback = ClassWeightCallback(class_weights, dataset_yaml='data.yaml')
        >>> model.add_callback("on_train_start", callback.on_train_start)
    """

    def __init__(
        self,
        class_weights: Dict[str, float],
        dataset_yaml: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize class weight callback.

        Args:
            class_weights: Dict mapping class names to weights
                          e.g., {'anthracnose_fruit_rot': 3.0, 'powdery_mildew_fruit': 3.0}
            dataset_yaml: Path to dataset YAML (to extract class names)
            class_names: Explicit list of class names (alternative to dataset_yaml)
        """
        self.class_weights = class_weights
        self.dataset_yaml = dataset_yaml
        self.class_names = class_names
        self.applied = False

    def on_train_start(self, trainer):
        """
        Apply weights when training starts.

        This is called by Ultralytics at the beginning of training.
        It modifies the model's BCE loss to use weighted loss.

        Args:
            trainer: Ultralytics trainer instance
        """
        if self.applied:
            return

        # Get class names from trainer's data config
        if self.class_names is None:
            if hasattr(trainer, 'data') and 'names' in trainer.data:
                self.class_names = list(trainer.data['names'].values())
            elif self.dataset_yaml:
                with open(self.dataset_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                    self.class_names = list(data.get('names', {}).values())
            else:
                raise ValueError(
                    "Cannot determine class names. "
                    "Provide either dataset_yaml or class_names parameter."
                )

        # Create weight tensor
        weight_tensor = self._create_weight_tensor(len(self.class_names))

        # Modify BCE loss with pos_weight
        model = trainer.model
        model.criterion = model.init_criterion()
        model.criterion.bce = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=weight_tensor.to(trainer.device)
        )

        self.applied = True

        # Print confirmation
        print(f"\n{'='*70}")
        print("✅ Class weights applied to BCE loss")
        print(f"{'='*70}")
        weighted_classes = [name for name in self.class_names if name in self.class_weights]
        for class_name in weighted_classes:
            weight = self.class_weights[class_name]
            print(f"  {class_name}: {weight}x")
        print(f"{'='*70}\n")

    def _create_weight_tensor(self, num_classes: int) -> torch.Tensor:
        """
        Convert class name -> weight dict to tensor.

        Args:
            num_classes: Total number of classes

        Returns:
            torch.Tensor: Weight tensor aligned with class indices
                         Shape: (num_classes,)
        """
        weights = []
        for i, class_name in enumerate(self.class_names):
            if i >= num_classes:
                break
            # Use specified weight or default to 1.0
            weight = self.class_weights.get(class_name, 1.0)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)


def create_class_weight_callback(
    class_weights: Dict[str, float],
    dataset_yaml: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> ClassWeightCallback:
    """
    Factory function to create class weight callback.

    Args:
        class_weights: Dict mapping class names to weights
        dataset_yaml: Path to dataset YAML
        class_names: Explicit list of class names

    Returns:
        ClassWeightCallback instance

    Example:
        >>> callback = create_class_weight_callback(
        ...     class_weights={'anthracnose_fruit_rot': 3.0},
        ...     dataset_yaml='data/processed/merged/data.yaml'
        ... )
        >>> model.add_callback("on_train_start", callback.on_train_start)
    """
    return ClassWeightCallback(
        class_weights=class_weights,
        dataset_yaml=dataset_yaml,
        class_names=class_names,
    )
