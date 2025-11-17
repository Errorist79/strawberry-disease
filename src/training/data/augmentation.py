"""
Custom augmentation utilities.

Provides additional augmentation capabilities beyond YOLO's built-in augmentations.
"""

from pathlib import Path
from typing import Dict, Optional

from ..config.augmentation_config import AugmentationConfig


def apply_augmentation(
    model_train_args: Dict, augmentation_config: AugmentationConfig
) -> Dict:
    """
    Apply augmentation configuration to YOLO training arguments.

    Args:
        model_train_args: Existing training arguments dictionary
        augmentation_config: Augmentation configuration

    Returns:
        Updated training arguments with augmentation parameters
    """
    # Convert augmentation config to dict and merge
    aug_dict = augmentation_config.to_dict()
    model_train_args.update(aug_dict)

    return model_train_args


def create_custom_augmentation_yaml(
    augmentation_config: AugmentationConfig, output_path: Path
) -> Path:
    """
    Create a YAML file with custom augmentation parameters.

    This can be used with YOLO's cfg parameter for more control.

    Args:
        augmentation_config: Augmentation configuration
        output_path: Path to save YAML file

    Returns:
        Path to created YAML file
    """
    import yaml

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict
    aug_dict = augmentation_config.to_dict()

    # Save to YAML
    with open(output_path, "w") as f:
        yaml.dump(aug_dict, f, default_flow_style=False)

    return output_path
