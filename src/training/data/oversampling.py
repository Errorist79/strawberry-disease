"""
Oversampling strategies for handling class imbalance.

Provides tools to create augmented copies of underrepresented classes.
"""

import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class OversamplingStrategy:
    """Handle oversampling of underrepresented classes."""

    def __init__(
        self,
        dataset_yaml: Path,
        oversample_config: Dict[str, int],
        output_dir: Path,
        seed: int = 42,
    ):
        """
        Initialize oversampling strategy.

        Args:
            dataset_yaml: Path to original dataset YAML
            oversample_config: Dict mapping class names to multipliers
                              e.g., {'anthracnose_fruit_rot': 3, 'powdery_mildew_fruit': 3}
            output_dir: Directory to save oversampled dataset
            seed: Random seed for reproducibility
        """
        self.dataset_yaml = Path(dataset_yaml)
        self.oversample_config = oversample_config
        self.output_dir = Path(output_dir)
        self.seed = seed

        random.seed(seed)

        # Load dataset config
        with open(self.dataset_yaml, "r") as f:
            self.config = yaml.safe_load(f)

        self.dataset_path = Path(self.config.get("path", self.dataset_yaml.parent))
        self.class_names = self.config.get("names", [])

        # Create class name to ID mapping
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

    def oversample_split(
        self, split: str = "train", vary_augmentation: bool = True
    ) -> Path:
        """
        Oversample a dataset split.

        Args:
            split: Split to oversample (train, val)
            vary_augmentation: Whether to vary copies (add slight variations)

        Returns:
            Path to oversampled split directory
        """
        # Get source split path
        split_rel_path = self.config.get(split)
        if not split_rel_path:
            raise ValueError(f"Split '{split}' not found in dataset config")

        src_split_path = self.dataset_path / split_rel_path

        # Support both split/images and split structure
        if (src_split_path / "images").exists():
            src_images_dir = src_split_path / "images"
            src_labels_dir = src_split_path / "labels"
        else:
            src_images_dir = src_split_path
            src_labels_dir = src_split_path.parent / "labels"

        # Create output directories
        dst_split_path = self.output_dir / split
        dst_images_dir = dst_split_path / "images"
        dst_labels_dir = dst_split_path / "labels"
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)

        # First, copy all original images and labels
        print(f"Copying original {split} split...")
        image_files = list(src_images_dir.glob("*.jpg")) + list(
            src_images_dir.glob("*.png")
        )

        for img_file in image_files:
            shutil.copy(img_file, dst_images_dir / img_file.name)

            label_file = src_labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy(label_file, dst_labels_dir / (img_file.stem + ".txt"))

        # Now, find images containing target classes and create copies
        print(f"Oversampling underrepresented classes...")
        oversample_stats = {class_name: 0 for class_name in self.oversample_config}

        for img_file in image_files:
            label_file = src_labels_dir / (img_file.stem + ".txt")

            if not label_file.exists():
                continue

            # Read label file to check for target classes
            with open(label_file, "r") as f:
                lines = f.readlines()

            # Check if image contains any target class
            contains_target = False
            target_classes = set()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if 0 <= class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                        if class_name in self.oversample_config:
                            contains_target = True
                            target_classes.add(class_name)

            # If contains target class, create additional copies
            if contains_target:
                # Determine multiplier (use max if multiple target classes)
                max_multiplier = max(
                    self.oversample_config[cls] for cls in target_classes
                )

                # Create (multiplier - 1) additional copies (original already exists)
                for copy_idx in range(1, max_multiplier):
                    # Create copy with suffix
                    copy_stem = f"{img_file.stem}_copy{copy_idx}"
                    copy_img_path = dst_images_dir / f"{copy_stem}{img_file.suffix}"
                    copy_label_path = dst_labels_dir / f"{copy_stem}.txt"

                    # Copy image and label
                    shutil.copy(img_file, copy_img_path)
                    shutil.copy(label_file, copy_label_path)

                    # Update stats
                    for cls in target_classes:
                        oversample_stats[cls] += 1

        # Print statistics
        print(f"\nOversampling statistics for {split}:")
        for class_name, count in oversample_stats.items():
            multiplier = self.oversample_config[class_name]
            print(
                f"  {class_name}: {count} additional copies "
                f"(target multiplier: {multiplier}x)"
            )

        return dst_split_path

    def create_oversampled_dataset(
        self, splits: List[str] = None, vary_augmentation: bool = True
    ) -> Path:
        """
        Create complete oversampled dataset.

        Args:
            splits: List of splits to oversample (default: ['train'])
            vary_augmentation: Whether to vary copies

        Returns:
            Path to new dataset.yaml
        """
        if splits is None:
            splits = ["train"]  # Only oversample training set by default

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Oversample specified splits
        for split in splits:
            if split in self.config:
                self.oversample_split(split, vary_augmentation)

        # Copy non-oversampled splits
        for split in ["train", "val", "test"]:
            if split not in splits and split in self.config:
                src_split_path = self.dataset_path / self.config[split]

                if src_split_path.exists():
                    dst_split_path = self.output_dir / split
                    if not dst_split_path.exists():
                        print(f"Copying {split} split without oversampling...")
                        shutil.copytree(src_split_path, dst_split_path)

        # Create new dataset.yaml
        new_config = self.config.copy()
        new_config["path"] = str(self.output_dir.absolute())

        output_yaml = self.output_dir / "dataset.yaml"
        with open(output_yaml, "w") as f:
            yaml.dump(new_config, f, default_flow_style=False)

        print(f"\n✅ Oversampled dataset created: {output_yaml}")

        return output_yaml


def calculate_dynamic_oversample_ratios(
    dataset_yaml: Path,
    target_balance: float = 0.8,
    max_multiplier: int = 5,
) -> Dict[str, int]:
    """
    Calculate dynamic oversampling ratios based on class distribution.

    Automatically determines multipliers for underrepresented classes.

    Args:
        dataset_yaml: Path to dataset YAML
        target_balance: Target balance ratio (0-1). Each class should have
                       at least target_balance * average instances
        max_multiplier: Maximum multiplier to apply

    Returns:
        Dict mapping class names to multipliers

    Example:
        >>> ratios = calculate_dynamic_oversample_ratios(
        ...     'data/merged/dataset.yaml',
        ...     target_balance=0.8,
        ...     max_multiplier=5
        ... )
        >>> # {'anthracnose_fruit_rot': 5, 'powdery_mildew_fruit': 4, ...}
    """
    from .dataset_loader import DatasetLoader

    loader = DatasetLoader(dataset_yaml)
    validation = loader.validate()

    if 'train' not in validation['stats']:
        print("Warning: No training split found, cannot calculate ratios")
        return {}

    class_dist = validation['stats']['train']['class_distribution']
    total_instances = validation['stats']['train']['total_instances']

    if not class_dist:
        return {}

    # Calculate target instances per class
    num_classes = len(class_dist)
    avg_instances = total_instances / num_classes
    target_instances = avg_instances * target_balance

    # Calculate multipliers
    oversample_config = {}

    for class_name, count in class_dist.items():
        if count < target_instances:
            # Calculate required multiplier
            required_multiplier = target_instances / count
            multiplier = min(int(required_multiplier) + 1, max_multiplier)

            if multiplier > 1:
                oversample_config[class_name] = multiplier

    print(f"\nDynamic Oversampling Ratios:")
    print(f"  Target balance: {target_balance}")
    print(f"  Average instances: {avg_instances:.0f}")
    print(f"  Target minimum: {target_instances:.0f}")
    print(f"\nCalculated multipliers:")

    for class_name, multiplier in sorted(
        oversample_config.items(), key=lambda x: x[1], reverse=True
    ):
        original_count = class_dist[class_name]
        new_count = original_count * multiplier
        print(f"  {class_name}: {multiplier}x ({original_count} → {new_count})")

    return oversample_config


def create_oversampled_dataset(
    dataset_yaml: Path,
    oversample_config: Dict[str, int] = None,
    output_dir: Path = None,
    splits: List[str] = None,
    seed: int = 42,
    dynamic: bool = False,
    target_balance: float = 0.8,
    max_multiplier: int = 5,
) -> Path:
    """
    Create oversampled dataset (convenience function).

    Args:
        dataset_yaml: Path to original dataset YAML
        oversample_config: Dict mapping class names to multipliers
                          If None and dynamic=True, will be calculated automatically
        output_dir: Output directory
        splits: Splits to oversample (default: ['train'])
        seed: Random seed
        dynamic: If True, calculate oversample ratios automatically
        target_balance: For dynamic mode, target balance ratio (0-1)
        max_multiplier: For dynamic mode, maximum multiplier

    Returns:
        Path to oversampled dataset.yaml

    Example:
        >>> # Manual oversampling
        >>> oversample_config = {
        ...     'anthracnose_fruit_rot': 3,
        ...     'powdery_mildew_fruit': 3,
        ... }
        >>> new_yaml = create_oversampled_dataset(
        ...     'data/raw/dataset.yaml',
        ...     oversample_config,
        ...     'data/processed/oversampled'
        ... )

        >>> # Dynamic oversampling
        >>> new_yaml = create_oversampled_dataset(
        ...     'data/merged/dataset.yaml',
        ...     output_dir='data/processed/oversampled',
        ...     dynamic=True,
        ...     target_balance=0.8,
        ...     max_multiplier=5
        ... )
    """
    # Calculate dynamic ratios if requested
    if dynamic and oversample_config is None:
        oversample_config = calculate_dynamic_oversample_ratios(
            dataset_yaml=dataset_yaml,
            target_balance=target_balance,
            max_multiplier=max_multiplier,
        )

        if not oversample_config:
            print("Warning: No oversampling needed (dataset is balanced)")
            # Just copy dataset
            import shutil

            if output_dir:
                output_dir = Path(output_dir)
                dataset_path = Path(dataset_yaml).parent
                shutil.copytree(dataset_path, output_dir, dirs_exist_ok=True)
                return output_dir / 'dataset.yaml'
            return dataset_yaml

    if not oversample_config:
        raise ValueError("oversample_config must be provided or dynamic=True must be set")

    if not output_dir:
        raise ValueError("output_dir must be provided")

    strategy = OversamplingStrategy(
        dataset_yaml=dataset_yaml,
        oversample_config=oversample_config,
        output_dir=output_dir,
        seed=seed,
    )

    return strategy.create_oversampled_dataset(splits=splits)
