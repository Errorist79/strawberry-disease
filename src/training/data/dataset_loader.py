"""
Dataset loading and validation utilities.

Provides tools for loading YOLO datasets and validating their structure.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


class DatasetLoader:
    """Load and validate YOLO format datasets."""

    def __init__(self, dataset_yaml: Path):
        """
        Initialize dataset loader.

        Args:
            dataset_yaml: Path to dataset.yaml file
        """
        self.dataset_yaml = Path(dataset_yaml)
        if not self.dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {self.dataset_yaml}")

        with open(self.dataset_yaml, "r") as f:
            self.config = yaml.safe_load(f)

        self.dataset_path = Path(self.config.get("path", self.dataset_yaml.parent))
        self.class_names = self.config.get("names", [])
        self.num_classes = self.config.get("nc", len(self.class_names))

    def get_split_path(self, split: str) -> Path:
        """
        Get path to a dataset split.

        Args:
            split: Split name (train, val, test)

        Returns:
            Path to split directory
        """
        if split not in self.config:
            raise ValueError(f"Split '{split}' not found in dataset config")

        split_rel_path = self.config[split]
        return self.dataset_path / split_rel_path

    def validate(self) -> Dict[str, any]:
        """
        Validate dataset structure and content.

        Returns:
            Dictionary with validation results and statistics
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        # Check dataset path
        if not self.dataset_path.exists():
            results["valid"] = False
            results["errors"].append(f"Dataset path not found: {self.dataset_path}")
            return results

        # Validate each split
        for split_name in ["train", "val", "test"]:
            if split_name not in self.config:
                if split_name != "test":  # test is optional
                    results["warnings"].append(f"Split '{split_name}' not found")
                continue

            try:
                split_path = self.get_split_path(split_name)
                split_stats = self._validate_split(split_path, split_name)
                results["stats"][split_name] = split_stats

                # Check for empty splits
                if split_stats["num_images"] == 0:
                    results["errors"].append(f"No images found in {split_name} split")
                    results["valid"] = False

                # Check for missing labels
                if split_stats["images_without_labels"] > 0:
                    results["warnings"].append(
                        f"{split_name}: {split_stats['images_without_labels']} "
                        f"images without labels"
                    )

            except Exception as e:
                results["valid"] = False
                results["errors"].append(f"Error validating {split_name}: {str(e)}")

        # Check class distribution
        if "train" in results["stats"]:
            train_stats = results["stats"]["train"]
            class_dist = train_stats.get("class_distribution", {})

            # Warn about imbalanced classes
            if class_dist:
                max_count = max(class_dist.values())
                min_count = min(class_dist.values())
                if max_count / min_count > 10:
                    results["warnings"].append(
                        f"Severe class imbalance detected (ratio: {max_count/min_count:.1f}:1). "
                        f"Consider using class weights or oversampling."
                    )

        return results

    def _validate_split(self, split_path: Path, split_name: str) -> Dict:
        """Validate a single dataset split."""
        stats = {
            "num_images": 0,
            "num_labels": 0,
            "images_without_labels": 0,
            "labels_without_images": 0,
            "class_distribution": {},
            "total_instances": 0,
        }

        # Find images and labels directories
        # Support both split/images and split structure
        if (split_path / "images").exists():
            images_dir = split_path / "images"
            labels_dir = split_path / "labels"
        else:
            images_dir = split_path
            labels_dir = split_path.parent / "labels"

        if not images_dir.exists():
            return stats

        # Get all images
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))

        stats["num_images"] = len(image_files)

        # Check for corresponding labels
        if labels_dir.exists():
            for img_file in image_files:
                label_file = labels_dir / (img_file.stem + ".txt")

                if not label_file.exists():
                    stats["images_without_labels"] += 1
                else:
                    stats["num_labels"] += 1

                    # Parse label file for class distribution
                    try:
                        with open(label_file, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:  # class x y w h
                                    class_id = int(parts[0])
                                    if 0 <= class_id < self.num_classes:
                                        class_name = self.class_names[class_id]
                                        stats["class_distribution"][class_name] = (
                                            stats["class_distribution"].get(
                                                class_name, 0
                                            )
                                            + 1
                                        )
                                        stats["total_instances"] += 1
                    except Exception as e:
                        # Skip invalid label files
                        pass

        return stats

    def get_class_weights(
        self, method: str = "inverse", power: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate class weights based on training set distribution.

        Args:
            method: Weighting method (inverse, sqrt_inverse)
            power: Power to raise weights to (for fine-tuning)

        Returns:
            Dictionary mapping class names to weights
        """
        # First validate to get class distribution
        validation = self.validate()

        if "train" not in validation["stats"]:
            raise ValueError("Training split not found, cannot calculate class weights")

        class_dist = validation["stats"]["train"]["class_distribution"]

        if not class_dist:
            raise ValueError("No class distribution found in training set")

        # Calculate weights
        total_instances = sum(class_dist.values())
        num_classes = len(class_dist)

        weights = {}
        for class_name, count in class_dist.items():
            if method == "inverse":
                # Inverse frequency
                weight = total_instances / (count * num_classes)
            elif method == "sqrt_inverse":
                # Square root of inverse frequency (less aggressive)
                weight = (total_instances / (count * num_classes)) ** 0.5
            else:
                raise ValueError(f"Unknown weighting method: {method}")

            # Apply power
            weights[class_name] = weight**power

        return weights

    def create_subset(
        self, output_path: Path, train_fraction: float = 0.1, val_fraction: float = 0.1
    ) -> Path:
        """
        Create a small subset of the dataset for quick testing.

        Args:
            output_path: Output directory
            train_fraction: Fraction of training data to include
            val_fraction: Fraction of validation data to include

        Returns:
            Path to subset dataset.yaml
        """
        import random

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        subset_config = self.config.copy()
        subset_config["path"] = str(output_path.absolute())

        for split_name, fraction in [
            ("train", train_fraction),
            ("val", val_fraction),
        ]:
            if split_name not in self.config:
                continue

            # Get source split
            src_split = self.get_split_path(split_name)
            if not src_split.exists():
                continue

            # Create destination
            dst_split = output_path / split_name
            dst_images = dst_split / "images"
            dst_labels = dst_split / "labels"
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)

            # Get images
            src_images = src_split / "images"
            src_labels = src_split / "labels"

            image_files = list(src_images.glob("*.jpg")) + list(
                src_images.glob("*.png")
            )
            num_to_select = max(1, int(len(image_files) * fraction))

            # Randomly select images
            selected = random.sample(image_files, num_to_select)

            # Copy images and labels
            for img_file in selected:
                shutil.copy(img_file, dst_images / img_file.name)

                label_file = src_labels / (img_file.stem + ".txt")
                if label_file.exists():
                    shutil.copy(label_file, dst_labels / (img_file.stem + ".txt"))

        # Save config
        output_yaml = output_path / "data.yaml"
        with open(output_yaml, "w") as f:
            yaml.dump(subset_config, f)

        return output_yaml


def validate_dataset(dataset_yaml: Path, verbose: bool = True) -> bool:
    """
    Validate a YOLO dataset.

    Args:
        dataset_yaml: Path to dataset YAML
        verbose: Print validation results

    Returns:
        True if valid, False otherwise
    """
    loader = DatasetLoader(dataset_yaml)
    results = loader.validate()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset Validation: {dataset_yaml}")
        print(f"{'='*60}")

        # Print errors
        if results["errors"]:
            print("\n❌ ERRORS:")
            for error in results["errors"]:
                print(f"  - {error}")

        # Print warnings
        if results["warnings"]:
            print("\n⚠️  WARNINGS:")
            for warning in results["warnings"]:
                print(f"  - {warning}")

        # Print statistics
        if results["stats"]:
            print("\n📊 STATISTICS:")
            for split_name, stats in results["stats"].items():
                print(f"\n  {split_name.upper()}:")
                print(f"    Images: {stats['num_images']}")
                print(f"    Labels: {stats['num_labels']}")
                print(f"    Instances: {stats['total_instances']}")

                if stats["class_distribution"]:
                    print(f"    Class distribution:")
                    for class_name, count in sorted(
                        stats["class_distribution"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    ):
                        percentage = (
                            100 * count / stats["total_instances"]
                            if stats["total_instances"] > 0
                            else 0
                        )
                        print(f"      {class_name}: {count} ({percentage:.1f}%)")

        # Print overall result
        print(f"\n{'='*60}")
        if results["valid"]:
            print("✅ Dataset is VALID")
        else:
            print("❌ Dataset is INVALID")
        print(f"{'='*60}\n")

    return results["valid"]
