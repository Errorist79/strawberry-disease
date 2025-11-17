"""
Oversampling strategies for handling class imbalance.

Provides tools to create augmented copies of underrepresented classes.
"""

import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import albumentations as A
import cv2
import numpy as np
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
        np.random.seed(seed)

        # Load dataset config
        with open(self.dataset_yaml, "r") as f:
            self.config = yaml.safe_load(f)

        self.dataset_path = Path(self.config.get("path", self.dataset_yaml.parent))
        self.class_names = self.config.get("names", [])

        # Create class name to ID mapping
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

        # Initialize augmentation transforms
        self._init_augmentations()

    def _validate_bbox(self, bbox: List[float]) -> bool:
        """
        Validate YOLO format bounding box.

        Args:
            bbox: [x_center, y_center, width, height] in normalized coordinates

        Returns:
            True if valid, False otherwise
        """
        if len(bbox) < 4:
            return False

        x_center, y_center, width, height = bbox[:4]

        # Check if values are in valid range [0, 1]
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            return False

        # Check if width and height are positive and <= 1
        if not (0 < width <= 1 and 0 < height <= 1):
            return False

        # Check if bbox is within image bounds
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2

        if not (0 <= x_min < x_max <= 1 and 0 <= y_min < y_max <= 1):
            return False

        return True

    def _init_augmentations(self):
        """Initialize augmentation pipelines for creating varied copies."""
        # Bbox parameters with clipping to handle edge cases
        bbox_params = A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=0.0,
            min_visibility=0.3,  # Keep bbox if at least 30% visible
            clip=True  # Clip bboxes to valid range
        )

        # Different augmentation presets for variety
        self.augmentation_presets = [
            # Preset 1: Horizontal flip + slight rotation (safe)
            A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-15, 15),
                    shear=(-5, 5),
                    p=0.5
                ),
            ], bbox_params=bbox_params),

            # Preset 2: Brightness/Contrast (no geometric changes, no bbox processing needed)
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            ]),

            # Preset 3: Rotation + blur
            A.Compose([
                A.Rotate(limit=15, p=0.6, border_mode=0),  # Reduced rotation
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=1.0),
                    A.GaussianBlur(blur_limit=3, p=1.0),
                ], p=0.3),
            ], bbox_params=bbox_params),

            # Preset 4: Color jitter + noise + flip
            A.Compose([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(std_range=(0.1, 0.3), per_channel=True, p=0.3),
                A.VerticalFlip(p=0.3),
            ], bbox_params=bbox_params),

            # Preset 5: Geometric + lighting (safer parameters)
            A.Compose([
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-20, 20),
                    p=0.5
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ], bbox_params=bbox_params),
        ]

    def _apply_augmentation(self, image: np.ndarray, bboxes: List, class_labels: List, preset_idx: int) -> tuple:
        """
        Apply augmentation to image and bounding boxes.

        Args:
            image: Input image (numpy array)
            bboxes: List of bounding boxes in YOLO format [[x_center, y_center, width, height], ...]
            class_labels: List of class IDs corresponding to bboxes
            preset_idx: Index of augmentation preset to use

        Returns:
            Tuple of (augmented_image, augmented_bboxes, class_labels)
        """
        preset = self.augmentation_presets[preset_idx % len(self.augmentation_presets)]

        # Apply augmentation
        try:
            # Check if this preset processes bboxes
            has_bbox_processor = hasattr(preset, 'processors') and 'bboxes' in preset.processors

            if not bboxes or not has_bbox_processor:
                # No bboxes OR color-only preset - just augment image
                augmented = preset(image=image)
                return augmented['image'], bboxes, class_labels
            else:
                # Preset has bbox processing and we have bboxes
                augmented = preset(image=image, bboxes=bboxes, class_labels=class_labels)

                # Check if we still have bounding boxes after augmentation
                if not augmented.get('bboxes'):
                    # All bboxes were clipped out - use original
                    return image, bboxes, class_labels

                return augmented['image'], augmented['bboxes'], augmented['class_labels']
        except Exception as e:
            # If augmentation fails, return original (silent for common bbox errors)
            if "Expected" not in str(e) and "x_max" not in str(e) and "label_fields" not in str(e):
                print(f"Warning: Unexpected augmentation error: {e}")
            return image, bboxes, class_labels

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
        augmentation_stats = {'succeeded': 0, 'fallback': 0, 'total': 0}

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
                # Only accept bounding box format (class x y w h = exactly 5 values)
                # Filter out segmentation polygons (>5 values) for detection task
                if len(parts) == 5:
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

                # Load image once for augmentation
                if vary_augmentation:
                    image = cv2.imread(str(img_file))
                    if image is None:
                        print(f"Warning: Could not read image {img_file}, skipping augmentation")
                        vary_augmentation_for_this_image = False
                    else:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        vary_augmentation_for_this_image = True

                        # Parse and validate bounding boxes from label file
                        bboxes = []
                        class_labels = []
                        invalid_bbox_count = 0

                        for line in lines:
                            parts = line.strip().split()
                            # Only accept bounding box format (class x y w h = exactly 5 values)
                            # Filter out segmentation polygons (>5 values) for detection task
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                x_center, y_center, width, height = map(float, parts[1:5])
                                bbox = [x_center, y_center, width, height]

                                # Validate bbox before adding
                                if self._validate_bbox(bbox):
                                    bboxes.append(bbox)
                                    class_labels.append(class_id)
                                else:
                                    invalid_bbox_count += 1

                        # Warn if invalid bboxes were found
                        if invalid_bbox_count > 0:
                            print(f"  Skipped {invalid_bbox_count} invalid bbox(es) in {img_file.name}")
                else:
                    vary_augmentation_for_this_image = False

                # Create (multiplier - 1) additional copies (original already exists)
                for copy_idx in range(1, max_multiplier):
                    # Create copy with suffix
                    copy_stem = f"{img_file.stem}_copy{copy_idx}"
                    copy_img_path = dst_images_dir / f"{copy_stem}{img_file.suffix}"
                    copy_label_path = dst_labels_dir / f"{copy_stem}.txt"

                    if vary_augmentation_for_this_image:
                        # Apply augmentation
                        original_image_copy = image.copy()
                        aug_image, aug_bboxes, aug_class_labels = self._apply_augmentation(
                            image.copy(), bboxes.copy(), class_labels.copy(), copy_idx
                        )

                        # Track augmentation success
                        augmentation_stats['total'] += 1
                        # Check if image was actually augmented (not just bbox comparison)
                        # Color augmentation doesn't change bboxes but still augments image
                        image_changed = not np.array_equal(aug_image, original_image_copy)
                        if image_changed:
                            augmentation_stats['succeeded'] += 1
                        else:
                            augmentation_stats['fallback'] += 1

                        # Save augmented image
                        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(copy_img_path), aug_image_bgr)

                        # Save augmented labels
                        with open(copy_label_path, 'w') as f:
                            for class_id, bbox in zip(aug_class_labels, aug_bboxes):
                                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                    else:
                        # Just copy without augmentation
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

        # Print augmentation statistics if enabled
        if vary_augmentation and augmentation_stats['total'] > 0:
            print(f"\nAugmentation statistics:")
            print(f"  Total copies created: {augmentation_stats['total']}")
            print(f"  Successfully augmented: {augmentation_stats['succeeded']} "
                  f"({100*augmentation_stats['succeeded']/augmentation_stats['total']:.1f}%)")
            print(f"  Used original (fallback): {augmentation_stats['fallback']} "
                  f"({100*augmentation_stats['fallback']/augmentation_stats['total']:.1f}%)")

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
                split_rel_path = self.config[split]
                src_split_path = self.dataset_path / split_rel_path

                # Determine source structure
                # Handle both "split/images" and "split" paths in YAML
                if split_rel_path.endswith('/images'):
                    # YAML has "split/images", so parent is the split dir
                    src_base = src_split_path.parent
                else:
                    src_base = src_split_path

                if src_base.exists():
                    dst_split_path = self.output_dir / split
                    if not dst_split_path.exists():
                        print(f"Copying {split} split without oversampling...")
                        shutil.copytree(src_base, dst_split_path)

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
    target_balance: float = 0.65,
    max_multiplier: int = 5,
    min_multiplier: int = 2,
) -> Dict[str, int]:
    """
    Calculate dynamic oversampling ratios based on class distribution.

    Automatically determines multipliers for underrepresented classes.

    Args:
        dataset_yaml: Path to dataset YAML
        target_balance: Target balance ratio (0-1). Each class should have
                       at least target_balance * average instances
                       Default: 0.65 (more conservative to prevent overfitting)
        max_multiplier: Maximum multiplier to apply (default: 5)
        min_multiplier: Minimum multiplier to apply (default: 2)
                       Classes requiring less than this won't be oversampled

    Returns:
        Dict mapping class names to multipliers

    Example:
        >>> ratios = calculate_dynamic_oversample_ratios(
        ...     'data/merged/dataset.yaml',
        ...     target_balance=0.65,
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

            # Only apply if meets minimum threshold
            if multiplier >= min_multiplier:
                oversample_config[class_name] = multiplier

    print(f"\nDynamic Oversampling Ratios:")
    print(f"  Target balance: {target_balance}")
    print(f"  Average instances: {avg_instances:.0f}")
    print(f"  Target minimum: {target_instances:.0f}")
    print(f"  Min multiplier: {min_multiplier}x")
    print(f"  Max multiplier: {max_multiplier}x")
    print(f"\nCalculated multipliers:")

    for class_name, multiplier in sorted(
        oversample_config.items(), key=lambda x: x[1], reverse=True
    ):
        original_count = class_dist[class_name]
        new_count = original_count * multiplier
        print(f"  {class_name}: {multiplier}x ({original_count} → {new_count})")

    if not oversample_config:
        print("  No classes require oversampling with current parameters.")

    return oversample_config


def create_oversampled_dataset(
    dataset_yaml: Path,
    oversample_config: Dict[str, int] = None,
    output_dir: Path = None,
    splits: List[str] = None,
    seed: int = 42,
    dynamic: bool = False,
    target_balance: float = 0.65,
    max_multiplier: int = 5,
    min_multiplier: int = 2,
    vary_augmentation: bool = True,
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
        target_balance: For dynamic mode, target balance ratio (0-1). Default: 0.65
        max_multiplier: For dynamic mode, maximum multiplier. Default: 5
        min_multiplier: For dynamic mode, minimum multiplier. Default: 2
        vary_augmentation: If True, apply different augmentations to each copy. Default: True

    Returns:
        Path to oversampled dataset.yaml

    Example:
        >>> # Manual oversampling with augmentation
        >>> oversample_config = {
        ...     'anthracnose_fruit_rot': 3,
        ...     'powdery_mildew_fruit': 3,
        ... }
        >>> new_yaml = create_oversampled_dataset(
        ...     'data/raw/dataset.yaml',
        ...     oversample_config,
        ...     'data/processed/oversampled',
        ...     vary_augmentation=True
        ... )

        >>> # Dynamic oversampling (recommended)
        >>> new_yaml = create_oversampled_dataset(
        ...     'data/merged/dataset.yaml',
        ...     output_dir='data/processed/oversampled',
        ...     dynamic=True,
        ...     target_balance=0.65,
        ...     max_multiplier=5,
        ...     min_multiplier=2,
        ...     vary_augmentation=True
        ... )
    """
    # Calculate dynamic ratios if requested
    if dynamic and oversample_config is None:
        oversample_config = calculate_dynamic_oversample_ratios(
            dataset_yaml=dataset_yaml,
            target_balance=target_balance,
            max_multiplier=max_multiplier,
            min_multiplier=min_multiplier,
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

    return strategy.create_oversampled_dataset(splits=splits, vary_augmentation=vary_augmentation)
