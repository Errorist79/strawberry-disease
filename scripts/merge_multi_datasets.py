#!/usr/bin/env python3
"""
Multi-dataset merger for strawberry disease detection.

Merges multiple datasets into a single unified dataset:
1. Kaggle Dataset (2,500 images) - optional
2. Roboflow #1 (4,918 images)
3. Roboflow #2 (2,757 images)
4. healthy_merged (946 images) - PlantVillage + Kaggle tipburn healthy-only

Supports two strategies:
- granular: 10 classes (7 diseases + 3 healthy types)
- simple: 8 classes (7 diseases + healthy)

Note: Kaggle dataset has no healthy class, only disease classes.

Usage (for fine-tuning):
    python scripts/merge_multi_datasets.py \
        --roboflow1 data/external/roboflow_4918/data.yaml \
        --roboflow2 data/external/roboflow_2757/data.yaml \
        --plantvillage data/external/healthy_merged/data.yaml \
        --output data/processed/merged_for_finetune \
        --strategy granular
"""

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml


class MultiDatasetMerger:
    """Merge multiple YOLO format datasets with class harmonization."""

    def __init__(self, output_dir: Path, strategy: str = 'granular'):
        """
        Initialize merger.

        Args:
            output_dir: Output directory for merged dataset
            strategy: 'granular' (12 classes) or 'simple' (9 classes)
        """
        self.output_dir = Path(output_dir)
        self.strategy = strategy
        self.datasets = []
        self.class_mapping = self._create_class_mapping()
        self.stats = defaultdict(lambda: defaultdict(int))
        self.skipped_files = {
            'binary': [],
            'mac_resource_fork': [],
            'parse_error': []
        }

    def _create_class_mapping(self):
        """Create global class mapping based on strategy."""
        if self.strategy == 'granular':
            # 10 classes: 7 diseases + 3 healthy types (leaf_scorch removed)
            # Note: Kaggle dataset has no healthy class
            return {
                'angular_leafspot': 0,
                'anthracnose_fruit_rot': 1,
                'blossom_blight': 2,
                'gray_mold': 3,
                'leaf_spot': 4,
                'powdery_mildew_leaf': 5,
                'powdery_mildew_fruit': 6,
                'healthy_leaf': 7,
                'healthy_flower': 8,
                'healthy_fruit': 9,
            }
        else:  # simple
            # 8 classes: 7 diseases + healthy (leaf_scorch removed)
            # Note: Kaggle dataset has no healthy class
            return {
                'angular_leafspot': 0,
                'anthracnose_fruit_rot': 1,
                'blossom_blight': 2,
                'gray_mold': 3,
                'leaf_spot': 4,
                'powdery_mildew_leaf': 5,
                'powdery_mildew_fruit': 6,
                'healthy': 7,  # All healthy merged (from Roboflow #2 + healthy_merged)
            }

    def add_dataset(self, name: str, yaml_path: Path, class_map: dict, priority: int = 1):
        """
        Add a dataset to merge.

        Args:
            name: Dataset name (for prefixing files)
            yaml_path: Path to dataset.yaml
            class_map: Mapping from dataset classes to global classes
            priority: Priority for conflict resolution (1=highest)
        """
        if not yaml_path.exists():
            print(f"Warning: Dataset YAML not found: {yaml_path}")
            return

        self.datasets.append({
            'name': name,
            'yaml_path': yaml_path,
            'class_map': class_map,
            'priority': priority
        })

        print(f"Added dataset: {name} (priority={priority})")

    def merge(self) -> Path:
        """
        Merge all datasets.

        Returns:
            Path to merged data.yaml
        """
        if not self.datasets:
            print("❌ No datasets to merge!")
            sys.exit(1)

        print(f"\n{'='*70}")
        print(f"Merging {len(self.datasets)} Datasets")
        print(f"Strategy: {self.strategy} ({len(self.class_mapping)} classes)")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")

        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Process each dataset
        for idx, dataset in enumerate(self.datasets, 1):
            print(f"\n[{idx}/{len(self.datasets)}] Processing: {dataset['name']}")
            self._process_dataset(dataset)

        # Create merged data.yaml
        merged_yaml = self._create_merged_yaml()

        # Print statistics
        self._print_stats()

        print(f"\n{'='*70}")
        print("✅ MERGE COMPLETE!")
        print(f"{'='*70}")
        print(f"Output: {merged_yaml}")
        print(f"Total classes: {len(self.class_mapping)}")
        print(f"{'='*70}\n")

        return merged_yaml

    def _process_dataset(self, dataset: dict):
        """Process a single dataset and add to merged dataset."""
        # Load dataset config
        with open(dataset['yaml_path']) as f:
            config = yaml.safe_load(f)

        # Get dataset path - prefer 'path' field, fallback to yaml parent directory
        if 'path' in config:
            dataset_path = Path(config['path'])
            # If path is relative, make it relative to yaml_path
            if not dataset_path.is_absolute():
                dataset_path = dataset['yaml_path'].parent / dataset_path
        else:
            # Use yaml file's parent directory
            dataset_path = dataset['yaml_path'].parent

        class_names = config.get('names', [])

        if not class_names:
            print(f"  Warning: No class names found in {dataset['yaml_path']}")
            return

        print(f"  Source: {dataset_path}")
        print(f"  Classes: {len(class_names)}")

        # Process each split
        for split in ['train', 'val', 'valid', 'test']:
            # Handle both 'val' and 'valid'
            output_split = 'val' if split == 'valid' else split

            if split not in config:
                continue

            images_rel = config[split]

            # Handle relative paths that start with ../
            # Roboflow uses ../train/images but actual path is train/images
            if images_rel.startswith('../'):
                images_rel = images_rel[3:]  # Remove ../

            images_in = dataset_path / images_rel

            # Find labels directory
            labels_in = self._find_labels_dir(images_in, dataset_path, images_rel)

            if not images_in.exists():
                print(f"  ⚠️  Warning: {split} images not found")
                print(f"      Expected path: {images_in}")
                print(f"      Dataset path: {dataset_path}")
                print(f"      Images rel: {images_rel}")
                continue

            if not labels_in.exists():
                print(f"  ⚠️  Warning: {split} labels not found at {labels_in}")
                continue

            # Process images/labels
            count = self._process_split(
                dataset['name'],
                dataset['class_map'],
                class_names,
                images_in,
                labels_in,
                output_split
            )

            if count > 0:
                print(f"  ✅ {output_split}: {count} images")

    def _find_labels_dir(self, images_dir: Path, dataset_path: Path, images_rel: str) -> Path:
        """Find labels directory (handles various structures)."""
        # Try 1: Parallel structure (train/images → train/labels)
        labels_dir = Path(str(images_dir).replace('images', 'labels'))
        if labels_dir.exists():
            return labels_dir

        # Try 2: Parent/labels/split structure
        split_name = Path(images_rel).name
        labels_dir = dataset_path / 'labels' / split_name
        if labels_dir.exists():
            return labels_dir

        # Try 3: Same level as images
        labels_dir = images_dir.parent / 'labels'
        if labels_dir.exists():
            return labels_dir

        return labels_dir  # Return best guess

    def _is_binary_file(self, file_path: Path) -> bool:
        """
        Check if a file contains binary data.

        Args:
            file_path: Path to the file to check

        Returns:
            True if file appears to be binary, False otherwise
        """
        try:
            # Read first 512 bytes to check for binary content
            with open(file_path, 'rb') as f:
                chunk = f.read(512)

            # Check for null bytes (common in binary files)
            if b'\x00' in chunk:
                return True

            # Try to decode as UTF-8
            try:
                chunk.decode('utf-8')
                return False
            except UnicodeDecodeError:
                # Try latin-1 as fallback
                try:
                    chunk.decode('latin-1')
                    # If it decodes but has unusual characters, check content
                    text = chunk.decode('latin-1')
                    # YOLO labels should only have digits, spaces, dots, and newlines
                    allowed_chars = set('0123456789. \n\r\t-')
                    if any(c not in allowed_chars for c in text[:100]):
                        return True
                    return False
                except:
                    return True
        except Exception:
            return True

    def _process_split(
        self,
        dataset_name: str,
        class_map: dict,
        class_names: list,
        images_dir: Path,
        labels_dir: Path,
        output_split: str
    ) -> int:
        """Process a single dataset split."""
        count = 0
        prefix = dataset_name.replace(' ', '_').replace('-', '_').lower()

        # Process each label file
        for label_file in labels_dir.glob('*.txt'):
            # Skip Mac OS resource fork files
            if label_file.name.startswith('._'):
                self.skipped_files['mac_resource_fork'].append(str(label_file))
                continue

            # Check for binary files before processing
            if self._is_binary_file(label_file):
                self.skipped_files['binary'].append(str(label_file))
                print(f"  ⚠️  Warning: Skipping binary file: {label_file.name}")
                continue

            # Read labels with error handling for different encodings
            try:
                with open(label_file, encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                # Try with latin-1 encoding as fallback
                try:
                    with open(label_file, encoding='latin-1') as f:
                        lines = f.readlines()
                except Exception as e:
                    print(f"  ⚠️  Warning: Could not read {label_file.name}: {e}")
                    continue

            # Remap classes
            remapped_lines = []
            for line in lines:
                try:
                    parts = line.strip().split()

                    # Handle both bounding box and segmentation polygon formats
                    if len(parts) < 5:
                        # Invalid format, skip
                        continue
                    elif len(parts) == 5:
                        # Bounding box format: class x_center y_center width height
                        old_class_id = int(parts[0])
                        bbox_parts = parts[1:]
                    elif len(parts) > 5:
                        # Segmentation polygon format: class x1 y1 x2 y2 x3 y3 ...
                        # Convert polygon to bounding box by finding min/max x,y
                        old_class_id = int(parts[0])

                        # Extract polygon coordinates (pairs of x,y)
                        coords = [float(x) for x in parts[1:]]
                        x_coords = [coords[i] for i in range(0, len(coords), 2)]
                        y_coords = [coords[i] for i in range(1, len(coords), 2)]

                        # Calculate bounding box
                        x_min = min(x_coords)
                        x_max = max(x_coords)
                        y_min = min(y_coords)
                        y_max = max(y_coords)

                        # Convert to YOLO format (center x, center y, width, height)
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min

                        bbox_parts = [f"{x_center:.6f}", f"{y_center:.6f}", f"{width:.6f}", f"{height:.6f}"]
                    else:
                        continue

                    old_class_id = int(parts[0])
                except (ValueError, IndexError) as e:
                    # Skip lines with parsing errors (corrupted data, invalid format, etc.)
                    if label_file not in [f for f in self.skipped_files['parse_error']]:
                        self.skipped_files['parse_error'].append(str(label_file))
                        print(f"  ⚠️  Warning: Parse error in {label_file.name}: {e}")
                    continue

                # Check bounds
                if old_class_id >= len(class_names):
                    continue

                old_class_name = class_names[old_class_id]

                # Apply class mapping
                if old_class_name in class_map:
                    new_class_name = class_map[old_class_name]

                    if new_class_name in self.class_mapping:
                        new_class_id = self.class_mapping[new_class_name]
                        # Create bounding box line with remapped class
                        bbox_line = f"{new_class_id} {' '.join(bbox_parts)}"
                        remapped_lines.append(bbox_line + '\n')

                        # Update stats
                        self.stats[output_split][new_class_name] += 1

            # Skip if no valid labels
            if not remapped_lines:
                continue

            # Find corresponding image
            img_name = label_file.stem
            img_file = self._find_image_file(images_dir, img_name)

            if not img_file:
                continue

            # Create unique filename with dataset prefix
            new_img_name = f"{prefix}_{img_file.name}"
            new_label_name = f"{prefix}_{label_file.name}"

            # Copy files
            out_img = self.output_dir / output_split / 'images' / new_img_name
            out_label = self.output_dir / output_split / 'labels' / new_label_name

            shutil.copy(img_file, out_img)

            with open(out_label, 'w') as f:
                f.writelines(remapped_lines)

            count += 1

        return count

    def _find_image_file(self, images_dir: Path, stem: str) -> Path:
        """Find image file with various extensions."""
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            img_file = images_dir / f"{stem}{ext}"
            if img_file.exists():
                return img_file
        return None

    def _create_merged_yaml(self) -> Path:
        """Create merged dataset.yaml."""
        # Get class names in order
        class_list = [None] * len(self.class_mapping)
        for name, class_id in self.class_mapping.items():
            class_list[class_id] = name

        # Remove None (unused IDs)
        class_list = [n for n in class_list if n is not None]

        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_list),
            'names': class_list
        }

        output_yaml = self.output_dir / 'data.yaml'
        with open(output_yaml, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return output_yaml

    def _print_stats(self):
        """Print merge statistics."""
        print(f"\n{'='*70}")
        print("MERGED DATASET STATISTICS")
        print(f"{'='*70}")

        for split in ['train', 'val', 'test']:
            if split not in self.stats or not self.stats[split]:
                continue

            print(f"\n{split.upper()}:")
            total = sum(self.stats[split].values())
            print(f"  Total instances: {total}")

            # Sort by class ID
            class_items = []
            for class_name in self.stats[split].keys():
                class_id = self.class_mapping[class_name]
                count = self.stats[split][class_name]
                class_items.append((class_id, class_name, count))

            class_items.sort()

            print(f"  Class distribution:")
            for class_id, class_name, count in class_items:
                pct = 100 * count / total if total > 0 else 0
                print(f"    [{class_id}] {class_name}: {count} ({pct:.1f}%)")

        # Print skipped files summary
        total_skipped = sum(len(files) for files in self.skipped_files.values())
        if total_skipped > 0:
            print(f"\n{'='*70}")
            print("SKIPPED FILES SUMMARY")
            print(f"{'='*70}")

            if self.skipped_files['mac_resource_fork']:
                print(f"\n⚠️  Mac OS Resource Fork Files: {len(self.skipped_files['mac_resource_fork'])}")
                print("  These are Mac filesystem metadata files (._*) and were automatically skipped.")

            if self.skipped_files['binary']:
                print(f"\n⚠️  Binary/Corrupted Files: {len(self.skipped_files['binary'])}")
                print("  These files contain binary data instead of text and were skipped:")
                for f in self.skipped_files['binary'][:5]:  # Show first 5
                    print(f"    - {f}")
                if len(self.skipped_files['binary']) > 5:
                    print(f"    ... and {len(self.skipped_files['binary']) - 5} more")

            if self.skipped_files['parse_error']:
                print(f"\n⚠️  Parse Error Files: {len(self.skipped_files['parse_error'])}")
                print("  These files had invalid YOLO format and were skipped:")
                for f in self.skipped_files['parse_error'][:5]:  # Show first 5
                    print(f"    - {f}")
                if len(self.skipped_files['parse_error']) > 5:
                    print(f"    ... and {len(self.skipped_files['parse_error']) - 5} more")

            print(f"\n💡 Recommendation:")
            print(f"  Review and clean up these files in your source datasets.")
            print(f"  Total files skipped: {total_skipped}")


def create_default_class_mappings(strategy: str):
    """Create default class mappings for each dataset."""
    if strategy == 'granular':
        return {
            'kaggle': {
                # Kaggle class names use spaces
                'angular leafspot': 'angular_leafspot',
                'anthracnose fruit rot': 'anthracnose_fruit_rot',
                'blossom blight': 'blossom_blight',
                'gray mold': 'gray_mold',
                'leaf spot': 'leaf_spot',
                'powdery mildew fruit': 'powdery_mildew_fruit',
                'powdery mildew leaf': 'powdery_mildew_leaf',
                # Note: Kaggle dataset has no healthy class
            },
            'roboflow1': {
                'Angular Leafspot': 'angular_leafspot',
                'Anthracnose Fruit Rot': 'anthracnose_fruit_rot',
                'Blossom Blight': 'blossom_blight',
                'Gray Mold': 'gray_mold',
                'Leaf Spot': 'leaf_spot',
                'Powdery Mildew Leaf': 'powdery_mildew_leaf',
                'Powdery Mildew Fruit': 'powdery_mildew_fruit',
            },
            'roboflow2': {
                'Angular Leafspot': 'angular_leafspot',
                'Anthracnose Fruit Rot': 'anthracnose_fruit_rot',
                'Blossom Blight': 'blossom_blight',
                'Gray Mold': 'gray_mold',
                'Leaf Spot': 'leaf_spot',
                'Powdery Mildew Leaf': 'powdery_mildew_leaf',
                'Powdery Mildew Fruit': 'powdery_mildew_fruit',
                'Healthy Flower': 'healthy_flower',
                'Healthy Fruit': 'healthy_fruit',
                'Healthy Leaf': 'healthy_leaf',
            },
            'plantvillage': {
                'healthy_leaf': 'healthy_leaf',
                # Note: Leaf_scorch removed - not used in fine-tuning
            },
        }
    else:  # simple
        return {
            'kaggle': {
                # Kaggle class names use spaces
                'angular leafspot': 'angular_leafspot',
                'anthracnose fruit rot': 'anthracnose_fruit_rot',
                'blossom blight': 'blossom_blight',
                'gray mold': 'gray_mold',
                'leaf spot': 'leaf_spot',
                'powdery mildew fruit': 'powdery_mildew_fruit',
                'powdery mildew leaf': 'powdery_mildew_leaf',
                # Note: Kaggle dataset has no healthy class
            },
            'roboflow1': {
                'Angular Leafspot': 'angular_leafspot',
                'Anthracnose Fruit Rot': 'anthracnose_fruit_rot',
                'Blossom Blight': 'blossom_blight',
                'Gray Mold': 'gray_mold',
                'Leaf Spot': 'leaf_spot',
                'Powdery Mildew Leaf': 'powdery_mildew_leaf',
                'Powdery Mildew Fruit': 'powdery_mildew_fruit',
            },
            'roboflow2': {
                'Angular Leafspot': 'angular_leafspot',
                'Anthracnose Fruit Rot': 'anthracnose_fruit_rot',
                'Blossom Blight': 'blossom_blight',
                'Gray Mold': 'gray_mold',
                'Leaf Spot': 'leaf_spot',
                'Powdery Mildew Leaf': 'powdery_mildew_leaf',
                'Powdery Mildew Fruit': 'powdery_mildew_fruit',
                'Healthy Flower': 'healthy',
                'Healthy Fruit': 'healthy',
                'Healthy Leaf': 'healthy',
            },
            'plantvillage': {
                'healthy_leaf': 'healthy',
                # Note: Leaf_scorch removed - not used in fine-tuning
            },
        }


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple strawberry disease datasets"
    )
    parser.add_argument(
        '--kaggle',
        type=Path,
        help='Path to Kaggle dataset.yaml'
    )
    parser.add_argument(
        '--roboflow1',
        type=Path,
        help='Path to Roboflow #1 (4918) dataset.yaml'
    )
    parser.add_argument(
        '--roboflow2',
        type=Path,
        help='Path to Roboflow #2 (2757) dataset.yaml'
    )
    parser.add_argument(
        '--plantvillage',
        type=Path,
        help='Path to PlantVillage strawberry dataset.yaml'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for merged dataset'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='granular',
        choices=['granular', 'simple'],
        help='Class mapping strategy (granular=12 classes, simple=9 classes)'
    )

    args = parser.parse_args()

    # Check at least one dataset provided
    if not any([args.kaggle, args.roboflow1, args.roboflow2, args.plantvillage]):
        print("❌ Error: At least one dataset must be provided!")
        parser.print_help()
        sys.exit(1)

    # Create merger
    merger = MultiDatasetMerger(
        output_dir=args.output,
        strategy=args.strategy
    )

    # Get class mappings
    mappings = create_default_class_mappings(args.strategy)

    # Add datasets (in priority order)
    if args.kaggle:
        merger.add_dataset('kaggle', args.kaggle, mappings['kaggle'], priority=1)

    if args.roboflow1:
        merger.add_dataset('roboflow1', args.roboflow1, mappings['roboflow1'], priority=2)

    if args.roboflow2:
        merger.add_dataset('roboflow2', args.roboflow2, mappings['roboflow2'], priority=3)

    if args.plantvillage:
        merger.add_dataset('plantvillage', args.plantvillage, mappings['plantvillage'], priority=4)

    # Merge
    merged_yaml = merger.merge()

    print(f"\n🎉 SUCCESS!")
    print(f"Merged dataset ready at: {args.output}")
    print(f"Use in training with: --data-yaml {merged_yaml}")


if __name__ == '__main__':
    main()
