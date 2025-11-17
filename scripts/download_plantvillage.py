#!/usr/bin/env python3
"""
PlantVillage YOLO dataset downloader and strawberry filter.

Downloads the PlantVillage dataset in YOLO format from Kaggle and filters
only strawberry classes (Strawberry___healthy, Strawberry___Leaf_scorch).

Dataset: https://www.kaggle.com/datasets/sebastianpalaciob/plantvillage-for-object-detection-yolo

Usage:
    python scripts/download_plantvillage.py --output data/external/plantvillage
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import yaml


def download_plantvillage_yolo(output_dir: Path) -> Path:
    """
    Download PlantVillage YOLO dataset from Kaggle.

    Args:
        output_dir: Directory to save downloaded dataset

    Returns:
        Path to downloaded dataset directory

    Raises:
        ImportError: If kaggle package not installed
        Exception: If download fails
    """
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle package not found!")
        print("Install with: pip install kaggle")
        print("\nThen configure:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Create API token (downloads kaggle.json)")
        print("3. Place in ~/.kaggle/kaggle.json (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\kaggle.json (Windows)")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Downloading PlantVillage YOLO Dataset from Kaggle")
    print(f"{'='*70}")
    print(f"Dataset: sebastianpalaciob/plantvillage-for-object-detection-yolo")
    print(f"Output: {output_dir}")
    print()

    try:
        # Download dataset
        kaggle.api.dataset_download_files(
            'sebastianpalaciob/plantvillage-for-object-detection-yolo',
            path=output_dir,
            unzip=True
        )

        print(f"✅ Download complete: {output_dir}")
        return output_dir

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check Kaggle API credentials (~/.kaggle/kaggle.json)")
        print("2. Verify dataset exists and you have access")
        print("3. Check internet connection")
        sys.exit(1)


def filter_strawberry_classes(plantvillage_dir: Path, output_dir: Path) -> Path:
    """
    Filter only strawberry classes from PlantVillage dataset.

    PlantVillage has 38 plant-disease combinations. We only need:
    - Strawberry___healthy
    - Strawberry___Leaf_scorch

    Args:
        plantvillage_dir: Input PlantVillage directory
        output_dir: Output directory for filtered dataset

    Returns:
        Path to filtered dataset.yaml
    """
    print(f"\n{'='*70}")
    print("Filtering Strawberry Classes")
    print(f"{'='*70}")

    # Navigate to actual dataset directory (handle nested structure)
    # Structure: PlantVillage_for_object_detection/Dataset/
    if (plantvillage_dir / 'PlantVillage_for_object_detection' / 'Dataset').exists():
        dataset_base = plantvillage_dir / 'PlantVillage_for_object_detection' / 'Dataset'
        print(f"Found nested structure, using: {dataset_base}")
    else:
        dataset_base = plantvillage_dir

    # Find classes.yaml or data.yaml
    yaml_path = None
    for yaml_name in ['classes.yaml', 'data.yaml']:
        candidate = dataset_base / yaml_name
        if candidate.exists():
            yaml_path = candidate
            print(f"Found config: {yaml_path}")
            break

    # Try searching recursively
    if not yaml_path:
        for yaml_name in ['classes.yaml', 'data.yaml']:
            for candidate in plantvillage_dir.rglob(yaml_name):
                yaml_path = candidate
                dataset_base = yaml_path.parent
                print(f"Found config at: {yaml_path}")
                break
            if yaml_path:
                break

    if not yaml_path:
        print(f"❌ classes.yaml or data.yaml not found in {plantvillage_dir}")
        print(f"Directory contents:")
        for item in plantvillage_dir.rglob('*'):
            if item.is_file() and item.suffix in ['.yaml', '.yml']:
                print(f"  {item}")
        sys.exit(1)

    # Load dataset config
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    print(f"Original dataset: {config['nc']} classes")

    # Find strawberry class IDs
    class_names = config['names']
    strawberry_classes = {}

    for class_id, name in enumerate(class_names):
        if name.startswith('Strawberry'):
            strawberry_classes[class_id] = name

    if not strawberry_classes:
        print(f"❌ No strawberry classes found in dataset!")
        print(f"Available classes: {class_names[:5]}...")
        sys.exit(1)

    print(f"\nFound {len(strawberry_classes)} strawberry classes:")
    for class_id, name in strawberry_classes.items():
        print(f"  {class_id}: {name}")

    # Create output directories
    output_dir = Path(output_dir)
    for split in ['train', 'valid', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Base path for dataset
    base_path = Path(config.get('path', dataset_base))

    # Process each split
    stats = {'train': 0, 'valid': 0, 'test': 0}

    for split in ['train', 'valid', 'test']:
        split_key = split
        if split not in config and split == 'valid':
            # Try 'val' as alternative
            split_key = 'val'

        if split_key not in config:
            continue

        # Get paths - handle both relative and absolute paths
        images_rel = config[split_key]
        if isinstance(images_rel, str):
            images_in = base_path / images_rel if not Path(images_rel).is_absolute() else Path(images_rel)
        else:
            images_in = base_path / images_rel

        # PlantVillage structure: Dataset/images/ and Dataset/labels/
        # If images path includes 'images', labels should be sibling directory
        if 'images' in str(images_in):
            labels_in = Path(str(images_in).replace('images', 'labels'))
        else:
            # Direct approach for PlantVillage: images/ and labels/ are siblings
            labels_in = dataset_base / 'labels' / split

        # Try multiple label locations
        if not labels_in.exists():
            labels_in = images_in.parent.parent / 'labels' / Path(images_rel).name

        if not labels_in.exists() and 'images' in images_rel:
            labels_rel = images_rel.replace('images', 'labels')
            labels_in = base_path / labels_rel

        if not images_in.exists():
            print(f"  Warning: {split} images not found at {images_in}")
            continue

        if not labels_in.exists():
            print(f"  Warning: {split} labels not found at {labels_in}")
            print(f"  Tried: {labels_in}")
            continue

        print(f"\nProcessing {split}...")
        print(f"  Images: {images_in}")
        print(f"  Labels: {labels_in}")

        # Process each label file
        for label_file in labels_in.glob('*.txt'):
            with open(label_file) as f:
                lines = f.readlines()

            # Check if contains strawberry classes
            has_strawberry = False
            filtered_lines = []

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                if class_id in strawberry_classes:
                    has_strawberry = True

                    # Remap class ID to new index (0, 1, 2...)
                    new_class_id = list(strawberry_classes.keys()).index(class_id)
                    parts[0] = str(new_class_id)
                    filtered_lines.append(' '.join(parts) + '\n')

            # Skip if no strawberry classes
            if not has_strawberry:
                continue

            # Find corresponding image
            img_name = label_file.stem
            img_file = None

            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                candidate = images_in / f"{img_name}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break

            if not img_file:
                print(f"  Warning: Image not found for {label_file.name}")
                continue

            # Copy files
            out_img = output_dir / split / 'images' / img_file.name
            out_label = output_dir / split / 'labels' / label_file.name

            shutil.copy(img_file, out_img)

            with open(out_label, 'w') as f:
                f.writelines(filtered_lines)

            stats[split] += 1

        print(f"  ✅ Filtered {stats[split]} images")

    # Create new data.yaml
    new_config = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(strawberry_classes),
        'names': list(strawberry_classes.values())
    }

    output_yaml = output_dir / 'data.yaml'
    with open(output_yaml, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)

    print(f"\n{'='*70}")
    print("✅ PlantVillage Strawberry Filtering Complete!")
    print(f"{'='*70}")
    print(f"Output: {output_yaml}")
    print(f"\nStatistics:")
    print(f"  Classes: {len(strawberry_classes)}")
    for name in strawberry_classes.values():
        print(f"    - {name}")
    print(f"\n  Images:")
    total = sum(stats.values())
    for split, count in stats.items():
        if count > 0:
            print(f"    {split}: {count}")
    print(f"    Total: {total}")
    print(f"{'='*70}\n")

    return output_yaml


def main():
    parser = argparse.ArgumentParser(
        description="Download and filter PlantVillage YOLO dataset for strawberry classes"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/external/plantvillage_strawberry'),
        help='Output directory for filtered dataset'
    )
    parser.add_argument(
        '--raw-output',
        type=Path,
        default=Path('data/external/plantvillage_raw'),
        help='Directory to save raw downloaded dataset'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download, only filter existing dataset'
    )

    args = parser.parse_args()

    # Download if not skipped
    if not args.skip_download:
        raw_dir = download_plantvillage_yolo(args.raw_output)
    else:
        raw_dir = args.raw_output
        if not raw_dir.exists():
            print(f"❌ Raw dataset not found: {raw_dir}")
            print("Remove --skip-download to download first")
            sys.exit(1)

    # Filter strawberry classes
    output_yaml = filter_strawberry_classes(raw_dir, args.output)

    print(f"\n🎉 SUCCESS!")
    print(f"PlantVillage strawberry dataset ready at: {args.output}")
    print(f"Use in training with: --data-yaml {output_yaml}")


if __name__ == '__main__':
    main()
