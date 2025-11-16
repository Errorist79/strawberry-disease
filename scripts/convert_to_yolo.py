#!/usr/bin/env python3
"""
Convert strawberry disease dataset from JSON format to YOLO format.

This script converts COCO-style JSON annotations to YOLO format.
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict

import yaml


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height] (normalized).

    Args:
        bbox: [x, y, width, height] in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        [x_center, y_center, width, height] normalized to 0-1
    """
    x, y, w, h = bbox

    # Convert to center coordinates
    x_center = x + w / 2
    y_center = y + h / 2

    # Normalize
    x_center /= img_width
    y_center /= img_height
    w /= img_width
    h /= img_height

    return [x_center, y_center, w, h]


def extract_class_from_filename(filename):
    """Extract disease class from filename."""
    # Filename format: disease_name123.jpg
    parts = filename.split('.')
    base = parts[0]

    # Remove trailing numbers
    class_name = ''.join([c for c in base if not c.isdigit()])

    return class_name


def process_json_annotation(json_path):
    """
    Process a JSON annotation file.

    Returns:
        List of YOLO format annotations: [[class_id, x_center, y_center, width, height], ...]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Get image dimensions
    img_width = data.get('imageWidth', 640)
    img_height = data.get('imageHeight', 640)

    annotations = []

    # Process shapes/objects
    for shape in data.get('shapes', []):
        label = shape.get('label', '').lower()
        points = shape.get('points', [])

        if not points or len(points) < 2:
            continue

        # Convert polygon to bounding box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

        width = x_max - x_min
        height = y_max - y_min

        # Convert to YOLO format
        bbox = [x_min, y_min, width, height]
        yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)

        annotations.append({
            'label': label,
            'bbox': yolo_bbox
        })

    return annotations


def convert_dataset(source_dir, output_dir, split_name):
    """
    Convert dataset split to YOLO format.

    Args:
        source_dir: Source directory with images and JSON files
        output_dir: Output directory for YOLO format
        split_name: Name of split (train, val, test)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Create output directories
    images_dir = output_path / split_name / 'images'
    labels_dir = output_path / split_name / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))

    print(f"Processing {split_name}: {len(image_files)} images")

    # Collect class names
    class_names = set()

    for img_file in image_files:
        # Check for corresponding JSON
        json_file = img_file.with_suffix('.json')

        if not json_file.exists():
            # No annotation, skip or treat as healthy
            print(f"Warning: No annotation for {img_file.name}, skipping")
            continue

        # Process annotation
        annotations = process_json_annotation(json_file)

        if not annotations:
            # No objects, skip
            continue

        # Copy image
        shutil.copy(img_file, images_dir / img_file.name)

        # Collect class names
        for ann in annotations:
            class_names.add(ann['label'])

        # We'll write labels after we have class mapping
        # Store temporarily
        label_file = labels_dir / img_file.with_suffix('.txt').name
        label_file.write_text('')  # Placeholder

    return class_names, image_files, images_dir, labels_dir


def create_yolo_labels(image_files, source_dir, images_dir, labels_dir, class_to_id):
    """Create YOLO format label files."""
    source_path = Path(source_dir)

    count = 0
    for img_file in image_files:
        json_file = source_path / img_file.with_suffix('.json').name

        if not json_file.exists():
            continue

        annotations = process_json_annotation(json_file)

        if not annotations:
            continue

        label_file = labels_dir / img_file.with_suffix('.txt').name

        with open(label_file, 'w') as f:
            for ann in annotations:
                label = ann['label']
                if label not in class_to_id:
                    continue

                class_id = class_to_id[label]
                bbox = ann['bbox']

                # Write YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        count += 1

    print(f"  Created {count} label files")


def main():
    # Paths
    raw_dir = Path('data/raw')
    output_dir = Path('data/processed/yolo_dataset')

    # Process all splits
    all_classes = set()
    split_data = {}

    for split in ['train', 'val', 'test']:
        split_dir = raw_dir / split

        if not split_dir.exists():
            print(f"Warning: {split} directory not found, skipping")
            continue

        class_names, image_files, images_dir, labels_dir = convert_dataset(
            split_dir, output_dir, split
        )

        all_classes.update(class_names)
        split_data[split] = {
            'image_files': image_files,
            'source_dir': split_dir,
            'images_dir': images_dir,
            'labels_dir': labels_dir
        }

    # Create class mapping
    class_list = sorted(list(all_classes))
    class_to_id = {name: idx for idx, name in enumerate(class_list)}

    print(f"\nFound {len(class_list)} classes:")
    for name, idx in class_to_id.items():
        print(f"  {idx}: {name}")

    # Create YOLO labels with proper class IDs
    print("\nCreating YOLO format labels...")
    for split, data in split_data.items():
        print(f"Processing {split}...")
        create_yolo_labels(
            data['image_files'],
            data['source_dir'],
            data['images_dir'],
            data['labels_dir'],
            class_to_id
        )

    # Create data.yaml
    yaml_path = output_dir / 'data.yaml'

    yaml_data = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_list),
        'names': class_list
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"\nDataset conversion complete!")
    print(f"Output directory: {output_dir}")
    print(f"Config file: {yaml_path}")

    # Print summary
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split / 'images'
        if split_dir.exists():
            count = len(list(split_dir.glob('*.jpg'))) + len(list(split_dir.glob('*.png')))
            print(f"  {split}: {count} images")


if __name__ == '__main__':
    main()
