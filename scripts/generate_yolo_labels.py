#!/usr/bin/env python3
"""
Generate YOLO format labels for Kaggle tipburn dataset.
All images are classified as 'healthy_leaf' with placeholder full-image bboxes.
"""

import os
from pathlib import Path

def generate_labels(images_dir, labels_dir, class_id=1):
    """
    Generate YOLO label files for all images in the directory.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory where label files will be created
        class_id: Class ID for healthy_leaf (default: 1)
    """
    # Create labels directory if it doesn't exist
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    images_dir = Path(images_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))

    print(f"Found {len(image_files)} images in {images_dir}")

    # Placeholder bbox: center at (0.5, 0.5), size 95% of image
    # Format: class_id x_center y_center width height
    placeholder_bbox = f"{class_id} 0.5 0.5 0.95 0.95"

    # Generate label file for each image
    created_count = 0
    for image_path in image_files:
        # Create corresponding label file
        label_filename = image_path.stem + '.txt'
        label_path = labels_dir / label_filename

        # Write placeholder bbox
        with open(label_path, 'w') as f:
            f.write(placeholder_bbox + '\n')

        created_count += 1
        if created_count % 100 == 0:
            print(f"  Created {created_count} label files...")

    print(f"\n✓ Successfully created {created_count} label files")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"  Class ID: {class_id} (healthy_leaf)")
    print(f"  Bbox: center (0.5, 0.5), size (0.95, 0.95)")

if __name__ == '__main__':
    # Paths
    base_dir = Path('/Users/errorist/Documents/personal-projects/strawberry-disease/data/external/kaggle_tipburn')
    images_dir = base_dir / 'healthy'
    labels_dir = base_dir / 'healthy_labels'

    # Generate labels
    generate_labels(images_dir, labels_dir, class_id=1)

    print("\n" + "="*60)
    print("Next steps:")
    print("1. Check labels directory to verify creation")
    print("2. Create data.yaml for the dataset")
    print("3. Later: Refine bboxes in Label Studio if needed")
    print("="*60)
