#!/usr/bin/env python3
"""
Validate CVAT export for PlantVillage re-annotation

This script validates that:
1. All bounding boxes are within image boundaries
2. YOLO format is correct
3. No annotations were lost
4. Quality improvements over original dataset
"""

import argparse
from pathlib import Path
from collections import defaultdict, Counter
import sys


def validate_bbox(bbox_line, image_name):
    """
    Validate YOLO format bbox
    Returns (is_valid, error_message, class_id, bbox_coords)
    """
    try:
        parts = bbox_line.strip().split()
        if len(parts) != 5:
            return False, f"Invalid format: expected 5 values, got {len(parts)}", None, None

        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        # Strict validation for YOLO format
        if not (0 <= x_center <= 1):
            return False, f"x_center out of range [0,1]: {x_center}", class_id, (x_center, y_center, width, height)
        if not (0 <= y_center <= 1):
            return False, f"y_center out of range [0,1]: {y_center}", class_id, (x_center, y_center, width, height)
        if not (0 < width <= 1):
            return False, f"width out of range (0,1]: {width}", class_id, (x_center, y_center, width, height)
        if not (0 < height <= 1):
            return False, f"height out of range (0,1]: {height}", class_id, (x_center, y_center, width, height)

        # Check if bbox extends outside image boundaries
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2

        EPSILON = 0.0001  # Allow tiny floating point errors

        if x_min < -EPSILON or x_max > 1 + EPSILON:
            return False, f"bbox extends outside horizontally: x_min={x_min:.6f}, x_max={x_max:.6f}", class_id, (x_center, y_center, width, height)
        if y_min < -EPSILON or y_max > 1 + EPSILON:
            return False, f"bbox extends outside vertically: y_min={y_min:.6f}, y_max={y_max:.6f}", class_id, (x_center, y_center, width, height)

        # Check if very close to boundary (warn but don't fail)
        boundary_warn = []
        if x_min < 0.001:
            boundary_warn.append("x_min close to 0")
        if x_max > 0.999:
            boundary_warn.append("x_max close to 1")
        if y_min < 0.001:
            boundary_warn.append("y_min close to 0")
        if y_max > 0.999:
            boundary_warn.append("y_max close to 1")

        warning = ", ".join(boundary_warn) if boundary_warn else None

        return True, warning, class_id, (x_center, y_center, width, height)

    except Exception as e:
        return False, f"Parse error: {str(e)}", None, None


def validate_dataset(dataset_path, split_name, class_names):
    """Validate a single split of the dataset"""

    images_dir = dataset_path / split_name / "images"
    labels_dir = dataset_path / split_name / "labels"

    if not images_dir.exists():
        return None, f"Images directory not found: {images_dir}"

    if not labels_dir.exists():
        return None, f"Labels directory not found: {labels_dir}"

    results = {
        'total_images': 0,
        'total_labels': 0,
        'total_instances': 0,
        'valid_instances': 0,
        'invalid_instances': 0,
        'class_distribution': Counter(),
        'errors': [],
        'warnings': [],
        'images_without_labels': [],
        'labels_without_images': [],
    }

    # Get all images and labels
    image_files = set()
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']:
        image_files.update([f.stem for f in images_dir.glob(ext)])

    label_files = set([f.stem for f in labels_dir.glob("*.txt")])

    results['total_images'] = len(image_files)
    results['total_labels'] = len(label_files)

    # Find mismatches
    results['images_without_labels'] = sorted(list(image_files - label_files))
    results['labels_without_images'] = sorted(list(label_files - image_files))

    # Validate each label file
    for label_file in sorted(labels_dir.glob("*.txt")):
        image_name = label_file.stem

        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            if not line.strip():
                continue

            results['total_instances'] += 1

            is_valid, message, class_id, bbox = validate_bbox(line, image_name)

            if class_id is not None:
                results['class_distribution'][class_id] += 1

            if is_valid:
                results['valid_instances'] += 1
                if message:  # Boundary warning
                    results['warnings'].append({
                        'image': image_name,
                        'line': line_num,
                        'message': message,
                        'bbox': bbox
                    })
            else:
                results['invalid_instances'] += 1
                results['errors'].append({
                    'image': image_name,
                    'label_file': str(label_file),
                    'line': line_num,
                    'content': line.strip(),
                    'error': message,
                    'class': class_id,
                    'bbox': bbox
                })

    return results, None


def compare_datasets(original_stats, new_stats, split_name):
    """Compare original vs new annotations"""

    print(f"\n{'─'*70}")
    print(f"COMPARISON: {split_name.upper()} Split")
    print(f"{'─'*70}")

    # Instance counts
    orig_total = original_stats['total_instances']
    new_total = new_stats['total_instances']
    diff = new_total - orig_total

    print(f"\n📊 Total Instances:")
    print(f"   Original: {orig_total}")
    print(f"   New:      {new_total}")
    print(f"   Change:   {diff:+d} ({diff/orig_total*100:+.1f}%)" if orig_total > 0 else "   Change:   N/A")

    # Healthy instances
    orig_healthy = original_stats['class_distribution'].get(1, 0)
    new_healthy = new_stats['class_distribution'].get(1, 0)
    healthy_diff = new_healthy - orig_healthy

    print(f"\n🍓 Healthy Instances (Class 1):")
    print(f"   Original: {orig_healthy}")
    print(f"   New:      {new_healthy}")
    print(f"   Change:   {healthy_diff:+d} ({healthy_diff/orig_healthy*100:+.1f}%)" if orig_healthy > 0 else "   Change:   N/A")

    # Invalid bboxes
    orig_invalid = original_stats['invalid_instances']
    new_invalid = new_stats['invalid_instances']

    print(f"\n❌ Invalid BBoxes:")
    print(f"   Original: {orig_invalid} ({orig_invalid/orig_total*100:.2f}%)" if orig_total > 0 else "   Original: 0")
    print(f"   New:      {new_invalid} ({new_invalid/new_total*100:.2f}%)" if new_total > 0 else "   New:      0")

    if new_invalid == 0 and orig_invalid > 0:
        print(f"   ✅ SUCCESS: All {orig_invalid} invalid bboxes fixed!")
    elif new_invalid < orig_invalid:
        print(f"   ✅ IMPROVED: Reduced by {orig_invalid - new_invalid} invalid bboxes")
    elif new_invalid > orig_invalid:
        print(f"   ⚠️  WARNING: Increased by {new_invalid - orig_invalid} invalid bboxes!")
    else:
        print(f"   ⚠️  NO CHANGE: Still have {new_invalid} invalid bboxes")


def main():
    parser = argparse.ArgumentParser(description='Validate CVAT exported annotations')
    parser.add_argument('--input', required=True, help='Path to re-annotated dataset')
    parser.add_argument('--original', required=True, help='Path to original dataset for comparison')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'], help='Splits to validate')

    args = parser.parse_args()

    input_path = Path(args.input)
    original_path = Path(args.original)

    print("=" * 70)
    print("CVAT Export Validation")
    print("=" * 70)
    print(f"\nInput:    {input_path}")
    print(f"Original: {original_path}")
    print(f"Splits:   {', '.join(args.splits)}")

    # Class names (from PlantVillage)
    class_names = ['Strawberry___Leaf_scorch', 'Strawberry___healthy']

    all_valid = True
    validation_results = {}
    original_results = {}

    # Validate each split
    for split in args.splits:
        print(f"\n{'='*70}")
        print(f"Validating {split.upper()} split...")
        print(f"{'='*70}")

        # Validate new annotations
        new_stats, error = validate_dataset(input_path, split, class_names)

        if error:
            print(f"❌ ERROR: {error}")
            all_valid = False
            continue

        validation_results[split] = new_stats

        print(f"\n📊 Dataset Statistics:")
        print(f"   Images: {new_stats['total_images']}")
        print(f"   Labels: {new_stats['total_labels']}")
        print(f"   Instances: {new_stats['total_instances']}")
        print(f"\n📈 Class Distribution:")
        for class_id in sorted(new_stats['class_distribution'].keys()):
            count = new_stats['class_distribution'][class_id]
            percentage = (count / new_stats['total_instances'] * 100) if new_stats['total_instances'] > 0 else 0
            label = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
            marker = " ← HEALTHY" if class_id == 1 else ""
            print(f"   [{class_id}] {label}: {count} ({percentage:.1f}%){marker}")

        print(f"\n✅ Valid instances:   {new_stats['valid_instances']}")
        print(f"❌ Invalid instances: {new_stats['invalid_instances']}")

        if new_stats['invalid_instances'] > 0:
            print(f"\n⚠️  VALIDATION FAILED: Found {new_stats['invalid_instances']} invalid bboxes!")
            all_valid = False

            # Show first 10 errors
            print(f"\nFirst 10 errors:")
            for i, error in enumerate(new_stats['errors'][:10], 1):
                print(f"\n{i}. Image: {error['image']}")
                print(f"   Line: {error['line']}")
                print(f"   Content: {error['content']}")
                print(f"   Error: {error['error']}")

            if len(new_stats['errors']) > 10:
                print(f"\n... and {len(new_stats['errors']) - 10} more errors")

        if new_stats['warnings']:
            print(f"\n⚠️  {len(new_stats['warnings'])} bbox(es) very close to image boundaries")
            print(f"   (Not errors, but consider reviewing)")

        if new_stats['images_without_labels']:
            print(f"\n⚠️  {len(new_stats['images_without_labels'])} image(s) without labels")
            if len(new_stats['images_without_labels']) <= 5:
                for img in new_stats['images_without_labels']:
                    print(f"      {img}")

        if new_stats['labels_without_images']:
            print(f"\n⚠️  {len(new_stats['labels_without_images'])} label(s) without images")

        # Load original for comparison
        original_stats, error = validate_dataset(original_path, split, class_names)

        if not error:
            original_results[split] = original_stats
            compare_datasets(original_stats, new_stats, split)

    # Final summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")

    total_instances = sum(r['total_instances'] for r in validation_results.values())
    total_valid = sum(r['valid_instances'] for r in validation_results.values())
    total_invalid = sum(r['invalid_instances'] for r in validation_results.values())
    total_healthy = sum(r['class_distribution'].get(1, 0) for r in validation_results.values())

    print(f"\n📊 Overall Statistics:")
    print(f"   Total instances: {total_instances}")
    print(f"   Valid: {total_valid}")
    print(f"   Invalid: {total_invalid}")
    print(f"   Healthy: {total_healthy}")

    if original_results:
        orig_total = sum(r['total_instances'] for r in original_results.values())
        orig_invalid = sum(r['invalid_instances'] for r in original_results.values())
        orig_healthy = sum(r['class_distribution'].get(1, 0) for r in original_results.values())

        print(f"\n📈 Improvement:")
        print(f"   Invalid bboxes: {orig_invalid} → {total_invalid} ({orig_invalid - total_invalid} fixed)")
        print(f"   Healthy instances: {orig_healthy} → {total_healthy} ({total_healthy - orig_healthy:+d})")

    print()

    if all_valid and total_invalid == 0:
        print("✅ VALIDATION PASSED!")
        print("   All bounding boxes are valid and within image boundaries.")
        print("   Dataset is ready for training!")
        return 0
    else:
        print("❌ VALIDATION FAILED!")
        print(f"   Found {total_invalid} invalid bounding boxes.")
        print("   Please fix errors in CVAT and re-export.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
