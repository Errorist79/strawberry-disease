#!/usr/bin/env python3
"""
Analyze PlantVillage dataset to identify:
- Invalid bounding boxes
- Class distribution
- Healthy sample statistics
- Image quality issues
"""

import os
from pathlib import Path
from collections import defaultdict, Counter
import yaml

def validate_bbox(bbox_line, image_path):
    """
    Validate YOLO format bbox: class x_center y_center width height
    Returns (is_valid, error_message, class_id, bbox)
    """
    try:
        parts = bbox_line.strip().split()
        if len(parts) != 5:
            return False, f"Invalid format: expected 5 values, got {len(parts)}", None, None

        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        # Check ranges
        if not (0 <= x_center <= 1):
            return False, f"x_center out of range: {x_center}", class_id, (x_center, y_center, width, height)
        if not (0 <= y_center <= 1):
            return False, f"y_center out of range: {y_center}", class_id, (x_center, y_center, width, height)
        if not (0 < width <= 1):
            return False, f"width out of range: {width}", class_id, (x_center, y_center, width, height)
        if not (0 < height <= 1):
            return False, f"height out of range: {height}", class_id, (x_center, y_center, width, height)

        # Check if bbox goes outside image
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2

        if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1:
            return False, f"bbox extends outside image: ({x_min:.3f}, {y_min:.3f}, {x_max:.3f}, {y_max:.3f})", class_id, (x_center, y_center, width, height)

        return True, None, class_id, (x_center, y_center, width, height)

    except Exception as e:
        return False, f"Parse error: {str(e)}", None, None


def analyze_split(split_path, split_name, class_names):
    """Analyze a single split (train/valid/test)"""

    images_dir = split_path / "images"
    labels_dir = split_path / "labels"

    stats = {
        'total_images': 0,
        'total_instances': 0,
        'class_distribution': Counter(),
        'invalid_bboxes': [],
        'valid_bboxes': [],
        'images_without_labels': [],
        'labels_without_images': [],
        'multi_instance_images': [],
    }

    # Get all image and label files
    image_files = set([f.stem for f in images_dir.glob("*.jpg")])
    image_files.update([f.stem for f in images_dir.glob("*.png")])
    label_files = set([f.stem for f in labels_dir.glob("*.txt")])

    stats['total_images'] = len(image_files)

    # Find mismatches
    stats['images_without_labels'] = list(image_files - label_files)
    stats['labels_without_images'] = list(label_files - image_files)

    # Analyze each label file
    for label_file in labels_dir.glob("*.txt"):
        image_name = label_file.stem

        # Find corresponding image
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            potential_path = images_dir / f"{image_name}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break

        if not image_path:
            continue

        with open(label_file, 'r') as f:
            lines = f.readlines()

        if len(lines) > 1:
            stats['multi_instance_images'].append(image_name)

        for line_num, line in enumerate(lines, 1):
            if not line.strip():
                continue

            is_valid, error_msg, class_id, bbox = validate_bbox(line, image_path)

            stats['total_instances'] += 1

            if class_id is not None:
                stats['class_distribution'][class_id] += 1

            if is_valid:
                stats['valid_bboxes'].append({
                    'image': image_name,
                    'class': class_id,
                    'bbox': bbox
                })
            else:
                stats['invalid_bboxes'].append({
                    'image': image_name,
                    'label_file': str(label_file),
                    'line': line_num,
                    'content': line.strip(),
                    'error': error_msg,
                    'class': class_id,
                    'bbox': bbox
                })

    return stats


def main():
    # Dataset path
    dataset_root = Path("/Users/errorist/Documents/personal-projects/strawberry-disease/data/external/plantvillage_strawberry")

    # Load class names
    with open(dataset_root / "data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)

    class_names = data_config['names']
    print(f"\n{'='*70}")
    print(f"PlantVillage Dataset Analysis")
    print(f"{'='*70}\n")
    print(f"Classes: {class_names}")
    print(f"  [0] {class_names[0]}")
    print(f"  [1] {class_names[1]} ← HEALTHY CLASS")
    print()

    # Analyze each split
    splits = ['train', 'valid', 'test']
    all_stats = {}

    for split in splits:
        split_path = dataset_root / split
        if not split_path.exists():
            print(f"⚠️  {split.upper()} split not found: {split_path}")
            continue

        print(f"\n{'─'*70}")
        print(f"Analyzing {split.upper()} split...")
        print(f"{'─'*70}")

        stats = analyze_split(split_path, split, class_names)
        all_stats[split] = stats

        print(f"📊 Total images: {stats['total_images']}")
        print(f"📊 Total instances: {stats['total_instances']}")
        print(f"\n📈 Class Distribution:")
        for class_id in sorted(stats['class_distribution'].keys()):
            count = stats['class_distribution'][class_id]
            percentage = (count / stats['total_instances'] * 100) if stats['total_instances'] > 0 else 0
            class_label = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
            marker = " ← HEALTHY" if class_id == 1 else ""
            print(f"  [{class_id}] {class_label}: {count} ({percentage:.1f}%){marker}")

        print(f"\n✅ Valid bboxes: {len(stats['valid_bboxes'])}")
        print(f"❌ Invalid bboxes: {len(stats['invalid_bboxes'])}")

        if stats['invalid_bboxes']:
            print(f"\n⚠️  INVALID BBOX DETAILS:")
            # Group by error type
            error_types = defaultdict(list)
            for invalid in stats['invalid_bboxes']:
                error_key = invalid['error'].split(':')[0]  # Get error category
                error_types[error_key].append(invalid)

            for error_type, errors in error_types.items():
                print(f"\n  {error_type}: {len(errors)} cases")
                # Show first 5 examples
                for i, error in enumerate(errors[:5], 1):
                    print(f"    {i}. {error['image']}")
                    print(f"       Line {error['line']}: {error['content']}")
                    print(f"       Error: {error['error']}")
                if len(errors) > 5:
                    print(f"    ... and {len(errors) - 5} more")

        if stats['images_without_labels']:
            print(f"\n⚠️  Images without labels: {len(stats['images_without_labels'])}")
            print(f"    Examples: {', '.join(stats['images_without_labels'][:5])}")

        if stats['labels_without_images']:
            print(f"\n⚠️  Labels without images: {len(stats['labels_without_images'])}")
            print(f"    Examples: {', '.join(stats['labels_without_images'][:5])}")

        if stats['multi_instance_images']:
            print(f"\n📦 Multi-instance images: {len(stats['multi_instance_images'])}")
            print(f"    (Images with multiple objects - this is OK)")

    # Summary report
    print(f"\n{'='*70}")
    print(f"SUMMARY REPORT")
    print(f"{'='*70}\n")

    total_images = sum(stats['total_images'] for stats in all_stats.values())
    total_instances = sum(stats['total_instances'] for stats in all_stats.values())
    total_invalid = sum(len(stats['invalid_bboxes']) for stats in all_stats.values())
    total_healthy = sum(stats['class_distribution'].get(1, 0) for stats in all_stats.values())
    total_leaf_scorch = sum(stats['class_distribution'].get(0, 0) for stats in all_stats.values())

    print(f"📁 Total dataset:")
    print(f"   Images: {total_images}")
    print(f"   Instances: {total_instances}")
    print(f"   Invalid bboxes: {total_invalid} ({total_invalid/total_instances*100:.2f}%)" if total_instances > 0 else "   Invalid bboxes: 0")

    print(f"\n🍓 Healthy samples (class 1):")
    print(f"   Total instances: {total_healthy}")
    print(f"   Train: {all_stats.get('train', {}).get('class_distribution', {}).get(1, 0)}")
    print(f"   Valid: {all_stats.get('valid', {}).get('class_distribution', {}).get(1, 0)}")
    print(f"   Test: {all_stats.get('test', {}).get('class_distribution', {}).get(1, 0)}")

    print(f"\n🍂 Leaf scorch samples (class 0):")
    print(f"   Total instances: {total_leaf_scorch}")
    print(f"   Train: {all_stats.get('train', {}).get('class_distribution', {}).get(0, 0)}")
    print(f"   Valid: {all_stats.get('valid', {}).get('class_distribution', {}).get(0, 0)}")
    print(f"   Test: {all_stats.get('test', {}).get('class_distribution', {}).get(0, 0)}")

    print(f"\n{'='*70}")

    # Export detailed invalid bbox list for fixing
    if total_invalid > 0:
        output_file = Path("/Users/errorist/Documents/personal-projects/strawberry-disease/logs/plantvillage_invalid_bboxes.txt")
        with open(output_file, 'w') as f:
            f.write("PlantVillage Dataset - Invalid Bounding Boxes\n")
            f.write("=" * 70 + "\n\n")

            for split, stats in all_stats.items():
                if stats['invalid_bboxes']:
                    f.write(f"\n{split.upper()} Split - {len(stats['invalid_bboxes'])} invalid bboxes:\n")
                    f.write("-" * 70 + "\n")
                    for invalid in stats['invalid_bboxes']:
                        f.write(f"File: {invalid['label_file']}\n")
                        f.write(f"Image: {invalid['image']}\n")
                        f.write(f"Line: {invalid['line']}\n")
                        f.write(f"Content: {invalid['content']}\n")
                        f.write(f"Error: {invalid['error']}\n")
                        f.write("\n")

        print(f"\n💾 Detailed invalid bbox list saved to:")
        print(f"   {output_file}")

    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
