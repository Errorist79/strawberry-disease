#!/usr/bin/env python3
"""
Convert Grounded-SAM-2 JSON results to YOLO format.

Input:  JSON files with bbox in xyxy format (absolute coordinates)
Output: YOLO .txt files with format: class x_center y_center width height (normalized 0-1)

Usage:
    python convert_json_to_yolo.py --json_dir /path/to/json --output_dir /path/to/output

This script will:
1. Read all JSON result files
2. Convert xyxy bboxes to normalized xcywh format
3. Assign class ID (1 = healthy_leaf)
4. Generate YOLO .txt label files
5. Create validation report
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Class mapping
CLASS_MAP = {
    "strawberry leaf": 1,  # healthy_leaf class ID
    "leaf": 1,
    "healthy leaf": 1
}
DEFAULT_CLASS_ID = 1  # All Kaggle images are healthy leaves

def xyxy_to_yolo(bbox, img_width, img_height):
    """
    Convert bbox from xyxy (absolute) to yolo (normalized xcywh) format.

    Args:
        bbox: [x1, y1, x2, y2] in absolute pixel coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        tuple: (x_center, y_center, width, height) normalized to [0, 1]
    """
    x1, y1, x2, y2 = bbox

    # Calculate center and dimensions
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    # Normalize to [0, 1]
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    # Clamp to valid range [0, 1]
    x_center_norm = np.clip(x_center_norm, 0, 1)
    y_center_norm = np.clip(y_center_norm, 0, 1)
    width_norm = np.clip(width_norm, 0, 1)
    height_norm = np.clip(height_norm, 0, 1)

    return x_center_norm, y_center_norm, width_norm, height_norm

def validate_yolo_bbox(class_id, x_center, y_center, width, height):
    """
    Validate YOLO format bbox.

    Returns:
        tuple: (is_valid, error_message)
    """
    # Check all values are in [0, 1]
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
        return False, f"Center out of range: ({x_center:.4f}, {y_center:.4f})"

    if not (0 < width <= 1 and 0 < height <= 1):
        return False, f"Size out of range: ({width:.4f}, {height:.4f})"

    # Check bbox doesn't extend outside image
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2

    if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1:
        return False, f"Bbox extends outside image: ({x_min:.4f}, {y_min:.4f}, {x_max:.4f}, {y_max:.4f})"

    return True, None

def process_json_file(json_path, output_labels_dir):
    """
    Process a single JSON file and generate corresponding YOLO label file.

    Returns:
        dict: Statistics about the conversion
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_path = Path(data["image_path"])
    img_width, img_height = data["image_size"]
    detections = data["detections"]

    stats = {
        "image_name": image_path.stem,
        "detections": len(detections),
        "valid_boxes": 0,
        "invalid_boxes": 0,
        "errors": []
    }

    if len(detections) == 0:
        # No detections - create empty label file (or skip)
        label_path = output_labels_dir / f"{image_path.stem}.txt"
        label_path.touch()  # Create empty file
        return stats

    # Convert each detection to YOLO format
    yolo_lines = []
    for detection in detections:
        bbox_xyxy = detection["bbox"]  # [x1, y1, x2, y2]
        label = detection.get("label", "strawberry leaf")
        confidence = detection.get("confidence", 1.0)

        # Get class ID
        class_id = CLASS_MAP.get(label.lower(), DEFAULT_CLASS_ID)

        # Convert to YOLO format
        x_center, y_center, width, height = xyxy_to_yolo(
            bbox_xyxy, img_width, img_height
        )

        # Validate
        is_valid, error_msg = validate_yolo_bbox(
            class_id, x_center, y_center, width, height
        )

        if is_valid:
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            yolo_lines.append(yolo_line)
            stats["valid_boxes"] += 1
        else:
            stats["invalid_boxes"] += 1
            stats["errors"].append({
                "bbox": bbox_xyxy,
                "error": error_msg
            })

    # Write YOLO label file
    label_path = output_labels_dir / f"{image_path.stem}.txt"
    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_lines))
        if yolo_lines:  # Add final newline
            f.write('\n')

    return stats

def main():
    parser = argparse.ArgumentParser(description="Convert Grounded-SAM-2 JSON to YOLO format")
    parser.add_argument("--json_dir", type=str, required=True,
                        help="Directory containing JSON result files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for YOLO labels")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Optional: Images directory (for verification)")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)

    # Create output directories
    output_labels_dir = output_dir / "labels"
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files
    json_files = sorted(json_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON files in {json_dir}")
    print(f"Output labels directory: {output_labels_dir}")
    print("\nConverting JSON to YOLO format...")
    print("=" * 60)

    # Process all files
    all_stats = []
    for json_path in tqdm(json_files, desc="Converting"):
        stats = process_json_file(json_path, output_labels_dir)
        all_stats.append(stats)

    # Generate summary
    total_images = len(all_stats)
    total_detections = sum(s["detections"] for s in all_stats)
    total_valid = sum(s["valid_boxes"] for s in all_stats)
    total_invalid = sum(s["invalid_boxes"] for s in all_stats)
    images_with_no_detections = sum(1 for s in all_stats if s["detections"] == 0)
    images_with_errors = sum(1 for s in all_stats if s["invalid_boxes"] > 0)

    summary = {
        "total_images": total_images,
        "images_with_detections": total_images - images_with_no_detections,
        "images_with_no_detections": images_with_no_detections,
        "total_detections": total_detections,
        "valid_boxes": total_valid,
        "invalid_boxes": total_invalid,
        "images_with_errors": images_with_errors,
        "avg_detections_per_image": round(total_detections / total_images, 2) if total_images > 0 else 0
    }

    # Save summary
    summary_path = output_dir / "conversion_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save detailed stats
    stats_path = output_dir / "conversion_details.json"
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Total images:                 {total_images}")
    print(f"Images with detections:       {total_images - images_with_no_detections}")
    print(f"Images with NO detections:    {images_with_no_detections}")
    print(f"Total detections:             {total_detections}")
    print(f"Valid bboxes:                 {total_valid}")
    print(f"Invalid bboxes:               {total_invalid}")
    print(f"Images with errors:           {images_with_errors}")
    print(f"Avg detections per image:     {summary['avg_detections_per_image']}")
    print("=" * 60)
    print(f"\nYOLO labels saved to: {output_labels_dir}")
    print(f"Summary: {summary_path}")
    print(f"Details: {stats_path}")

    if total_invalid > 0:
        print(f"\n⚠️  WARNING: {total_invalid} invalid bboxes detected!")
        print("Check conversion_details.json for error details")
    else:
        print("\n✓ All bboxes valid!")

    if images_with_no_detections > 0:
        print(f"\n⚠️  WARNING: {images_with_no_detections} images had no detections")
        print("These images may need manual annotation")

    print("\nNext step: Review labels in Label Studio and fix any errors")

if __name__ == "__main__":
    main()
