#!/usr/bin/env python3
"""
Roboflow dataset downloader.

Downloads strawberry disease datasets from Roboflow Universe.

Supported datasets:
1. strawberry-disease/strawberry-disease-detection-dataset (4,918 images)
2. research-proj/strawberry-diseases-detection (2,757 images)

Usage:
    python scripts/download_roboflow.py \
        --dataset strawberry-disease/strawberry-disease-detection-dataset \
        --output data/external/roboflow_4918
"""

import argparse
import subprocess
import sys
from pathlib import Path


def download_roboflow_dataset(dataset_id: str, output_dir: Path, format: str = 'yolov8') -> Path:
    """
    Download dataset from Roboflow Universe using roboflow CLI.

    Args:
        dataset_id: Roboflow dataset ID (workspace/dataset-name)
        output_dir: Output directory
        format: Export format (yolov8, yolov5, coco, etc.)

    Returns:
        Path to downloaded dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Downloading Roboflow Dataset")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_id}")
    print(f"Format: {format}")
    print(f"Output: {output_dir}")
    print()

    # Check if roboflow is installed
    try:
        import roboflow
    except ImportError:
        print("❌ Roboflow package not found!")
        print("Install with: pip install roboflow")
        sys.exit(1)

    try:
        # Initialize Roboflow
        print("Initializing Roboflow...")
        print("\nℹ️  You'll need a Roboflow API key.")
        print("Get your API key from: https://app.roboflow.com/settings/api")
        print()

        api_key = input("Enter your Roboflow API key (or press Enter to use environment variable ROBOFLOW_API_KEY): ").strip()

        if not api_key:
            import os
            api_key = os.getenv('ROBOFLOW_API_KEY')

        if not api_key:
            print("❌ No API key provided!")
            print("Either:")
            print("1. Enter API key when prompted")
            print("2. Set environment variable: export ROBOFLOW_API_KEY=your_key")
            sys.exit(1)

        rf = roboflow.Roboflow(api_key=api_key)

        # Parse dataset ID
        parts = dataset_id.split('/')
        if len(parts) != 2:
            print(f"❌ Invalid dataset ID format: {dataset_id}")
            print("Expected format: workspace/dataset-name")
            sys.exit(1)

        workspace, dataset_name = parts

        print(f"Accessing workspace: {workspace}")
        print(f"Dataset: {dataset_name}")

        # Get project
        project = rf.workspace(workspace).project(dataset_name)

        # Download latest version (version 1 is usually the latest/published version)
        print("\nDownloading dataset (using latest version)...")

        # Most Roboflow datasets use version 1 as the published/stable version
        # The version() method takes an integer
        dataset = project.version(1).download(
            format,
            location=str(output_dir)
        )

        print(f"\n✅ Download complete!")
        print(f"Dataset saved to: {output_dir}")

        return output_dir

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check API key is valid")
        print("2. Verify dataset ID is correct")
        print("3. Ensure you have access to the dataset")
        print("4. Check internet connection")
        sys.exit(1)


def download_via_web(dataset_id: str, output_dir: Path):
    """
    Print instructions for manual download via web.

    Alternative when API download fails or for easier setup.
    """
    print(f"\n{'='*70}")
    print("Manual Download Instructions")
    print(f"{'='*70}")

    # Map common dataset IDs to URLs
    dataset_urls = {
        'strawberry-disease/strawberry-disease-detection-dataset':
            'https://universe.roboflow.com/strawberry-disease/strawberry-disease-detection-dataset',
        'research-proj/strawberry-diseases-detection':
            'https://universe.roboflow.com/research-proj/strawberry-diseases-detection',
    }

    url = dataset_urls.get(dataset_id, f"https://universe.roboflow.com/{dataset_id}")

    print(f"\n1. Go to: {url}")
    print("2. Click 'Download Dataset'")
    print("3. Select format: YOLO v8")
    print("4. Download and extract to:")
    print(f"   {output_dir.absolute()}")
    print("\n5. Verify data.yaml exists:")
    print(f"   {output_dir / 'data.yaml'}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download Roboflow strawberry disease datasets"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=[
            'strawberry-disease/strawberry-disease-detection-dataset',
            'research-proj/strawberry-diseases-detection',
        ],
        help='Roboflow dataset ID'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for downloaded dataset'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='yolov8',
        choices=['yolov8', 'yolov5', 'yolov7', 'coco', 'voc'],
        help='Dataset export format'
    )
    parser.add_argument(
        '--manual',
        action='store_true',
        help='Show manual download instructions instead of API download'
    )

    args = parser.parse_args()

    if args.manual:
        download_via_web(args.dataset, args.output)
    else:
        download_roboflow_dataset(args.dataset, args.output, args.format)

        print(f"\n🎉 SUCCESS!")
        print(f"Dataset ready at: {args.output}")
        print(f"Use in training with: --data-yaml {args.output / 'data.yaml'}")


if __name__ == '__main__':
    main()
