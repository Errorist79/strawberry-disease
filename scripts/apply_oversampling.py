#!/usr/bin/env python3
"""
Apply oversampling to dataset.

Supports both manual and dynamic oversampling strategies.

Usage:
    # Dynamic oversampling (automatic calculation)
    python scripts/apply_oversampling.py \
        --dataset data/processed/merged_4datasets/data.yaml \
        --output data/processed/final_oversampled \
        --dynamic \
        --target-balance 0.8 \
        --max-multiplier 5

    # Manual oversampling
    python scripts/apply_oversampling.py \
        --dataset data/processed/merged_4datasets/data.yaml \
        --output data/processed/final_oversampled \
        --classes anthracnose_fruit_rot:5 powdery_mildew_fruit:4
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.data.oversampling import (
    calculate_dynamic_oversample_ratios,
    create_oversampled_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Apply oversampling to balance dataset classes"
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Path to dataset.yaml'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for oversampled dataset'
    )
    parser.add_argument(
        '--dynamic',
        action='store_true',
        help='Use dynamic oversampling (automatic calculation)'
    )
    parser.add_argument(
        '--target-balance',
        type=float,
        default=0.8,
        help='Target balance ratio for dynamic mode (0-1, default: 0.8)'
    )
    parser.add_argument(
        '--max-multiplier',
        type=int,
        default=5,
        help='Maximum multiplier for dynamic mode (default: 5)'
    )
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        help='Manual class multipliers (format: class_name:multiplier)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train'],
        help='Splits to oversample (default: train)'
    )

    args = parser.parse_args()

    # Validate dataset exists
    if not args.dataset.exists():
        print(f"❌ Dataset not found: {args.dataset}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("Dataset Oversampling")
    print(f"{'='*70}")
    print(f"Input: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Mode: {'Dynamic' if args.dynamic else 'Manual'}")
    print(f"Splits: {', '.join(args.splits)}")
    print(f"{'='*70}\n")

    # Parse manual classes if provided
    oversample_config = None
    if args.classes and not args.dynamic:
        oversample_config = {}
        for class_spec in args.classes:
            try:
                class_name, multiplier = class_spec.split(':')
                oversample_config[class_name] = int(multiplier)
            except ValueError:
                print(f"❌ Invalid class format: {class_spec}")
                print("Expected format: class_name:multiplier")
                sys.exit(1)

        print("Manual oversampling configuration:")
        for class_name, multiplier in oversample_config.items():
            print(f"  {class_name}: {multiplier}x")
        print()

    # Create oversampled dataset
    output_yaml = create_oversampled_dataset(
        dataset_yaml=args.dataset,
        oversample_config=oversample_config,
        output_dir=args.output,
        splits=args.splits,
        dynamic=args.dynamic,
        target_balance=args.target_balance,
        max_multiplier=args.max_multiplier,
    )

    print(f"\n{'='*70}")
    print("✅ OVERSAMPLING COMPLETE!")
    print(f"{'='*70}")
    print(f"Output: {output_yaml}")
    print(f"\nUse in training with:")
    print(f"  --data-yaml {output_yaml}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
