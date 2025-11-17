#!/usr/bin/env python3
"""
Apply oversampling to dataset with augmentation support.

Supports both manual and dynamic oversampling strategies with optional
augmentation variations for each copy.

Usage:
    # Dynamic oversampling with augmentation (RECOMMENDED)
    python scripts/apply_oversampling.py \
        --dataset data/processed/merged_4datasets/data.yaml \
        --output data/processed/final_oversampled \
        --dynamic \
        --target-balance 0.65 \
        --max-multiplier 5 \
        --min-multiplier 2 \
        --vary-augmentation

    # Manual oversampling without augmentation
    python scripts/apply_oversampling.py \
        --dataset data/processed/merged_4datasets/data.yaml \
        --output data/processed/final_oversampled \
        --classes anthracnose_fruit_rot:5 powdery_mildew_fruit:4 \
        --no-augmentation
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
        help='Path to data.yaml'
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
        default=0.65,
        help='Target balance ratio for dynamic mode (0-1, default: 0.65)'
    )
    parser.add_argument(
        '--max-multiplier',
        type=int,
        default=5,
        help='Maximum multiplier for dynamic mode (default: 5)'
    )
    parser.add_argument(
        '--min-multiplier',
        type=int,
        default=2,
        help='Minimum multiplier for dynamic mode (default: 2)'
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
    parser.add_argument(
        '--vary-augmentation',
        action='store_true',
        default=True,
        help='Apply different augmentations to each copy (default: True)'
    )
    parser.add_argument(
        '--no-augmentation',
        action='store_true',
        help='Disable augmentation (just copy files)'
    )

    args = parser.parse_args()

    # Handle augmentation flags
    vary_augmentation = args.vary_augmentation and not args.no_augmentation

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
    print(f"Augmentation: {'Enabled (each copy gets unique augmentation)' if vary_augmentation else 'Disabled (exact copies)'}")
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
        min_multiplier=args.min_multiplier,
        vary_augmentation=vary_augmentation,
    )

    print(f"\n{'='*70}")
    print("✅ OVERSAMPLING COMPLETE!")
    print(f"{'='*70}")
    print(f"Output: {output_yaml}")

    if vary_augmentation:
        print(f"\n💡 Each copy has unique augmentation applied!")
        print(f"   Recommended training preset: balanced_oversampled")
        print(f"\nUse in training with:")
        print(f"  python scripts/train_model.py \\")
        print(f"    --data-yaml {output_yaml} \\")
        print(f"    --preset balanced_oversampled \\")
        print(f"    --model-size l")
    else:
        print(f"\n⚠️  WARNING: Augmentation disabled - using exact copies")
        print(f"   This may lead to overfitting!")
        print(f"\nUse in training with:")
        print(f"  --data-yaml {output_yaml}")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
