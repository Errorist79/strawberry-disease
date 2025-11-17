#!/usr/bin/env python3
"""Convert YOLOv8 results.csv to TensorBoard events"""

import pandas as pd
import sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def csv_to_tensorboard(csv_path: Path, output_dir: Path = None):
    """
    Convert YOLOv8 results.csv to TensorBoard event files.

    Args:
        csv_path: Path to results.csv
        output_dir: Output directory for TensorBoard events (default: same as csv)
    """
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        sys.exit(1)

    # Read CSV
    print(f"📖 Reading {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Output directory
    if output_dir is None:
        output_dir = csv_path.parent / 'tensorboard'
    output_dir.mkdir(exist_ok=True)

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=str(output_dir))
    print(f"📝 Writing TensorBoard events to {output_dir}")

    # Write metrics for each epoch
    for idx, row in df.iterrows():
        epoch = int(row['epoch'])

        # Training losses
        writer.add_scalar('Loss/train_box', row['train/box_loss'], epoch)
        writer.add_scalar('Loss/train_cls', row['train/cls_loss'], epoch)
        writer.add_scalar('Loss/train_dfl', row['train/dfl_loss'], epoch)

        # Validation losses
        writer.add_scalar('Loss/val_box', row['val/box_loss'], epoch)
        writer.add_scalar('Loss/val_cls', row['val/cls_loss'], epoch)
        writer.add_scalar('Loss/val_dfl', row['val/dfl_loss'], epoch)

        # Metrics
        writer.add_scalar('Metrics/mAP50', row['metrics/mAP50(B)'], epoch)
        writer.add_scalar('Metrics/mAP50-95', row['metrics/mAP50-95(B)'], epoch)
        writer.add_scalar('Metrics/precision', row['metrics/precision(B)'], epoch)
        writer.add_scalar('Metrics/recall', row['metrics/recall(B)'], epoch)

        # Learning rates
        writer.add_scalar('LearningRate/pg0', row['lr/pg0'], epoch)
        writer.add_scalar('LearningRate/pg1', row['lr/pg1'], epoch)
        writer.add_scalar('LearningRate/pg2', row['lr/pg2'], epoch)

    writer.close()
    print(f"✅ Done! {len(df)} epochs written")
    print(f"\n🚀 Start TensorBoard with:")
    print(f"   tensorboard --logdir {output_dir.parent} --port 6006")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/csv_to_tensorboard.py <path_to_results.csv>")
        print("Example: python scripts/csv_to_tensorboard.py runs/detect/strawberry_disease/results.csv")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    csv_to_tensorboard(csv_path)
