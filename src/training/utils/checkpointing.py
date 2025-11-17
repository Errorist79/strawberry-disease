"""
Checkpointing utilities.

Handles saving and loading of training checkpoints.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Optional


def save_checkpoint(
    model_path: Path,
    config: Dict,
    output_dir: Path,
    checkpoint_name: str = "checkpoint",
):
    """
    Save training checkpoint.

    Args:
        model_path: Path to model weights
        config: Training configuration dict
        output_dir: Output directory
        checkpoint_name: Checkpoint name
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(exist_ok=True)

    # Copy model weights
    shutil.copy(model_path, checkpoint_dir / "weights.pt")

    # Save config
    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Checkpoint saved: {checkpoint_dir}")


def load_checkpoint(checkpoint_dir: Path) -> Dict:
    """
    Load training checkpoint.

    Args:
        checkpoint_dir: Checkpoint directory

    Returns:
        Dictionary with model_path and config
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    weights_path = checkpoint_dir / "weights.pt"
    config_path = checkpoint_dir / "config.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

    return {"model_path": weights_path, "config": config}
