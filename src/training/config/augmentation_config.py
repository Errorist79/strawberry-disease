"""
Augmentation configuration presets.

Provides different augmentation strategies optimized for strawberry disease detection,
based on analysis in docs/analyze.md.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class AugmentationConfig:
    """Base augmentation configuration."""

    # HSV color space augmentation
    hsv_h: float = 0.015  # Hue
    hsv_s: float = 0.7  # Saturation
    hsv_v: float = 0.4  # Value (brightness)

    # Geometric transformations
    degrees: float = 10.0  # Rotation degrees
    translate: float = 0.1  # Translation fraction
    scale: float = 0.5  # Scale variance
    shear: float = 0.0  # Shear degrees
    perspective: float = 0.0  # Perspective distortion

    # Flip augmentation
    fliplr: float = 0.5  # Horizontal flip probability
    flipud: float = 0.0  # Vertical flip probability

    # Advanced augmentation
    mosaic: float = 1.0  # Mosaic augmentation probability
    mixup: float = 0.0  # Mixup augmentation probability
    copy_paste: float = 0.0  # Copy-paste augmentation probability

    # Image quality
    blur: float = 0.0  # Blur kernel size
    noise: float = 0.0  # Gaussian noise std

    def to_dict(self) -> Dict:
        """Convert to dictionary for YOLO."""
        return {
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "fliplr": self.fliplr,
            "flipud": self.flipud,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
        }


@dataclass
class StandardAugmentation(AugmentationConfig):
    """
    Standard augmentation configuration.

    Moderate augmentation suitable for general training.
    """

    # Slightly enhanced from defaults
    hsv_h: float = 0.03
    hsv_s: float = 0.6
    hsv_v: float = 0.35
    degrees: float = 15.0
    translate: float = 0.15
    scale: float = 0.6
    mosaic: float = 1.0
    mixup: float = 0.05
    copy_paste: float = 0.1


@dataclass
class AggressiveAugmentation(AugmentationConfig):
    """
    Aggressive augmentation configuration.

    Based on recommendations from analyze.md to combat overfitting.
    Designed for strawberry disease detection in greenhouse environments.
    """

    # Enhanced HSV for greenhouse lighting variations
    hsv_h: float = 0.05  # Increased from 0.015 (greenhouse light variations)
    hsv_s: float = 0.5  # Decreased from 0.7 (more saturation changes)
    hsv_v: float = 0.3  # Decreased from 0.4 (brightness variations)

    # Enhanced geometric transformations
    degrees: float = 30.0  # Increased from 10 (more rotation)
    translate: float = 0.2  # Increased from 0.1 (position changes)
    scale: float = 0.7  # Increased from 0.5 (size variations)
    shear: float = 2.0  # Added shear for more diversity
    perspective: float = 0.0005  # Subtle perspective changes

    # Flip augmentation
    fliplr: float = 0.5  # Horizontal flip
    flipud: float = 0.1  # Occasional vertical flip (diseases can appear upside down)

    # Advanced augmentation (analyze.md recommendations)
    mosaic: float = 1.0  # Always use mosaic
    mixup: float = 0.15  # Class mixing for better generalization
    copy_paste: float = 0.3  # Instance augmentation

    # Subtle quality variations
    blur: float = 0.01  # Slight blur for camera focus variations
    noise: float = 0.005  # Minimal noise


@dataclass
class MinimalAugmentation(AugmentationConfig):
    """
    Minimal augmentation configuration.

    Very light augmentation for testing or when overfitting is not a concern.
    """

    hsv_h: float = 0.01
    hsv_s: float = 0.8
    hsv_v: float = 0.5
    degrees: float = 5.0
    translate: float = 0.05
    scale: float = 0.3
    mosaic: float = 0.5
    mixup: float = 0.0
    copy_paste: float = 0.0


# Preset mapping for easy access
AUGMENTATION_PRESETS = {
    "minimal": MinimalAugmentation,
    "standard": StandardAugmentation,
    "aggressive": AggressiveAugmentation,
}


def get_augmentation_preset(name: str) -> AugmentationConfig:
    """
    Get augmentation preset by name.

    Args:
        name: Preset name (minimal, standard, aggressive)

    Returns:
        AugmentationConfig instance

    Raises:
        ValueError: If preset name is not recognized
    """
    name = name.lower()
    if name not in AUGMENTATION_PRESETS:
        raise ValueError(
            f"Unknown augmentation preset: {name}. "
            f"Available: {list(AUGMENTATION_PRESETS.keys())}"
        )

    return AUGMENTATION_PRESETS[name]()
