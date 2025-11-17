# Modular Training Framework

## Overview

Extensible training framework for YOLOv8 models with advanced features for handling overfitting, class imbalance, and ensemble learning.

## Architecture

```
src/training/
├── config/                      # Configuration modules
│   ├── base_config.py          # ModelConfig, DataConfig, TrainingConfig
│   ├── augmentation_config.py  # Augmentation presets
│   └── training_presets.py     # Complete training presets
├── data/                        # Data handling
│   ├── dataset_loader.py       # Dataset validation and loading
│   ├── augmentation.py         # Augmentation utilities
│   └── oversampling.py         # Class balancing strategies
├── models/                      # Training modules
│   ├── yolo_trainer.py         # Single model trainer
│   └── ensemble_trainer.py     # Ensemble trainer
├── callbacks/                   # Training callbacks
│   ├── metrics_logger.py       # Custom metrics logging
│   └── class_monitor.py        # Per-class performance monitoring
└── utils/                       # Utilities
    ├── visualization.py         # Plotting and visualization
    └── checkpointing.py        # Model checkpointing
```

## Features

### 1. Configuration Management

**Dataclass-based configurations** for type safety and validation:

```python
from src.training.config.base_config import ModelConfig, TrainingConfig

model_config = ModelConfig(
    model_size="l",
    input_size=640,
    confidence_threshold=0.25,
)

training_config = TrainingConfig(
    epochs=200,
    batch_size=-1,  # Auto
    patience=30,
    dropout=0.3,
)
```

### 2. Augmentation Presets

Three built-in augmentation levels:

- **Minimal**: Light augmentation for testing
- **Standard**: Balanced augmentation for general use
- **Aggressive**: Maximum augmentation to combat overfitting

```python
from src.training.config.augmentation_config import AggressiveAugmentation

aug = AggressiveAugmentation()
# Automatically configured with optimal parameters
```

### 3. Training Presets

Complete configurations for common scenarios:

```python
from src.training.config.training_presets import get_preset

# Anti-overfitting preset with all optimizations
preset = get_preset(
    preset_name="anti_overfitting",
    dataset_yaml="dataset.yaml",
    model_size="l",
)

trainer = YOLOTrainer(
    model_config=preset.model,
    data_config=preset.data,
    training_config=preset.training,
    augmentation_config=preset.augmentation,
)
```

### 4. Dataset Validation

Comprehensive dataset validation with statistics:

```python
from src.training.data.dataset_loader import validate_dataset

is_valid = validate_dataset("dataset.yaml", verbose=True)
```

Output:
```
Dataset Validation: dataset.yaml
❌ ERRORS: None
⚠️  WARNINGS:
  - Severe class imbalance detected (ratio: 28.1:1)

📊 STATISTICS:
  TRAIN:
    Images: 1450
    Instances: 2847
    Class distribution:
      leaf_spot: 478 (16.8%)
      anthracnose_fruit_rot: 17 (0.6%)
      ...
✅ Dataset is VALID
```

### 5. Class Balancing

Two strategies for handling imbalanced classes:

**Class Weights:**
```python
data_config = DataConfig(
    dataset_yaml="dataset.yaml",
    class_weights={
        "anthracnose_fruit_rot": 3.0,
        "powdery_mildew_fruit": 3.0,
    }
)
```

**Oversampling:**
```python
from src.training.data.oversampling import create_oversampled_dataset

new_dataset = create_oversampled_dataset(
    dataset_yaml="original.yaml",
    oversample_config={
        "anthracnose_fruit_rot": 3,  # 3x multiplier
        "powdery_mildew_fruit": 3,
    },
    output_dir="data/oversampled",
)
```

### 6. Single Model Training

High-level API for training individual models:

```python
from src.training.models.yolo_trainer import YOLOTrainer

trainer = YOLOTrainer(
    model_config=model_config,
    data_config=data_config,
    training_config=training_config,
    augmentation_config=aug_config,
)

# Train
results = trainer.train()

# Validate
val_results = trainer.validate(split="val")
test_results = trainer.validate(split="test")

# Export
trainer.export(format="onnx", output_path="model.onnx")
```

### 7. Ensemble Training

Train multiple models with different configurations:

```python
from src.training.models.ensemble_trainer import EnsembleTrainer

models = [
    (model_config_1, training_config_1, aug_aggressive),
    (model_config_2, training_config_2, aug_standard),
    (model_config_3, training_config_3, aug_standard),
]

ensemble = EnsembleTrainer(
    models=models,
    data_config=data_config,
    output_dir="runs/ensemble",
    ensemble_name="strawberry_ensemble",
)

# Train all models
results = ensemble.train_all(parallel=False)

# Validate ensemble
ensemble_results = ensemble.validate_ensemble(split="test")

# Predict with ensemble (weighted voting)
predictions = ensemble.predict_ensemble("image.jpg")
```

## Configuration Classes

### ModelConfig

```python
@dataclass
class ModelConfig:
    model_size: str = "l"              # n, s, m, l, x
    pretrained: bool = True
    input_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
```

### DataConfig

```python
@dataclass
class DataConfig:
    dataset_yaml: Path
    train_split: str = "train"
    val_split: str = "val"
    test_split: Optional[str] = "test"
    class_weights: Optional[Dict[str, float]] = None
    oversample_classes: Optional[Dict[str, int]] = None
    cache: str = "disk"
    workers: int = 8
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Basic
    epochs: int = 200
    batch_size: int = -1
    device: str = "cpu"

    # Optimization
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.001

    # Regularization
    dropout: float = 0.3

    # Early stopping
    patience: int = 30
    save_period: int = 10
```

### AugmentationConfig

```python
@dataclass
class AugmentationConfig:
    # HSV
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    # Geometric
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5

    # Advanced
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
```

## Callbacks

### MetricsLogger

Logs detailed metrics during training:

```python
from src.training.callbacks import MetricsLogger

logger = MetricsLogger(log_dir="runs/metrics")

trainer = YOLOTrainer(..., callbacks=[logger])
```

### ClassPerformanceMonitor

Monitors per-class performance:

```python
from src.training.callbacks import ClassPerformanceMonitor

monitor = ClassPerformanceMonitor(
    class_names=["class1", "class2"],
    alert_threshold=0.3,
)

trainer = YOLOTrainer(..., callbacks=[monitor])
```

## Utilities

### Visualization

```python
from src.training.utils.visualization import plot_training_curves

plot_training_curves(
    metrics_file="runs/metrics/metrics_history.json",
    output_path="training_curves.png",
)
```

### Checkpointing

```python
from src.training.utils.checkpointing import save_checkpoint, load_checkpoint

# Save
save_checkpoint(
    model_path="best.pt",
    config=config_dict,
    output_dir="checkpoints",
)

# Load
checkpoint = load_checkpoint("checkpoints/checkpoint")
```

## Usage Examples

### Quick Test

```python
from src.training.config.training_presets import get_preset

preset = get_preset("quick_test", dataset_yaml="dataset.yaml")
trainer = YOLOTrainer(
    model_config=preset.model,
    data_config=preset.data,
    training_config=preset.training,
    augmentation_config=preset.augmentation,
)
trainer.train()
```

### Production Training

```python
from src.training.config.training_presets import get_preset

preset = get_preset(
    "anti_overfitting",
    dataset_yaml="dataset.yaml",
    model_size="l",
    class_weights={
        "anthracnose_fruit_rot": 3.0,
        "powdery_mildew_fruit": 3.0,
    }
)

trainer = YOLOTrainer(**preset.__dict__)
results = trainer.train()
```

### Ensemble for Maximum Accuracy

```python
from src.training.models.ensemble_trainer import EnsembleTrainer
from src.training.config.augmentation_config import (
    AggressiveAugmentation,
    StandardAugmentation,
)

models = [
    (
        ModelConfig(model_size="l"),
        TrainingConfig(epochs=200, device="0"),
        AggressiveAugmentation(),
    ),
    (
        ModelConfig(model_size="l"),
        TrainingConfig(epochs=200, device="0"),
        StandardAugmentation(),
    ),
    (
        ModelConfig(model_size="m"),
        TrainingConfig(epochs=200, device="0"),
        StandardAugmentation(),
    ),
]

ensemble = EnsembleTrainer(
    models=models,
    data_config=data_config,
    output_dir="runs/ensemble",
)

ensemble.train_all()
```

## Extension Guide

### Adding Custom Augmentation

```python
from src.training.config.augmentation_config import AugmentationConfig

@dataclass
class MyCustomAugmentation(AugmentationConfig):
    hsv_h: float = 0.08
    hsv_s: float = 0.4
    hsv_v: float = 0.2
    degrees: float = 45.0
    # ... customize parameters
```

### Adding Custom Callbacks

```python
class MyCallback:
    def on_train_start(self, trainer):
        print("Training started!")

    def on_train_epoch_end(self, trainer):
        # Access trainer.metrics, trainer.model, etc.
        pass

trainer = YOLOTrainer(..., callbacks=[MyCallback()])
```

### Creating Custom Presets

```python
from src.training.config.training_presets import PresetConfig

def my_preset(dataset_yaml: str) -> PresetConfig:
    return PresetConfig(
        model=ModelConfig(...),
        data=DataConfig(dataset_yaml=dataset_yaml, ...),
        training=TrainingConfig(...),
        augmentation=MyCustomAugmentation(),
        description="My custom preset",
    )
```

## Performance Optimizations

1. **Auto-batch sizing**: Use `batch_size=-1` for automatic optimal batch size
2. **Disk caching**: Use `cache='disk'` to save GPU memory
3. **Mixed precision**: `amp=True` (default) for faster training
4. **Multi-worker data loading**: `workers=8` for CPU preprocessing

## Best Practices

1. Always validate dataset before training
2. Use presets for common scenarios
3. Start with `anti_overfitting` preset for imbalanced data
4. Use ensemble for production models
5. Monitor val-test gap to detect overfitting
6. Save training config for reproducibility
7. Use callbacks for custom monitoring

## License

Same as parent project
