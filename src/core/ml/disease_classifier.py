"""
Disease classifier wrapper for YOLOv8 model.

This module provides a high-level interface for loading the YOLO model
and running inference on strawberry images.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class Detection:
    """Represents a single disease detection."""

    def __init__(
        self,
        class_label: str,
        confidence: float,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ):
        """
        Initialize detection.

        Args:
            class_label: Disease class name
            confidence: Detection confidence (0-1)
            bbox: Bounding box as (x, y, width, height), optional
        """
        self.class_label = class_label
        self.confidence = confidence
        self.bbox = bbox

    def __repr__(self) -> str:
        return f"Detection(class='{self.class_label}', conf={self.confidence:.3f})"


class DiseaseClassifier:
    """
    Wrapper for YOLOv8 disease detection model.

    This class handles:
    - Loading the YOLO model
    - Running inference on images
    - Parsing and filtering results
    """

    def __init__(self, model_path: Optional[Path] = None, confidence_threshold: Optional[float] = None):
        """
        Initialize the disease classifier.

        Args:
            model_path: Path to YOLO weights file. Uses config default if None.
            confidence_threshold: Minimum confidence threshold. Uses config default if None.
        """
        settings = get_settings()

        self.model_path = model_path or settings.yolo_model_path
        self.confidence_threshold = confidence_threshold or settings.yolo_confidence_threshold
        self.iou_threshold = settings.yolo_iou_threshold

        self.model: Optional[YOLO] = None
        self._is_loaded = False

        logger.info(
            f"DiseaseClassifier initialized with model_path={self.model_path}, "
            f"conf_threshold={self.confidence_threshold}"
        )

    def load_model(self) -> None:
        """
        Load the YOLO model into memory.

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if self._is_loaded:
            logger.debug("Model already loaded, skipping")
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        try:
            logger.info(f"Loading YOLO model from {self.model_path}")
            # PyTorch 2.6+ changed the default value of weights_only to True
            # TORCH_FORCE_WEIGHTS_ONLY_LOAD=0 is set in the Dockerfile for trusted model files
            self.model = YOLO(str(self.model_path))
            self._is_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(
        self, image_path: Path, confidence_threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to image file
            confidence_threshold: Override default confidence threshold

        Returns:
            List of Detection objects

        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If model is not loaded
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        conf_thresh = confidence_threshold or self.confidence_threshold

        try:
            logger.debug(f"Running inference on {image_path}")

            # Run YOLO inference
            results = self.model(
                str(image_path),
                conf=conf_thresh,
                iou=self.iou_threshold,
                verbose=False,
            )

            # Parse results
            detections = []
            for result in results:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # Get class ID and name
                    class_id = int(boxes.cls[i])
                    class_name = result.names[class_id]

                    # Get confidence
                    confidence = float(boxes.conf[i])

                    # Get bounding box coordinates
                    box = boxes.xywh[i]  # x_center, y_center, width, height
                    bbox = tuple(float(x) for x in box)

                    detection = Detection(
                        class_label=class_name,
                        confidence=confidence,
                        bbox=bbox,
                    )
                    detections.append(detection)

            logger.info(f"Found {len(detections)} detections in {image_path.name}")
            return detections

        except Exception as e:
            logger.error(f"Inference failed for {image_path}: {e}")
            raise

    def predict_batch(
        self, image_paths: List[Path], confidence_threshold: Optional[float] = None
    ) -> List[List[Detection]]:
        """
        Run inference on multiple images.

        Args:
            image_paths: List of image paths
            confidence_threshold: Override default confidence threshold

        Returns:
            List of detection lists (one per image)
        """
        results = []
        for image_path in image_paths:
            try:
                detections = self.predict(image_path, confidence_threshold)
                results.append(detections)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append([])

        return results

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._is_loaded:
            del self.model
            self.model = None
            self._is_loaded = False
            logger.info("Model unloaded from memory")
