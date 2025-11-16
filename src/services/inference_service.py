"""
Inference service for processing images with the YOLO model.

This service:
1. Monitors database for unprocessed images
2. Runs disease detection inference
3. Calculates risk scores
4. Stores predictions in database
"""

import time
from pathlib import Path
from typing import List

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.core.logging import setup_logging
from src.core.ml import DiseaseClassifier, RiskCalculator
from src.models.base import get_sync_db
from src.models.image import Image, ImageStatus
from src.models.prediction import Prediction

logger = setup_logging("inference_service")


class InferenceService:
    """
    Service for running disease detection inference on captured images.
    """

    def __init__(self, batch_size: int = 10, polling_interval: int = 30):
        """
        Initialize inference service.

        Args:
            batch_size: Number of images to process in one batch
            polling_interval: Seconds to wait between database polls
        """
        self.settings = get_settings()
        self.batch_size = batch_size
        self.polling_interval = polling_interval

        # Initialize ML components
        self.classifier = DiseaseClassifier()
        self.risk_calculator = RiskCalculator()

        logger.info(
            f"InferenceService initialized (batch_size={batch_size}, "
            f"polling_interval={polling_interval}s)"
        )

    def load_model(self) -> None:
        """Load YOLO model into memory."""
        logger.info("Loading YOLO model...")
        try:
            self.classifier.load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_pending_images(self, db: Session, limit: int = None) -> List[Image]:
        """
        Get pending images from database.

        Args:
            db: Database session
            limit: Maximum number of images to retrieve

        Returns:
            List of pending Image objects
        """
        limit = limit or self.batch_size

        stmt = (
            select(Image)
            .where(Image.status == ImageStatus.PENDING.value)
            .order_by(Image.captured_at)
            .limit(limit)
        )

        images = db.execute(stmt).scalars().all()
        return list(images)

    def process_image(self, image: Image, db: Session) -> bool:
        """
        Process a single image with inference.

        Args:
            image: Image object to process
            db: Database session

        Returns:
            True if successful, False otherwise
        """
        try:
            # Mark as processing
            image.status = ImageStatus.PROCESSING.value
            db.commit()

            logger.debug(f"Processing image {image.id}: {image.file_path}")

            # Check if file exists
            image_path = Path(image.file_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Run inference
            detections = self.classifier.predict(image_path)

            # If no detections, treat as healthy
            if not detections:
                logger.debug(f"No detections for image {image.id}, marking as healthy")
                detections = []

            # Calculate risks and create predictions
            predictions_data = []
            for detection in detections:
                risk_score = self.risk_calculator.calculate_risk(
                    detection.class_label, detection.confidence
                )

                prediction = Prediction(
                    image_id=image.id,
                    class_label=detection.class_label,
                    confidence=detection.confidence,
                    bbox_x=detection.bbox[0] if detection.bbox else None,
                    bbox_y=detection.bbox[1] if detection.bbox else None,
                    bbox_width=detection.bbox[2] if detection.bbox else None,
                    bbox_height=detection.bbox[3] if detection.bbox else None,
                    risk_score=risk_score,
                )
                db.add(prediction)
                predictions_data.append((detection.class_label, detection.confidence))

            # Calculate image-level risk aggregation
            if predictions_data:
                aggregated = self.risk_calculator.aggregate_image_risk(predictions_data)
            else:
                # No detections = healthy
                aggregated = self.risk_calculator.aggregate_image_risk([])

            # Update image record
            image.status = ImageStatus.COMPLETED.value
            image.max_risk_score = aggregated["max_risk"]
            image.dominant_disease = aggregated["dominant_disease"]
            image.prediction_count = aggregated["prediction_count"]
            image.error_message = None

            db.commit()

            logger.info(
                f"Processed image {image.id}: {len(detections)} detections, "
                f"max_risk={aggregated['max_risk']}, "
                f"dominant={aggregated['dominant_disease']}"
            )

            return True

        except Exception as e:
            logger.error(f"Error processing image {image.id}: {e}")

            # Mark as failed
            try:
                image.status = ImageStatus.FAILED.value
                image.error_message = str(e)[:1000]
                db.commit()
            except Exception as commit_error:
                logger.error(f"Failed to update error status: {commit_error}")
                db.rollback()

            return False

    def process_batch(self, db: Session) -> int:
        """
        Process a batch of pending images.

        Args:
            db: Database session

        Returns:
            Number of images successfully processed
        """
        # Get pending images
        images = self.get_pending_images(db)

        if not images:
            return 0

        logger.info(f"Processing batch of {len(images)} images")

        success_count = 0
        for image in images:
            if self.process_image(image, db):
                success_count += 1

        logger.info(f"Batch complete: {success_count}/{len(images)} successful")
        return success_count

    def run(self) -> None:
        """Run the inference service continuously."""
        logger.info("Starting inference service")

        # Load model
        self.load_model()

        # Get database session
        db_gen = get_sync_db()
        db = next(db_gen)

        try:
            while True:
                try:
                    # Process batch
                    count = self.process_batch(db)

                    if count == 0:
                        logger.debug(
                            f"No pending images, sleeping for {self.polling_interval}s"
                        )
                        time.sleep(self.polling_interval)
                    else:
                        # Process next batch immediately if there were images
                        logger.debug("Checking for more images...")

                except Exception as e:
                    logger.error(f"Error in inference loop: {e}")
                    time.sleep(self.polling_interval)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass
            logger.info("Inference service stopped")


def main():
    """Main entry point for inference service."""
    service = InferenceService()
    service.run()


if __name__ == "__main__":
    main()
