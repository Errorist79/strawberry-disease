"""
Camera image collection service.

This service periodically captures images from configured cameras
and stores them in the database for processing.

In simulation mode, it uses sample images instead of real camera streams.
"""

import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.core.logging import setup_logging
from src.models.base import get_sync_db
from src.models.camera import Camera
from src.models.image import Image, ImageStatus

logger = setup_logging("camera_collector")


class CameraCollector:
    """
    Camera image collection service.

    Handles capturing images from cameras (or simulation) and storing them.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize camera collector.

        Args:
            config_path: Path to cameras.yaml config file
        """
        self.settings = get_settings()
        self.config_path = config_path or Path("config/cameras.yaml")

        # Load camera configuration
        self.config = self._load_config()

        # Ensure image storage directory exists
        self.settings.image_storage_path.mkdir(parents=True, exist_ok=True)

        # Simulation state
        self.simulation_enabled = self.config.get("simulation", {}).get("enabled", True)
        self.sample_images_dir = Path(
            self.config.get("simulation", {}).get("sample_images_dir", "data/raw/test/images")
        )

        logger.info(f"CameraCollector initialized (simulation_mode={self.simulation_enabled})")

    def _load_config(self) -> dict:
        """Load camera configuration from YAML."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using empty config")
            return {"cameras": []}

    def initialize_cameras(self, db: Session) -> None:
        """
        Initialize cameras in database from config.

        Args:
            db: Database session
        """
        logger.info("Initializing cameras in database")

        for camera_config in self.config.get("cameras", []):
            # Check if camera exists
            stmt = select(Camera).where(Camera.name == camera_config["name"])
            existing = db.execute(stmt).scalar_one_or_none()

            if existing:
                # Update existing camera
                existing.location = camera_config.get("location", existing.location)
                existing.row_id = camera_config.get("row_id", existing.row_id)
                existing.stream_url = camera_config.get("stream_url")
                existing.is_active = camera_config.get("is_active", True)
                existing.description = camera_config.get("description")
                logger.debug(f"Updated camera: {camera_config['name']}")
            else:
                # Create new camera
                camera = Camera(
                    name=camera_config["name"],
                    location=camera_config["location"],
                    row_id=camera_config["row_id"],
                    stream_url=camera_config.get("stream_url"),
                    is_active=camera_config.get("is_active", True),
                    description=camera_config.get("description"),
                )
                db.add(camera)
                logger.info(f"Added new camera: {camera_config['name']}")

        db.commit()
        logger.info("Camera initialization complete")

    def capture_from_real_camera(self, camera: Camera) -> Optional[Path]:
        """
        Capture image from a real camera stream.

        Args:
            camera: Camera object

        Returns:
            Path to captured image, or None if failed
        """
        if not camera.stream_url:
            logger.warning(f"No stream URL configured for camera {camera.name}")
            return None

        try:
            # Open RTSP/HTTP stream
            cap = cv2.VideoCapture(camera.stream_url)

            if not cap.isOpened():
                logger.error(f"Failed to open stream for camera {camera.name}")
                return None

            # Read frame
            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.error(f"Failed to read frame from camera {camera.name}")
                return None

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{camera.name}_{timestamp}.jpg"
            filepath = self.settings.image_storage_path / filename

            # Save image
            cv2.imwrite(str(filepath), frame)
            logger.info(f"Captured image from {camera.name}: {filepath}")

            return filepath

        except Exception as e:
            logger.error(f"Error capturing from camera {camera.name}: {e}")
            return None

    def capture_from_simulation(self, camera: Camera) -> Optional[Path]:
        """
        Simulate image capture by copying a sample image.

        Args:
            camera: Camera object

        Returns:
            Path to copied image, or None if failed
        """
        if not self.sample_images_dir.exists():
            logger.error(f"Sample images directory not found: {self.sample_images_dir}")
            return None

        # Get list of sample images
        sample_images = list(self.sample_images_dir.glob("*.jpg")) + list(
            self.sample_images_dir.glob("*.png")
        )

        if not sample_images:
            logger.error(f"No sample images found in {self.sample_images_dir}")
            return None

        # Select a random sample image
        source_image = random.choice(sample_images)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera.name}_{timestamp}{source_image.suffix}"
        filepath = self.settings.image_storage_path / filename

        # Copy image
        shutil.copy(source_image, filepath)
        logger.info(f"Simulated capture for {camera.name}: {filepath}")

        return filepath

    def capture_and_store(self, camera: Camera, db: Session) -> Optional[Image]:
        """
        Capture image from camera and store in database.

        Args:
            camera: Camera object
            db: Database session

        Returns:
            Created Image object, or None if failed
        """
        # Capture image
        if self.simulation_enabled:
            image_path = self.capture_from_simulation(camera)
        else:
            image_path = self.capture_from_real_camera(camera)

        if not image_path:
            return None

        # Create database record
        try:
            image = Image(
                camera_id=camera.id,
                file_path=str(image_path),
                captured_at=datetime.now(),
                status=ImageStatus.PENDING.value,
            )
            db.add(image)
            db.commit()
            db.refresh(image)

            logger.info(f"Stored image record: ID={image.id}, path={image_path}")
            return image

        except Exception as e:
            logger.error(f"Failed to store image record: {e}")
            db.rollback()
            return None

    def collect_from_all_cameras(self, db: Session) -> int:
        """
        Collect images from all active cameras.

        Args:
            db: Database session

        Returns:
            Number of images successfully captured
        """
        # Get active cameras
        stmt = select(Camera).where(Camera.is_active == True)
        cameras = db.execute(stmt).scalars().all()

        logger.info(f"Collecting from {len(cameras)} active cameras")

        success_count = 0
        for camera in cameras:
            result = self.capture_and_store(camera, db)
            if result:
                success_count += 1

        logger.info(f"Successfully captured {success_count}/{len(cameras)} images")
        return success_count

    def run(self) -> None:
        """Run the collection service continuously."""
        logger.info("Starting camera collection service")

        # Get interval from settings
        interval_minutes = self.settings.camera_capture_interval_minutes
        interval_seconds = interval_minutes * 60

        logger.info(f"Capture interval: {interval_minutes} minutes")

        # Initialize database session
        db_gen = get_sync_db()
        db = next(db_gen)

        try:
            # Initialize cameras
            self.initialize_cameras(db)

            # Main collection loop
            while True:
                logger.info("Starting capture cycle")

                try:
                    count = self.collect_from_all_cameras(db)
                    logger.info(f"Capture cycle complete: {count} images captured")
                except Exception as e:
                    logger.error(f"Error in capture cycle: {e}")

                logger.info(f"Sleeping for {interval_minutes} minutes")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass
            logger.info("Camera collection service stopped")


def main():
    """Main entry point for camera collector service."""
    collector = CameraCollector()
    collector.run()


if __name__ == "__main__":
    main()
