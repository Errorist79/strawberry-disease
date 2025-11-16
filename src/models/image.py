"""Image model for storing captured images."""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin


class ImageStatus(str, Enum):
    """Image processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Image(Base, TimestampMixin):
    """
    Image model representing a captured image from a camera.

    Attributes:
        id: Unique identifier
        camera_id: Foreign key to camera
        file_path: Path to the image file
        captured_at: When the image was captured
        status: Processing status
        max_risk_score: Maximum risk score from all predictions on this image
        dominant_disease: Disease class with highest risk
        prediction_count: Number of predictions/detections in this image
        error_message: Error message if processing failed
        created_at: Record creation timestamp
        updated_at: Record update timestamp
    """

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    file_path: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default=ImageStatus.PENDING.value, nullable=False, index=True
    )
    max_risk_score: Mapped[Optional[int]] = mapped_column(nullable=True)
    dominant_disease: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    prediction_count: Mapped[int] = mapped_column(default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Relationships
    camera: Mapped["Camera"] = relationship("Camera", back_populates="images")
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="image", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Image(id={self.id}, camera_id={self.camera_id}, status='{self.status}')>"
