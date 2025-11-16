"""Prediction model for storing model inference results."""

from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin


class Prediction(Base, TimestampMixin):
    """
    Prediction model representing a single disease detection in an image.

    Attributes:
        id: Unique identifier
        image_id: Foreign key to image
        class_label: Disease class detected (e.g., 'gray_mold', 'healthy')
        confidence: Model confidence score (0-1)
        bbox_x: Bounding box x coordinate
        bbox_y: Bounding box y coordinate
        bbox_width: Bounding box width
        bbox_height: Bounding box height
        risk_score: Calculated risk score (0-100)
        created_at: Record creation timestamp
        updated_at: Record update timestamp
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    class_label: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_width: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bbox_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # Relationships
    image: Mapped["Image"] = relationship("Image", back_populates="predictions")

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, class='{self.class_label}', "
            f"confidence={self.confidence:.2f}, risk={self.risk_score})>"
        )
