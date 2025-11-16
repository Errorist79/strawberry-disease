"""Camera model for storing camera information."""

from typing import Optional

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin


class Camera(Base, TimestampMixin):
    """
    Camera model representing a physical camera in the greenhouse.

    Attributes:
        id: Unique identifier
        name: Camera name/identifier
        location: Physical location description
        row_id: Row or block identifier this camera monitors
        stream_url: RTSP or HTTP stream URL (optional for simulated cameras)
        is_active: Whether the camera is currently active
        description: Additional description
        created_at: Timestamp when camera was added
        updated_at: Timestamp of last update
    """

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    location: Mapped[str] = mapped_column(String(200), nullable=False)
    row_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    stream_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Relationships
    images: Mapped[list["Image"]] = relationship(
        "Image", back_populates="camera", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Camera(id={self.id}, name='{self.name}', row_id='{self.row_id}')>"
