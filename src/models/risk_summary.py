"""Risk summary model for storing aggregated risk data by row and time period."""

from datetime import datetime

from sqlalchemy import DateTime, Float, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class RiskSummary(Base, TimestampMixin):
    """
    Risk summary model for aggregated risk metrics by row/block and time period.

    Attributes:
        id: Unique identifier
        row_id: Row or block identifier
        time_bucket: Start of the time period this summary covers
        avg_risk_score: Average risk score for this row/time period
        max_risk_score: Maximum risk score for this row/time period
        min_risk_score: Minimum risk score for this row/time period
        sample_count: Number of images/predictions included in this summary
        dominant_disease: Most common disease class in this period
        disease_counts: JSON object with disease class counts (optional)
        created_at: Record creation timestamp
        updated_at: Record update timestamp
    """

    __tablename__ = "risk_summaries"
    __table_args__ = (
        UniqueConstraint("row_id", "time_bucket", name="uq_row_time_bucket"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    row_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    time_bucket: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    avg_risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    max_risk_score: Mapped[int] = mapped_column(nullable=False)
    min_risk_score: Mapped[int] = mapped_column(nullable=False)
    sample_count: Mapped[int] = mapped_column(nullable=False)
    dominant_disease: Mapped[str] = mapped_column(String(50), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<RiskSummary(row_id='{self.row_id}', "
            f"time={self.time_bucket}, avg_risk={self.avg_risk_score:.1f})>"
        )
