"""
Risk aggregation service.

This service periodically aggregates risk data by row/block and time period,
creating summaries for dashboard visualization and alerting.
"""

import time
from collections import Counter
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.core.config import get_settings
from src.core.logging import setup_logging
from src.models.base import get_sync_db
from src.models.camera import Camera
from src.models.image import Image, ImageStatus
from src.models.risk_summary import RiskSummary

logger = setup_logging("risk_aggregator")


class RiskAggregator:
    """
    Service for aggregating risk data by row and time period.
    """

    def __init__(self, aggregation_interval_hours: Optional[int] = None):
        """
        Initialize risk aggregator.

        Args:
            aggregation_interval_hours: Hours per aggregation window
        """
        self.settings = get_settings()
        self.interval_hours = aggregation_interval_hours or self.settings.risk_aggregation_interval_hours

        logger.info(f"RiskAggregator initialized (interval={self.interval_hours}h)")

    def get_time_bucket(self, dt: datetime) -> datetime:
        """
        Get the time bucket for a given datetime.

        Args:
            dt: Datetime to bucket

        Returns:
            Start of the time bucket
        """
        # Round down to the nearest interval
        hours_since_epoch = int(dt.timestamp() / 3600)
        bucket_hours = (hours_since_epoch // self.interval_hours) * self.interval_hours
        return datetime.fromtimestamp(bucket_hours * 3600)

    def get_rows_with_data(self, db: Session, time_bucket: datetime) -> List[str]:
        """
        Get list of row IDs that have completed images in the time bucket.

        Args:
            db: Database session
            time_bucket: Time bucket start

        Returns:
            List of row IDs
        """
        next_bucket = time_bucket + timedelta(hours=self.interval_hours)

        stmt = (
            select(Camera.row_id)
            .join(Image)
            .where(
                and_(
                    Image.status == ImageStatus.COMPLETED.value,
                    Image.captured_at >= time_bucket,
                    Image.captured_at < next_bucket,
                )
            )
            .distinct()
        )

        rows = db.execute(stmt).scalars().all()
        return list(rows)

    def aggregate_row_risk(
        self, db: Session, row_id: str, time_bucket: datetime
    ) -> Optional[dict]:
        """
        Aggregate risk data for a specific row and time bucket.

        Args:
            db: Database session
            row_id: Row identifier
            time_bucket: Time bucket start

        Returns:
            Aggregation dict or None if no data
        """
        next_bucket = time_bucket + timedelta(hours=self.interval_hours)

        # Query all completed images for this row in the time bucket
        stmt = (
            select(Image)
            .join(Camera)
            .where(
                and_(
                    Camera.row_id == row_id,
                    Image.status == ImageStatus.COMPLETED.value,
                    Image.captured_at >= time_bucket,
                    Image.captured_at < next_bucket,
                )
            )
        )

        images = db.execute(stmt).scalars().all()

        if not images:
            return None

        # Collect risk scores and diseases
        risk_scores = []
        diseases = []

        for image in images:
            if image.max_risk_score is not None:
                risk_scores.append(image.max_risk_score)
            if image.dominant_disease:
                diseases.append(image.dominant_disease)

        if not risk_scores:
            return None

        # Calculate statistics
        avg_risk = sum(risk_scores) / len(risk_scores)
        max_risk = max(risk_scores)
        min_risk = min(risk_scores)

        # Find most common disease
        if diseases:
            disease_counts = Counter(diseases)
            dominant_disease = disease_counts.most_common(1)[0][0]
        else:
            dominant_disease = "unknown"

        return {
            "row_id": row_id,
            "time_bucket": time_bucket,
            "avg_risk_score": avg_risk,
            "max_risk_score": max_risk,
            "min_risk_score": min_risk,
            "sample_count": len(risk_scores),
            "dominant_disease": dominant_disease,
        }

    def create_or_update_summary(self, db: Session, data: dict) -> bool:
        """
        Create or update risk summary in database.

        Args:
            db: Database session
            data: Aggregation data dict

        Returns:
            True if created/updated, False if skipped (already exists)
        """
        try:
            # Check if summary already exists
            stmt = select(RiskSummary).where(
                and_(
                    RiskSummary.row_id == data["row_id"],
                    RiskSummary.time_bucket == data["time_bucket"],
                )
            )
            existing = db.execute(stmt).scalar_one_or_none()

            if existing:
                # Update existing summary
                existing.avg_risk_score = data["avg_risk_score"]
                existing.max_risk_score = data["max_risk_score"]
                existing.min_risk_score = data["min_risk_score"]
                existing.sample_count = data["sample_count"]
                existing.dominant_disease = data["dominant_disease"]
                logger.debug(
                    f"Updated summary for row={data['row_id']}, time={data['time_bucket']}"
                )
            else:
                # Create new summary
                summary = RiskSummary(**data)
                db.add(summary)
                logger.info(
                    f"Created summary for row={data['row_id']}, time={data['time_bucket']}, "
                    f"avg_risk={data['avg_risk_score']:.1f}"
                )

            db.commit()
            return True

        except IntegrityError:
            db.rollback()
            logger.warning(f"Summary already exists (race condition), skipping")
            return False
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating/updating summary: {e}")
            return False

    def aggregate_time_bucket(self, db: Session, time_bucket: datetime) -> int:
        """
        Aggregate all rows for a specific time bucket.

        Args:
            db: Database session
            time_bucket: Time bucket start

        Returns:
            Number of summaries created/updated
        """
        logger.info(f"Aggregating time bucket: {time_bucket}")

        # Get rows with data in this bucket
        rows = self.get_rows_with_data(db, time_bucket)

        if not rows:
            logger.debug(f"No data for time bucket {time_bucket}")
            return 0

        logger.info(f"Found {len(rows)} rows with data")

        # Aggregate each row
        count = 0
        for row_id in rows:
            data = self.aggregate_row_risk(db, row_id, time_bucket)
            if data and self.create_or_update_summary(db, data):
                count += 1

        logger.info(f"Aggregation complete: {count} summaries created/updated")
        return count

    def aggregate_recent_buckets(self, db: Session, lookback_hours: int = 24) -> int:
        """
        Aggregate all time buckets in the lookback period.

        Args:
            db: Database session
            lookback_hours: How far back to aggregate

        Returns:
            Total number of summaries created/updated
        """
        now = datetime.now()
        start_time = now - timedelta(hours=lookback_hours)

        # Get all bucket start times in the period
        current_bucket = self.get_time_bucket(start_time)
        end_bucket = self.get_time_bucket(now)

        buckets = []
        while current_bucket <= end_bucket:
            buckets.append(current_bucket)
            current_bucket += timedelta(hours=self.interval_hours)

        logger.info(f"Aggregating {len(buckets)} time buckets (lookback={lookback_hours}h)")

        total_count = 0
        for bucket in buckets:
            count = self.aggregate_time_bucket(db, bucket)
            total_count += count

        return total_count

    def run(self, run_interval_minutes: int = 60) -> None:
        """
        Run the aggregation service continuously.

        Args:
            run_interval_minutes: Minutes between aggregation runs
        """
        logger.info(f"Starting risk aggregation service (run_interval={run_interval_minutes}m)")

        interval_seconds = run_interval_minutes * 60

        # Get database session
        db_gen = get_sync_db()
        db = next(db_gen)

        try:
            while True:
                logger.info("Starting aggregation cycle")

                try:
                    # Aggregate last 24 hours
                    count = self.aggregate_recent_buckets(db, lookback_hours=24)
                    logger.info(f"Aggregation cycle complete: {count} summaries processed")

                except Exception as e:
                    logger.error(f"Error in aggregation cycle: {e}")

                logger.info(f"Sleeping for {run_interval_minutes} minutes")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass
            logger.info("Risk aggregation service stopped")


def main():
    """Main entry point for risk aggregator service."""
    aggregator = RiskAggregator()
    aggregator.run()


if __name__ == "__main__":
    main()
