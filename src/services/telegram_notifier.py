"""
Telegram notification service.

This service monitors risk summaries and sends alerts via Telegram
when risk levels exceed configured thresholds.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from telegram import Bot
from telegram.error import TelegramError

from src.core.config import get_settings
from src.core.logging import setup_logging
from src.core.ml import RiskCalculator
from src.models.base import get_sync_db
from src.models.risk_summary import RiskSummary

logger = setup_logging("telegram_notifier")


class TelegramNotifier:
    """
    Service for sending Telegram notifications about high risk conditions.
    """

    def __init__(self):
        """Initialize Telegram notifier."""
        self.settings = get_settings()

        # Validate Telegram configuration
        if not self.settings.telegram_bot_token:
            logger.warning("TELEGRAM_BOT_TOKEN not configured, notifications disabled")
            self.enabled = False
        elif not self.settings.telegram_chat_id:
            logger.warning("TELEGRAM_CHAT_ID not configured, notifications disabled")
            self.enabled = False
        else:
            self.enabled = True

        # Initialize bot if enabled
        self.bot: Optional[Bot] = None
        if self.enabled:
            self.bot = Bot(token=self.settings.telegram_bot_token)

        # Notification cooldown tracking
        self.last_notification: Dict[str, datetime] = {}

        self.risk_calculator = RiskCalculator()

        logger.info(f"TelegramNotifier initialized (enabled={self.enabled})")

    def can_send_notification(self, row_id: str) -> bool:
        """
        Check if we can send notification for a row (cooldown check).

        Args:
            row_id: Row identifier

        Returns:
            True if notification can be sent
        """
        if row_id not in self.last_notification:
            return True

        cooldown_minutes = self.settings.telegram_notification_cooldown_minutes
        last_time = self.last_notification[row_id]
        time_since_last = datetime.now() - last_time

        return time_since_last >= timedelta(minutes=cooldown_minutes)

    def format_alert_message(self, summary: RiskSummary) -> str:
        """
        Format alert message for Telegram.

        Args:
            summary: Risk summary object

        Returns:
            Formatted message string
        """
        risk_category = self.risk_calculator.get_risk_category(summary.max_risk_score)

        # Emoji based on risk level
        emoji_map = {
            "low": "🟢",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴",
        }
        emoji = emoji_map.get(risk_category, "⚠️")

        # Risk bar
        risk_bar_length = 10
        filled = int((summary.max_risk_score / 100) * risk_bar_length)
        risk_bar = "█" * filled + "░" * (risk_bar_length - filled)

        message = f"""
{emoji} **HASTALLIK RİSK UYARISI** {emoji}

**Sıra:** {summary.row_id}
**Risk Seviyesi:** {risk_category.upper()}

**Metrikler:**
• Ortalama Risk: {summary.avg_risk_score:.1f}/100
• Maksimum Risk: {summary.max_risk_score}/100
• Risk Barı: {risk_bar} ({summary.max_risk_score}%)

**Hastalık Bilgisi:**
• Baskın Hastalık: {summary.dominant_disease.replace('_', ' ').title()}
• Örnek Sayısı: {summary.sample_count}

**Zaman:** {summary.time_bucket.strftime('%d.%m.%Y %H:%M')}

⚠️ Lütfen bu sıradaki bitkileri kontrol edin!
""".strip()

        return message

    async def send_alert(self, summary: RiskSummary) -> bool:
        """
        Send alert message via Telegram.

        Args:
            summary: Risk summary to alert about

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug("Telegram notifications disabled, skipping")
            return False

        # Check cooldown
        if not self.can_send_notification(summary.row_id):
            logger.debug(f"Cooldown active for {summary.row_id}, skipping notification")
            return False

        try:
            message = self.format_alert_message(summary)

            await self.bot.send_message(
                chat_id=self.settings.telegram_chat_id,
                text=message,
                parse_mode="Markdown",
            )

            # Update last notification time
            self.last_notification[summary.row_id] = datetime.now()

            logger.info(f"Sent Telegram alert for row {summary.row_id}")
            return True

        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram alert: {e}")
            return False

    def check_and_notify(self, db: Session) -> int:
        """
        Check for high-risk conditions and send notifications.

        Args:
            db: Database session

        Returns:
            Number of notifications sent
        """
        if not self.enabled:
            return 0

        # Get latest risk summary for each row
        subquery = (
            select(
                RiskSummary.row_id,
                func.max(RiskSummary.time_bucket).label("max_time"),
            )
            .group_by(RiskSummary.row_id)
            .subquery()
        )

        stmt = (
            select(RiskSummary)
            .join(
                subquery,
                (RiskSummary.row_id == subquery.c.row_id)
                & (RiskSummary.time_bucket == subquery.c.max_time),
            )
            .where(RiskSummary.max_risk_score >= self.settings.telegram_alert_threshold)
        )

        result = db.execute(stmt)
        high_risk_summaries = result.scalars().all()

        if not high_risk_summaries:
            logger.debug("No high-risk rows found")
            return 0

        logger.info(f"Found {len(high_risk_summaries)} rows above risk threshold")

        # Send alerts
        sent_count = 0
        for summary in high_risk_summaries:
            # Run async send in event loop
            try:
                result = asyncio.run(self.send_alert(summary))
                if result:
                    sent_count += 1
            except Exception as e:
                logger.error(f"Error in async alert send: {e}")

        return sent_count

    def run(self, check_interval_minutes: int = 15) -> None:
        """
        Run the notification service continuously.

        Args:
            check_interval_minutes: Minutes between risk checks
        """
        logger.info(f"Starting Telegram notification service (interval={check_interval_minutes}m)")

        if not self.enabled:
            logger.warning("Telegram notifications not configured, service will not send alerts")

        interval_seconds = check_interval_minutes * 60

        # Get database session
        db_gen = get_sync_db()
        db = next(db_gen)

        try:
            while True:
                logger.info("Checking for high-risk conditions")

                try:
                    count = self.check_and_notify(db)
                    if count > 0:
                        logger.info(f"Sent {count} notifications")
                    else:
                        logger.debug("No notifications sent")

                except Exception as e:
                    logger.error(f"Error in notification check: {e}")

                logger.info(f"Sleeping for {check_interval_minutes} minutes")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass
            logger.info("Telegram notification service stopped")


def main():
    """Main entry point for Telegram notifier service."""
    notifier = TelegramNotifier()
    notifier.run()


if __name__ == "__main__":
    main()
