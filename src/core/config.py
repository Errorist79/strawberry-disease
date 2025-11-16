"""
Configuration management for the strawberry disease detection system.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "strawberry_disease"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # Model
    yolo_model_path: Path = Field(default=Path("models/weights/best.pt"))
    yolo_confidence_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45

    # Camera
    camera_capture_interval_minutes: int = 10
    image_storage_path: Path = Field(default=Path("data/images"))

    # Risk Calculation
    risk_aggregation_interval_hours: int = 1
    high_risk_threshold: int = 70
    medium_risk_threshold: int = 40

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_token: str = "change_me_in_production"
    api_title: str = "Strawberry Disease Detection API"
    api_version: str = "0.1.0"

    # Telegram
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    telegram_alert_threshold: int = 70
    telegram_notification_cooldown_minutes: int = 60

    # Grafana
    grafana_port: int = 3000
    grafana_admin_user: str = "admin"
    grafana_admin_password: str = "admin"

    # Logging
    log_level: str = "INFO"
    log_file_path: Path = Field(default=Path("logs/app.log"))

    # Environment
    environment: str = "development"

    @field_validator("yolo_model_path", "image_storage_path", "log_file_path", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @property
    def database_url(self) -> str:
        """Construct the database URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment.lower() == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    return Settings()
