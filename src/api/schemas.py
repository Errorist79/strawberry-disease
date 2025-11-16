"""
Pydantic schemas for API request/response models.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# Camera schemas
class CameraBase(BaseModel):
    """Base camera schema."""

    name: str
    location: str
    row_id: str
    stream_url: Optional[str] = None
    is_active: bool = True
    description: Optional[str] = None


class CameraResponse(CameraBase):
    """Camera response schema."""

    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Image schemas
class ImageBase(BaseModel):
    """Base image schema."""

    camera_id: int
    file_path: str
    captured_at: datetime


class ImageResponse(ImageBase):
    """Image response schema."""

    id: int
    status: str
    max_risk_score: Optional[int] = None
    dominant_disease: Optional[str] = None
    prediction_count: int
    error_message: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Prediction schemas
class PredictionResponse(BaseModel):
    """Prediction response schema."""

    id: int
    image_id: int
    class_label: str
    confidence: float
    bbox_x: Optional[float] = None
    bbox_y: Optional[float] = None
    bbox_width: Optional[float] = None
    bbox_height: Optional[float] = None
    risk_score: int
    created_at: datetime

    class Config:
        from_attributes = True


class ImageDetailResponse(ImageResponse):
    """Detailed image response with predictions."""

    predictions: List[PredictionResponse] = []
    camera: CameraResponse


# Risk summary schemas
class RiskSummaryResponse(BaseModel):
    """Risk summary response schema."""

    id: int
    row_id: str
    time_bucket: datetime
    avg_risk_score: float
    max_risk_score: int
    min_risk_score: int
    sample_count: int
    dominant_disease: str
    created_at: datetime

    class Config:
        from_attributes = True


# Dashboard schemas
class RowRiskStatus(BaseModel):
    """Current risk status for a row."""

    row_id: str
    latest_time_bucket: datetime
    avg_risk_score: float
    max_risk_score: int
    sample_count: int
    dominant_disease: str
    risk_category: str  # low, medium, high, critical


class DashboardOverview(BaseModel):
    """Dashboard overview response."""

    total_cameras: int
    active_cameras: int
    total_images: int
    pending_images: int
    completed_images: int
    failed_images: int
    rows_at_risk: int  # Number of rows above threshold
    highest_risk_row: Optional[RowRiskStatus] = None


class RowTrendPoint(BaseModel):
    """Single point in a risk trend."""

    time_bucket: datetime
    avg_risk_score: float
    max_risk_score: int
    sample_count: int


class RowTrendResponse(BaseModel):
    """Trend data for a specific row."""

    row_id: str
    start_time: datetime
    end_time: datetime
    data_points: List[RowTrendPoint]


# Health check
class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    database: str
    version: str
