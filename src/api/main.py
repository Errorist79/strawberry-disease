"""
FastAPI application main module.

Provides REST API endpoints for accessing disease detection data,
risk summaries, and system status.
"""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.dependencies import get_async_db_session, verify_token
from src.api.schemas import (
    CameraResponse,
    DashboardOverview,
    HealthResponse,
    ImageDetailResponse,
    ImageResponse,
    RowRiskStatus,
    RowTrendPoint,
    RowTrendResponse,
    RiskSummaryResponse,
)
from src.core.config import get_settings
from src.core.ml import RiskCalculator
from src.models.camera import Camera
from src.models.image import Image, ImageStatus
from src.models.risk_summary import RiskSummary

settings = get_settings()
risk_calculator = RiskCalculator()

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="API for strawberry disease detection and risk monitoring system",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint (no auth required)
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: AsyncSession = Depends(get_async_db_session)):
    """
    Health check endpoint.

    Returns system status and database connectivity.
    """
    try:
        # Test database connection
        await db.execute(select(1))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        timestamp=datetime.now(),
        database=db_status,
        version=settings.api_version,
    )


# Camera endpoints
@app.get(
    "/api/cameras",
    response_model=List[CameraResponse],
    dependencies=[Depends(verify_token)],
    tags=["Cameras"],
)
async def list_cameras(
    active_only: bool = Query(False, description="Return only active cameras"),
    db: AsyncSession = Depends(get_async_db_session),
):
    """List all cameras in the system."""
    stmt = select(Camera)
    if active_only:
        stmt = stmt.where(Camera.is_active == True)

    result = await db.execute(stmt.order_by(Camera.name))
    cameras = result.scalars().all()
    return cameras


@app.get(
    "/api/cameras/{camera_id}",
    response_model=CameraResponse,
    dependencies=[Depends(verify_token)],
    tags=["Cameras"],
)
async def get_camera(camera_id: int, db: AsyncSession = Depends(get_async_db_session)):
    """Get details of a specific camera."""
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()

    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    return camera


# Image endpoints
@app.get(
    "/api/images",
    response_model=List[ImageResponse],
    dependencies=[Depends(verify_token)],
    tags=["Images"],
)
async def list_images(
    camera_id: Optional[int] = Query(None, description="Filter by camera ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    db: AsyncSession = Depends(get_async_db_session),
):
    """List images with optional filtering."""
    stmt = select(Image)

    if camera_id:
        stmt = stmt.where(Image.camera_id == camera_id)
    if status:
        stmt = stmt.where(Image.status == status)

    stmt = stmt.order_by(Image.captured_at.desc()).limit(limit).offset(offset)

    result = await db.execute(stmt)
    images = result.scalars().all()
    return images


@app.get(
    "/api/images/{image_id}",
    response_model=ImageDetailResponse,
    dependencies=[Depends(verify_token)],
    tags=["Images"],
)
async def get_image(image_id: int, db: AsyncSession = Depends(get_async_db_session)):
    """Get detailed information about a specific image including predictions."""
    stmt = (
        select(Image)
        .where(Image.id == image_id)
        .options(selectinload(Image.predictions), selectinload(Image.camera))
    )

    result = await db.execute(stmt)
    image = result.scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    return image


# Risk summary endpoints
@app.get(
    "/api/risk-summaries",
    response_model=List[RiskSummaryResponse],
    dependencies=[Depends(verify_token)],
    tags=["Risk"],
)
async def list_risk_summaries(
    row_id: Optional[str] = Query(None, description="Filter by row ID"),
    start_time: Optional[datetime] = Query(None, description="Start of time range"),
    end_time: Optional[datetime] = Query(None, description="End of time range"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_async_db_session),
):
    """List risk summaries with optional filtering."""
    stmt = select(RiskSummary)

    if row_id:
        stmt = stmt.where(RiskSummary.row_id == row_id)
    if start_time:
        stmt = stmt.where(RiskSummary.time_bucket >= start_time)
    if end_time:
        stmt = stmt.where(RiskSummary.time_bucket <= end_time)

    stmt = stmt.order_by(RiskSummary.time_bucket.desc()).limit(limit).offset(offset)

    result = await db.execute(stmt)
    summaries = result.scalars().all()
    return summaries


@app.get(
    "/api/rows/{row_id}/current-risk",
    response_model=RowRiskStatus,
    dependencies=[Depends(verify_token)],
    tags=["Risk"],
)
async def get_row_current_risk(row_id: str, db: AsyncSession = Depends(get_async_db_session)):
    """Get the most recent risk status for a specific row."""
    stmt = (
        select(RiskSummary)
        .where(RiskSummary.row_id == row_id)
        .order_by(RiskSummary.time_bucket.desc())
        .limit(1)
    )

    result = await db.execute(stmt)
    summary = result.scalar_one_or_none()

    if not summary:
        raise HTTPException(status_code=404, detail=f"No risk data found for row {row_id}")

    return RowRiskStatus(
        row_id=summary.row_id,
        latest_time_bucket=summary.time_bucket,
        avg_risk_score=summary.avg_risk_score,
        max_risk_score=summary.max_risk_score,
        sample_count=summary.sample_count,
        dominant_disease=summary.dominant_disease,
        risk_category=risk_calculator.get_risk_category(summary.max_risk_score),
    )


@app.get(
    "/api/rows/{row_id}/trend",
    response_model=RowTrendResponse,
    dependencies=[Depends(verify_token)],
    tags=["Risk"],
)
async def get_row_trend(
    row_id: str,
    hours: int = Query(24, ge=1, le=168, description="Number of hours to look back"),
    db: AsyncSession = Depends(get_async_db_session),
):
    """Get risk trend for a specific row over time."""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)

    stmt = (
        select(RiskSummary)
        .where(RiskSummary.row_id == row_id)
        .where(RiskSummary.time_bucket >= start_time)
        .where(RiskSummary.time_bucket <= end_time)
        .order_by(RiskSummary.time_bucket)
    )

    result = await db.execute(stmt)
    summaries = result.scalars().all()

    data_points = [
        RowTrendPoint(
            time_bucket=s.time_bucket,
            avg_risk_score=s.avg_risk_score,
            max_risk_score=s.max_risk_score,
            sample_count=s.sample_count,
        )
        for s in summaries
    ]

    return RowTrendResponse(
        row_id=row_id, start_time=start_time, end_time=end_time, data_points=data_points
    )


@app.get(
    "/api/rows/high-risk",
    response_model=List[RowRiskStatus],
    dependencies=[Depends(verify_token)],
    tags=["Risk"],
)
async def get_high_risk_rows(
    threshold: int = Query(70, ge=0, le=100, description="Risk threshold"),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_async_db_session),
):
    """Get rows with highest current risk."""
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
        .where(RiskSummary.max_risk_score >= threshold)
        .order_by(RiskSummary.max_risk_score.desc())
        .limit(limit)
    )

    result = await db.execute(stmt)
    summaries = result.scalars().all()

    return [
        RowRiskStatus(
            row_id=s.row_id,
            latest_time_bucket=s.time_bucket,
            avg_risk_score=s.avg_risk_score,
            max_risk_score=s.max_risk_score,
            sample_count=s.sample_count,
            dominant_disease=s.dominant_disease,
            risk_category=risk_calculator.get_risk_category(s.max_risk_score),
        )
        for s in summaries
    ]


# Dashboard endpoint
@app.get(
    "/api/dashboard/overview",
    response_model=DashboardOverview,
    dependencies=[Depends(verify_token)],
    tags=["Dashboard"],
)
async def get_dashboard_overview(db: AsyncSession = Depends(get_async_db_session)):
    """Get overview statistics for the dashboard."""
    # Count cameras
    total_cameras_result = await db.execute(select(func.count(Camera.id)))
    total_cameras = total_cameras_result.scalar()

    active_cameras_result = await db.execute(
        select(func.count(Camera.id)).where(Camera.is_active == True)
    )
    active_cameras = active_cameras_result.scalar()

    # Count images by status
    total_images_result = await db.execute(select(func.count(Image.id)))
    total_images = total_images_result.scalar()

    pending_images_result = await db.execute(
        select(func.count(Image.id)).where(Image.status == ImageStatus.PENDING.value)
    )
    pending_images = pending_images_result.scalar()

    completed_images_result = await db.execute(
        select(func.count(Image.id)).where(Image.status == ImageStatus.COMPLETED.value)
    )
    completed_images = completed_images_result.scalar()

    failed_images_result = await db.execute(
        select(func.count(Image.id)).where(Image.status == ImageStatus.FAILED.value)
    )
    failed_images = failed_images_result.scalar()

    # Get highest risk row
    high_risk_rows = await get_high_risk_rows(threshold=settings.high_risk_threshold, limit=1, db=db)
    highest_risk_row = high_risk_rows[0] if high_risk_rows else None

    # Count rows at risk
    rows_at_risk_result = await db.execute(
        select(func.count(func.distinct(RiskSummary.row_id))).where(
            RiskSummary.max_risk_score >= settings.high_risk_threshold
        )
    )
    rows_at_risk = rows_at_risk_result.scalar()

    return DashboardOverview(
        total_cameras=total_cameras,
        active_cameras=active_cameras,
        total_images=total_images,
        pending_images=pending_images,
        completed_images=completed_images,
        failed_images=failed_images,
        rows_at_risk=rows_at_risk,
        highest_risk_row=highest_risk_row,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
