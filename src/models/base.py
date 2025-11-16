"""
Base model and database configuration.
"""

from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import DateTime, create_engine, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.core.config import get_settings


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# Synchronous engine and session
def get_engine():
    """Get synchronous database engine."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        echo=settings.is_development,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


def get_session_maker():
    """Get synchronous session maker."""
    engine = get_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


# Asynchronous engine and session
def get_async_engine():
    """Get asynchronous database engine."""
    settings = get_settings()
    # Replace postgresql:// with postgresql+asyncpg://
    async_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
    return create_async_engine(
        async_url,
        echo=settings.is_development,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
    )


def get_async_session_maker():
    """Get asynchronous session maker."""
    engine = get_async_engine()
    return async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting async database sessions.

    Yields:
        AsyncSession: Database session
    """
    async_session = get_async_session_maker()
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_db():
    """
    Get synchronous database session for non-async code.

    Yields:
        Session: Database session
    """
    SessionLocal = get_session_maker()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
