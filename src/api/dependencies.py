"""
API dependencies for authentication and database sessions.
"""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.models.base import get_db

security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """
    Verify API token.

    Args:
        credentials: HTTP credentials

    Returns:
        Token string

    Raises:
        HTTPException: If token is invalid
    """
    settings = get_settings()

    if credentials.credentials != settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials


async def get_async_db_session() -> AsyncSession:
    """Dependency to get async database session."""
    async for session in get_db():
        yield session
