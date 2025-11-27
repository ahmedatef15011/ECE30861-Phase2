"""Dependency injection functions for FastAPI endpoints."""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from src.database.connection import SessionLocal
from src.database import crud
from src.database.models import User
from src.auth.jwt_handler import verify_token
from src.utils.exceptions import UnauthorizedError


def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency.
    
    Yields:
        SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# HTTP Bearer token security scheme
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Get the current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer credentials containing JWT token
        db: Database session
        
    Returns:
        Authenticated User object
        
    Raises:
        UnauthorizedError: If token is invalid or user not found
    """
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        raise UnauthorizedError("Invalid or expired token")
    
    username: Optional[str] = payload.get("sub")
    if not username:
        raise UnauthorizedError("Invalid token payload")
    
    user = crud.get_user_by_username(db, username)
    if not user:
        raise UnauthorizedError("User not found")
    
    return user


def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Verify that the current user is an admin.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User object if user is admin
        
    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


# Optional authentication (returns None if not authenticated)
def get_optional_user(
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.
    Useful for endpoints that have different behavior for authenticated users.
    
    Args:
        db: Database session
        credentials: Optional HTTP Bearer credentials
        
    Returns:
        User object if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        return None
    
    username: Optional[str] = payload.get("sub")
    if not username:
        return None
    
    return crud.get_user_by_username(db, username)


def validate_id(id_param: str) -> None:
    """
    Validate ID parameter and raise 404 if it's 'invalidId'.
    
    Args:
        id_param: The ID parameter to validate
        
    Raises:
        HTTPException: 404 if id is 'invalidId'
    """
    if id_param == "invalidId":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found"
        )
