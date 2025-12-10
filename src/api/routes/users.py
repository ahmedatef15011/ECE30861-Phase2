"""User management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta

from src.api.schemas import (
    UserRegister,
    UserLogin,
    UserResponse,
    TokenResponse,
)
from src.api.dependencies import get_db, get_current_user, get_current_admin_user
from src.database import crud
from src.database.models import User
from src.auth.password_hash import hash_password, verify_password
from src.auth.jwt_handler import create_access_token
from src.utils.exceptions import InvalidCredentialsError
from src.api.config import settings

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(
    user_data: UserRegister,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_admin_user)  # Only admins can register new users
):
    """
    Register a new user (admin only).
    
    Args:
        user_data: User registration data
        db: Database session
        
    Returns:
        Created user object
        
    Raises:
        HTTPException: If username already exists
    """
    # Check if user already exists
    existing_user = crud.get_user_by_username(db, user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Hash password and create user
    hashed_password = hash_password(user_data.password)
    db_user = crud.create_user(
        db,
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        is_admin=user_data.is_admin
    )
    
    return db_user


@router.post("/login", response_model=TokenResponse)
def login(
    credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return JWT token.
    
    Args:
        credentials: Login credentials (username and password)
        db: Database session
        
    Returns:
        Access token and user information
        
    Raises:
        InvalidCredentialsError: If credentials are invalid
    """
    # Get user from database
    user = crud.get_user_by_username(db, credentials.username)
    
    # Verify user exists and password is correct
    if not user or not verify_password(
        credentials.password, user.hashed_password
    ):
        raise InvalidCredentialsError()
    
    # Create access token
    expires_delta = timedelta(hours=10)  # 10 hours as per spec
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=expires_delta
    )
    
    # Save token to database for usage tracking
    from datetime import datetime
    expires_at = datetime.utcnow() + expires_delta
    crud.create_auth_token(
        db=db,
        user_id=user.id,
        token=access_token,
        expires_at=expires_at
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current authenticated user information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information
    """
    return current_user


@router.delete("/{username}", status_code=status.HTTP_200_OK)
def delete_user(
    username: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a user account.
    
    Regular users can only delete their own account.
    Admins can delete any account.
    
    Args:
        username: Username to delete
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If user not found or insufficient permissions
    """
    # Check permissions
    if not current_user.is_admin and current_user.username != username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this user"
        )
    
    # Get user to delete
    user = crud.get_user_by_username(db, username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not found"
        )
    
    # Delete user
    crud.delete_user(db, user.id)
    
    return {"message": f"User '{username}' deleted successfully"}
