"""System management endpoints (health, reset, etc.)."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import time

from src.api.schemas import HealthResponse, ResetResponse
from src.api.dependencies import get_db
from src.database.connection import engine
from src.database import models
from src.database import crud
from src.auth.password_hash import hash_password
from src.api.config import settings

router = APIRouter()

# Track startup time for uptime calculation
_startup_time = time.time()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    
    Returns system health status and basic operational metrics.
    Database connectivity is optional - endpoint always returns 200 OK.
    
    Returns:
        Health status information
    """
    # Check database connectivity (non-blocking)
    db_status = "not_checked"
    try:
        from src.api.dependencies import get_db
        db = next(get_db())
        db.execute("SELECT 1")
        db_status = "healthy"
        db.close()
    except Exception as e:
        db_status = f"unavailable: {str(e)[:50]}"
    
    uptime = time.time() - _startup_time
    
    return {
        "status": "ok",  # Always OK - app is running
        "timestamp": datetime.utcnow(),
        "database_status": db_status,
        "uptime_seconds": uptime
    }


@router.post("/reset", response_model=ResetResponse)
def reset_system(db: Session = Depends(get_db)):
    """
    Reset system to default state.
    
    This endpoint:
    1. Drops all tables
    2. Recreates all tables
    3. Creates the default admin user
    
    WARNING: This destroys all data!
    
    Args:
        db: Database session
        
    Returns:
        Reset confirmation message
    """
    # Drop all tables
    models.Base.metadata.drop_all(bind=engine)
    
    # Recreate all tables
    models.Base.metadata.create_all(bind=engine)
    
    # Create default admin user as per spec
    hashed_password = hash_password(settings.DEFAULT_ADMIN_PASSWORD)
    
    default_admin = crud.create_user(
        db,
        username=settings.DEFAULT_ADMIN_USERNAME,
        email=settings.DEFAULT_ADMIN_EMAIL,
        hashed_password=hashed_password,
        is_admin=True
    )
    
    return {
        "message": "System reset successfully. All data has been cleared and default admin user created.",
        "default_admin_created": True
    }
