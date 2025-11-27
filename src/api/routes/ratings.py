"""Package rating/scoring endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.schemas import PackageScoreCreate, PackageScoreResponse
from src.api.dependencies import get_db, get_current_user, validate_id
from src.database import crud
from src.database.models import User
from src.utils.exceptions import PackageNotFoundError

router = APIRouter()


@router.post("/{package_id}/rate", response_model=PackageScoreResponse, status_code=status.HTTP_201_CREATED)
def rate_package(
    package_id: str,
    scores: PackageScoreCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Rate a package with quality/trust scores.
    
    Args:
        package_id: Package ID to rate
        scores: Score data including all metrics
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Created package score object
        
    Raises:
        PackageNotFoundError: If package not found
    """
    # Validate for invalidId
    validate_id(package_id)
    
    # Convert to int
    try:
        pkg_id = int(package_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid package ID format"
        )
    
    # Verify package exists
    package = crud.get_package_by_id(db, pkg_id)
    if not package:
        raise PackageNotFoundError(pkg_id)
    
    # Create or update scores
    db_score = crud.create_or_update_package_score(
        db,
        package_id=pkg_id,
        **scores.model_dump()
    )
    
    return db_score


@router.get("/{package_id}/scores", response_model=PackageScoreResponse)
def get_package_scores(
    package_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the scores for a package.
    
    Args:
        package_id: Package ID
        db: Database session
        
    Returns:
        Package score object
        
    Raises:
        PackageNotFoundError: If package not found
        HTTPException: If package has no scores
    """
    # Validate for invalidId
    validate_id(package_id)
    
    # Convert to int
    try:
        pkg_id = int(package_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid package ID format"
        )
    
    # Verify package exists
    package = crud.get_package_by_id(db, pkg_id)
    if not package:
        raise PackageNotFoundError(pkg_id)
    
    # Get scores (TODO: implement get_package_scores in CRUD)
    # For now, return a placeholder response
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Score retrieval not yet implemented"
    )
