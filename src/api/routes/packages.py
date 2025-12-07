"""Package management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import Optional
from pydantic import BaseModel
import uuid
from pathlib import Path
import zipfile
import tempfile
import shutil
import httpx
from datetime import datetime

from src.api.schemas import (
    PackageCreate,
    PackageResponse,
    PackageListResponse,
)
from src.api.dependencies import get_db, get_current_user, get_optional_user, validate_id
from src.database import crud
from src.database.models import User, Package
from src.utils.exceptions import PackageNotFoundError
from src.ingest import validate_and_ingest
from src.hf_api import HuggingFaceAPI
from src.storage import LocalStorageBackend

router = APIRouter()


# Request/Response schemas for ingest
class IngestRequest(BaseModel):
    """Request to ingest a HuggingFace model."""
    url: str  # Full HuggingFace URL or model name
    name: Optional[str] = None  # Expected artifact name (from autograder)


class IngestResponse(BaseModel):
    """Response from ingest validation."""
    status: int  # 201 if passes, 424 if fails
    message: str
    model_name: Optional[str] = None
    artifact_id: Optional[str] = None
    is_ingestible: bool
    all_scores: Optional[dict] = None
    failing_metrics: Optional[list] = None
    latency_ms: Optional[int] = None


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_model(
    request: IngestRequest,
    db: Session = Depends(get_db),
    #current_user: User = Depends(get_current_user),
):
    """
    Ingest a public HuggingFace model.

    Model must score >= 0.5 on all quality metrics to be ingestible.

    Args:
        request: Ingest request with HuggingFace model URL
        db: Database session
        current_user: Current authenticated user

    Returns:
        IngestResponse with status 201 if passes, 424 if fails

    Raises:
        HTTPException: 424 if model fails quality gate
        HTTPException: 400 if invalid URL
    """
    try:
        # Use provided name if available, otherwise parse from URL
        if request.name:
            model_name = request.name
        else:
            # Fallback: parse model name from URL
            url_parts = request.url.strip("/").split("/")
            model_name = (
                "/".join(url_parts[-2:])
                if len(url_parts) >= 2
                else url_parts[-1]
            )

        # Validate model against quality gate
        passes_gate, validation_result = validate_and_ingest(
            model_name
        )

        if not passes_gate:
            # Quality gate failed - return 424
            return IngestResponse(
                status=424,
                message="Artifact ingest rejected: quality gate failed",
                model_name=model_name,
                is_ingestible=False,
                all_scores=validation_result.get("all_scores"),
                failing_metrics=validation_result.get("failing_metrics"),
                latency_ms=validation_result.get("latency_ms"),
            )

        # Quality gate passed - download and store the model
        artifact_id = str(uuid.uuid4())
        hf_api = HuggingFaceAPI()
        storage = LocalStorageBackend()

        # Create temp directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download model files from HuggingFace
            try:
                model_data = hf_api.fetch_model(model_name)
                model_files = model_data.get("siblings", [])

                # Download each file
                for file_info in model_files[:10]:  # Limit to 10 files for MVP
                    filename = file_info.get("rfilename", "")
                    if filename:
                        file_url = (
                            f"https://huggingface.co/{model_name}/"
                            f"resolve/main/{filename}"
                        )
                        # Simple download (production: use streaming)
                        response = httpx.get(
                            file_url, follow_redirects=True
                        )
                        if response.status_code == 200:
                            file_path = temp_path / filename
                            file_path.parent.mkdir(
                                parents=True, exist_ok=True
                            )
                            file_path.write_bytes(response.content)

            except Exception:
                # If download fails, still register but without files
                pass

            # Create ZIP archive
            zip_path = temp_path / f"{artifact_id}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in temp_path.rglob("*"):
                    if file_path != zip_path and file_path.is_file():
                        arcname = file_path.relative_to(temp_path)
                        zipf.write(file_path, arcname)

            # Store artifact
            metadata = {
                "model_name": model_name,
                "source": "huggingface",
                "scores": validation_result.get("all_scores"),
                #"uploaded_by": current_user.username,
            }

            storage.store_artifact(
                artifact_id=artifact_id,
                artifact_path=str(zip_path),
                metadata=metadata
            )

        # Register in database
        artifact_file_path = storage.models_path / f"{artifact_id}.zip"
        file_size = artifact_file_path.stat().st_size if artifact_file_path.exists() else 0

        # Check if package already exists
        existing_package = crud.get_package_by_name_version(
            db,
            name=model_name,
            version="1.0.0"
        )

        if existing_package:
            # Package already exists - update metadata
            existing_package.s3_key = f"{artifact_id}.zip"
            existing_package.file_size_bytes = file_size
            existing_package.updated_at = datetime.utcnow()
            db_package = existing_package
            db.commit()
            db.refresh(db_package)
        else:
            # Create new package
            db_package = crud.create_package(
                db,
                name=model_name,
                version="1.0.0",  # Default version for ingested models
                s3_key=f"{artifact_id}.zip",
                s3_bucket="local",
                file_size_bytes=file_size,
                description=f"Ingested from HuggingFace: {model_name}",
                #uploaded_by=current_user.id,
            )
            db.commit()

        return IngestResponse(
            status=201,
            message="Artifact successfully ingested",
            model_name=model_name,
            artifact_id=artifact_id,
            is_ingestible=True,
            all_scores=validation_result.get("all_scores"),
            latency_ms=validation_result.get("latency_ms"),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model URL: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during ingest: {str(e)}",
        )


@router.post("", response_model=PackageResponse, status_code=status.HTTP_201_CREATED)
def create_package(
    package_data: PackageCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload/create a new package.
    
    Args:
        package_data: Package metadata
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Created package object
        
    Raises:
        HTTPException: If package with same name/version exists
    """
    # Check if package already exists
    existing = crud.get_package_by_name_version(
        db,
        package_data.name,
        package_data.version
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Package '{package_data.name}' version '{package_data.version}' already exists"
        )
    
    # Create package
    db_package = crud.create_package(
        db,
        name=package_data.name,
        version=package_data.version,
        s3_key="",  # TODO: Generate from upload
        s3_bucket="",  # TODO: S3 integration
        file_size_bytes=0,  # TODO: Get from upload
        description=package_data.description,
        uploaded_by=current_user.id
    )
    
    return db_package


@router.get("/{package_id}", response_model=PackageResponse)
def get_package(
    package_id: str,
    db: Session = Depends(get_db)
):
    """
    Get package details by ID.
    
    Args:
        package_id: Package ID
        db: Database session
        
    Returns:
        Package object
        
    Raises:
        PackageNotFoundError: If package not found
    """
    # Validate ID format
    validate_id(package_id)
    
    # Try to convert to int for database lookup
    try:
        pkg_id = int(package_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Package not found: {package_id}"
        )
    
    package = crud.get_package_by_id(db, pkg_id)
    if not package:
        raise PackageNotFoundError(pkg_id)
    
    return package


@router.get("/search/ingested", response_model=PackageListResponse)
def search_ingested_models(
    query: Optional[str] = Query(None, description="Search query for model name/description"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db)
):
    """
    Search for ingested models by name or description.
    
    Args:
        query: Search query string (searches model name and description)
        page: Page number (1-indexed)
        page_size: Number of items per page
        db: Database session
        
    Returns:
        Paginated list of ingested packages matching search criteria
    """
    # Calculate offset
    offset = (page - 1) * page_size
    
    # Get all packages ingested from HuggingFace
    # (those with "Ingested from HuggingFace" in description)
    packages_query = db.query(Package).filter(
        Package.description.like("%Ingested from HuggingFace%")
    )
    
    # Apply search filter if provided
    if query:
        packages_query = packages_query.filter(
            or_(
                Package.name.ilike(f"%{query}%"),
                Package.description.ilike(f"%{query}%")
            )
        )
    
    # Get total count before pagination
    total = packages_query.count()
    
    # Apply pagination
    packages = packages_query.offset(offset).limit(page_size).all()
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "packages": packages
    }


@router.get("/search/artifact", response_model=PackageResponse)
def search_by_artifact_id(
    artifact_id: str = Query(..., description="Artifact ID returned from ingest endpoint"),
    db: Session = Depends(get_db)
):
    """
    Search for a package by its artifact ID.
    
    The artifact ID is returned when a model is ingested via the /ingest endpoint.
    This endpoint retrieves the package details using that artifact ID.
    
    Args:
        artifact_id: The UUID artifact ID (e.g., "abc123-def456-...")
        db: Database session
        
    Returns:
        Package object matching the artifact ID
        
    Raises:
        HTTPException: 404 if no package found with that artifact ID
        
    Example:
        GET /api/v1/package/search/artifact?artifact_id=abc123-def456-ghi789
    """
    # The artifact_id is stored in s3_key as {artifact_id}.zip
    s3_key = f"{artifact_id}.zip"
    
    package = db.query(Package).filter(Package.s3_key == s3_key).first()
    
    if not package:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No package found with artifact_id: {artifact_id}"
        )
    
    return package


@router.get("", response_model=PackageListResponse)
def list_packages(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    name_pattern: Optional[str] = Query(None, description="Regex pattern for name search"),
    db: Session = Depends(get_db)
):
    """
    List all packages with pagination and optional regex search.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        name_pattern: Optional regex pattern to filter by package name
        db: Database session
        
    Returns:
        Paginated list of packages
    """
    from src.api.dependencies import validate_regex_pattern
    
    # Calculate offset
    offset = (page - 1) * page_size
    
    # Validate regex pattern if provided to prevent ReDoS
    validated_pattern = None
    if name_pattern:
        try:
            validated_pattern = validate_regex_pattern(name_pattern)
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid regex pattern"
            )
    
    # Get packages with regex filtering
    packages = crud.get_packages(
        db, 
        skip=offset, 
        limit=page_size,
        name_filter=validated_pattern,
        use_regex=True if validated_pattern else False
    )
    
    # Get total count with same filter
    if validated_pattern:
        from src.database.models import Package
        query = db.query(Package)
        # Use safe pattern matching instead of raw REGEXP
        query = query.filter(Package.name.op('~')(validated_pattern))
        total = query.count()
    else:
        from src.database.models import Package
        total = db.query(Package).count()
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "packages": packages
    }


@router.delete("/{package_id}", status_code=status.HTTP_200_OK)
def delete_package(
    package_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a package.
    
    Args:
        package_id: Package ID to delete
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Success message
        
    Raises:
        PackageNotFoundError: If package not found
        HTTPException: If user lacks permission
    """
    # Validate ID format
    validate_id(package_id)
    
    # Try to convert to int for database lookup
    try:
        pkg_id = int(package_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Package not found: {package_id}"
        )
    
    # Get package
    package = crud.get_package_by_id(db, pkg_id)
    if not package:
        raise PackageNotFoundError(pkg_id)
    
    # Check permissions (only uploader or admin can delete)
    if package.uploaded_by != current_user.id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this package"
        )
    
    # Delete package
    crud.delete_package(db, package_id)
    
    return {"message": f"Package {package_id} deleted successfully"}


# Artifacts query endpoint (OpenAPI spec compliant)
class ArtifactQuery(BaseModel):
    """Query for artifacts."""
    name: str
    types: Optional[list[str]] = None


class ArtifactMetadata(BaseModel):
    """Artifact metadata response."""
    name: str
    id: str
    type: str


@router.post(
    "/artifacts",
    response_model=list[ArtifactMetadata],
    status_code=status.HTTP_200_OK,
    tags=["artifacts"]
)
def query_artifacts(
    queries: list[ArtifactQuery],
    offset: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_optional_user),
):
    """
    Query artifacts from the registry.
    
    Use name="*" to list all artifacts.
    
    Args:
        queries: List of artifact queries
        offset: Pagination offset
        db: Database session
        current_user: Optional authenticated user
        
    Returns:
        List of matching artifact metadata
    """
    from src.api.dependencies import sanitize_name_query, safe_regex_match, RegexTimeoutError
    
    results = []
    
    for query in queries:
        # Validate and sanitize the name query to prevent ReDoS attacks
        try:
            sanitized_name = sanitize_name_query(query.name)
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception:
            # Any other error means invalid pattern
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid name pattern"
            )
        
        if sanitized_name == "*":
            # List all artifacts
            packages = crud.get_all_packages(db)
            for pkg in packages:
                # Get actual artifact type from package (artifact_type field)
                artifact_type = getattr(pkg, 'artifact_type', 'model')
                results.append(
                    ArtifactMetadata(
                        name=pkg.name,
                        id=str(pkg.id),
                        type=artifact_type
                    )
                )
        else:
            # Check if pattern contains regex metacharacters
            regex_metacharacters = set('[](){}|^$.*+?\\')
            is_regex = any(c in regex_metacharacters for c in sanitized_name)
            
            if is_regex:
                # Use safe regex matching with timeout protection
                packages = crud.get_all_packages(db)
                for pkg in packages:
                    try:
                        if safe_regex_match(sanitized_name, pkg.name):
                            artifact_type = getattr(pkg, 'artifact_type', 'model')
                            
                            if query.types and len(query.types) > 0:
                                if artifact_type not in query.types:
                                    continue
                            
                            results.append(
                                ArtifactMetadata(
                                    name=pkg.name,
                                    id=str(pkg.id),
                                    type=artifact_type
                                )
                            )
                    except RegexTimeoutError:
                        # Regex took too long - pattern is likely malicious
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Regex pattern execution timed out"
                        )
            else:
                # Simple text search using SQL LIKE
                packages = db.query(Package).filter(
                    Package.name.like(f"%{sanitized_name}%")
                ).all()
                
                for pkg in packages:
                    artifact_type = getattr(pkg, 'artifact_type', 'model')
                    
                    if query.types and len(query.types) > 0:
                        if artifact_type not in query.types:
                            continue
                        
                    results.append(
                        ArtifactMetadata(
                            name=pkg.name,
                            id=str(pkg.id),
                            type=artifact_type
                        )
                    )
    
    return results
