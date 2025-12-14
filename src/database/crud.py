"""Database utility functions and CRUD operations."""

from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from .models import (
    User,
    Permission,
    AuthToken,
    Package,
    PackageScore,
    PackageLineage,
    DownloadHistory,
    SystemHealthMetric,
)


# ============================================================================
# User CRUD Operations
# ============================================================================

def create_user(
    db: Session,
    username: str,
    email: str,
    hashed_password: str,
    is_admin: bool = False
) -> User:
    """Create a new user."""
    user = User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        is_admin=is_admin
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username."""
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID."""
    return db.query(User).filter(User.id == user_id).first()


def delete_user(db: Session, user_id: int) -> bool:
    """Delete a user by ID."""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        db.delete(user)
        db.commit()
        return True
    return False


# ============================================================================
# Permission CRUD Operations
# ============================================================================

def create_permission(
    db: Session,
    user_id: int,
    can_upload: bool = False,
    can_search: bool = True,
    can_download: bool = True,
    can_rate: bool = False,
    can_delete: bool = False,
    max_uploads_per_day: Optional[int] = 10,
    max_downloads_per_day: Optional[int] = 100
) -> Permission:
    """Create permissions for a user."""
    permission = Permission(
        user_id=user_id,
        can_upload=can_upload,
        can_search=can_search,
        can_download=can_download,
        can_rate=can_rate,
        can_delete=can_delete,
        max_uploads_per_day=max_uploads_per_day,
        max_downloads_per_day=max_downloads_per_day
    )
    db.add(permission)
    db.commit()
    db.refresh(permission)
    return permission


def get_user_permissions(db: Session, user_id: int) -> Optional[Permission]:
    """Get permissions for a user."""
    return db.query(Permission).filter(Permission.user_id == user_id).first()


# ============================================================================
# Auth Token CRUD Operations
# ============================================================================

def create_auth_token(
    db: Session,
    user_id: int,
    token: str,
    expires_at: datetime
) -> AuthToken:
    """Create an authentication token."""
    auth_token = AuthToken(
        user_id=user_id,
        token=token,
        expires_at=expires_at
    )
    db.add(auth_token)
    db.commit()
    db.refresh(auth_token)
    return auth_token


def get_auth_token(db: Session, token: str) -> Optional[AuthToken]:
    """Get an authentication token if valid and under usage limit."""
    return db.query(AuthToken).filter(
        and_(
            AuthToken.token == token,
            AuthToken.is_revoked == False,
            AuthToken.expires_at > datetime.utcnow(),
            AuthToken.usage_count < 1000  # Enforce 1000 interaction limit
        )
    ).first()


def revoke_token(db: Session, token: str) -> bool:
    """Revoke an authentication token."""
    auth_token = db.query(AuthToken).filter(AuthToken.token == token).first()
    if auth_token:
        auth_token.is_revoked = True
        db.commit()
        return True
    return False


def increment_token_usage(db: Session, token: str) -> bool:
    """Increment the usage count for an authentication token.
    
    Args:
        db: Database session
        token: The JWT token string
        
    Returns:
        True if increment successful, False if token not found or limit exceeded
    """
    auth_token = db.query(AuthToken).filter(
        and_(
            AuthToken.token == token,
            AuthToken.is_revoked == False,
            AuthToken.expires_at > datetime.utcnow()
        )
    ).first()
    
    if auth_token and auth_token.usage_count < 1000:
        auth_token.usage_count += 1
        db.commit()
        return True
    return False


def update_time_used(db: Session, token: str) -> bool:
    """Update the last time used for an authentication token.
    
    Enforces a 0.5 second cooldown between token uses to prevent
    rapid-fire requests and reduce DoS risk.

    Args:
        db: Database Session
        token: the JWT token string

    Returns:
        True if time check passed and update successful, 
        False if token not found, invalid, or used too recently (< 0.5s)
    """
    auth_token = db.query(AuthToken).filter(
        and_(
            AuthToken.token == token,
            AuthToken.is_revoked == False,
            AuthToken.expires_at > datetime.utcnow()
        )
    ).first()

    if not auth_token:
        return False
    
    # Check if token has been used before
    if auth_token.last_used_at:
        # Calculate time since last use
        time_since_last_use = datetime.utcnow() - auth_token.last_used_at
        
        # Enforce 0.5 second cooldown
        if time_since_last_use.total_seconds() < 0.5:
            return False
    
    # Update last used time
    auth_token.last_used_at = datetime.utcnow()
    db.commit()
    return True


# ============================================================================
# Package CRUD Operations
# ============================================================================

def create_package(
    db: Session,
    name: str,
    version: str,
    s3_key: str,
    s3_bucket: str,
    file_size_bytes: int,
    artifact_type: str = "model",
    description: Optional[str] = None,
    author: Optional[str] = None,
    license: Optional[str] = None,
    readme_content: Optional[str] = None,
    source_url: Optional[str] = None,
    repository_url: Optional[str] = None,
    uploaded_by: Optional[int] = None,
    is_sensitive: bool = False,
    access_control_script: Optional[str] = None,
    ingest_status: str = "approved",  # New: "approved", "rejected", "pending"
    quality_gate_result: Optional[dict] = None  # New: JSON with eval results
) -> Package:
    """Create a new package."""
    package = Package(
        name=name,
        version=version,
        artifact_type=artifact_type,
        s3_key=s3_key,
        s3_bucket=s3_bucket,
        file_size_bytes=file_size_bytes,
        description=description,
        author=author,
        license=license,
        readme_content=readme_content,
        source_url=source_url,
        repository_url=repository_url,
        uploaded_by=uploaded_by,
        is_sensitive=is_sensitive,
        access_control_script=access_control_script,
        ingest_status=ingest_status,
        quality_gate_result=quality_gate_result
    )
    db.add(package)
    db.commit()
    db.refresh(package)
    return package


def get_package_by_id(db: Session, package_id: int) -> Optional[Package]:
    """Get package by ID."""
    return db.query(Package).filter(Package.id == package_id).first()


def get_package_by_name_version(db: Session, name: str, version: str) -> Optional[Package]:
    """Get package by name and version."""
    return db.query(Package).filter(
        and_(Package.name == name, Package.version == version)
    ).first()


def get_packages_by_name(db: Session, name: str) -> List[Package]:
    """
    Get all packages matching a name (any version).
    
    Supports multiple name formats for parent model lookup:
    - Exact match: "trained-gender"
    - With owner prefix: "crangana-trained-gender"
    - Just the repo part of "owner/repo": extracts "repo"
    
    Args:
        db: Database session
        name: Package name to search for
        
    Returns:
        List of packages matching the name
    """
    # Extract just the repo name if it's in owner/repo format
    if "/" in name:
        name = name.split("/")[-1]
    
    # Try exact match first
    packages = db.query(Package).filter(Package.name == name).all()
    if packages:
        return packages
    
    # Try with owner-repo format (owner/repo -> owner-repo)
    # This handles cases like "microsoft/resnet-50" -> "microsoft-resnet-50"
    if "-" in name:
        # Also try just the last part after the last hyphen for cases like
        # "microsoft-resnet-50" where we want to match "resnet-50"
        parts = name.split("-", 1)
        if len(parts) == 2:
            packages = db.query(Package).filter(Package.name == parts[1]).all()
            if packages:
                return packages
    
    return []


def get_packages(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    name_filter: Optional[str] = None,
    use_regex: bool = False
) -> List[Package]:
    """
    Get all packages with optional filtering and pagination.
    
    Args:
        db: Database session
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return
        name_filter: Filter pattern for package name (and README when using regex)
        use_regex: If True, treat name_filter as regex pattern; if False, use SQL LIKE
    
    Returns:
        List of packages matching the filter
    """
    query = db.query(Package)
    
    if name_filter:
        if use_regex:
            # Use regex pattern matching on both name and README content
            # PostgreSQL uses ~ operator, SQLite uses REGEXP
            # Search in name OR readme_content
            from sqlalchemy import or_
            query = query.filter(
                or_(
                    Package.name.op('~')(name_filter),
                    Package.readme_content.op('~')(name_filter)
                )
            )
        else:
            # Use SQL LIKE for simple pattern matching
            query = query.filter(Package.name.ilike(f"%{name_filter}%"))
    
    return query.offset(skip).limit(limit).all()


# Alias for backward compatibility
list_packages = get_packages


def get_all_packages(db: Session) -> List[Package]:
    """Get all packages without pagination."""
    return db.query(Package).all()


def delete_package(db: Session, package_id: int) -> bool:
    """Delete a package by ID."""
    package = db.query(Package).filter(Package.id == package_id).first()
    if package:
        db.delete(package)
        db.commit()
        return True
    return False


def increment_download_count(db: Session, package_id: int) -> None:
    """Increment download count for a package."""
    package = db.query(Package).filter(Package.id == package_id).first()
    if package:
        package.download_count += 1
        db.commit()


# ============================================================================
# Package Score CRUD Operations
# ============================================================================

def create_or_update_package_score(
    db: Session,
    package_id: int,
    **scores
) -> PackageScore:
    """Create or update package scores."""
    # Check if score already exists
    existing_score = db.query(PackageScore).filter(
        PackageScore.package_id == package_id
    ).first()
    
    if existing_score:
        # Update existing scores
        for key, value in scores.items():
            if hasattr(existing_score, key):
                setattr(existing_score, key, value)
        existing_score.scored_at = datetime.utcnow()
        db.commit()
        db.refresh(existing_score)
        return existing_score
    else:
        # Create new score
        score = PackageScore(package_id=package_id, **scores)
        db.add(score)
        db.commit()
        db.refresh(score)
        return score


def get_package_scores(db: Session, package_id: int) -> Optional[PackageScore]:
    """Get scores for a package."""
    return db.query(PackageScore).filter(
        PackageScore.package_id == package_id
    ).first()


# ============================================================================
# Package Lineage CRUD Operations
# ============================================================================

def create_lineage(
    db: Session,
    parent_package_id: int,
    child_package_id: int,
    relationship_type: str = "depends_on"
) -> PackageLineage:
    """Create a lineage relationship between packages."""
    lineage = PackageLineage(
        parent_package_id=parent_package_id,
        child_package_id=child_package_id,
        relationship_type=relationship_type
    )
    db.add(lineage)
    db.commit()
    db.refresh(lineage)
    return lineage


def get_package_parents(db: Session, package_id: int) -> List[PackageLineage]:
    """Get all parent packages (dependencies)."""
    return db.query(PackageLineage).filter(
        PackageLineage.child_package_id == package_id
    ).all()


def get_package_children(db: Session, package_id: int) -> List[PackageLineage]:
    """Get all child packages (dependents)."""
    return db.query(PackageLineage).filter(
        PackageLineage.parent_package_id == package_id
    ).all()


def update_dependent_tree_scores(
    db: Session,
    package_id: int,
    max_depth: int = 3,
    visited: Optional[set] = None
) -> int:
    """
    Update tree scores for all dependent (child) packages recursively.
    
    When a package's net_score is updated, this function recalculates
    the tree scores for all child packages that depend on it.
    
    Args:
        db: Database session
        package_id: ID of package whose score was updated
        max_depth: Maximum recursion depth (default 3)
        visited: Set of already-visited package IDs (for cycle detection)
        
    Returns:
        Number of packages updated
    """
    import logging
    from src.scoring import MetricScorer
    
    logger = logging.getLogger(__name__)
    
    if visited is None:
        visited = set()
    
    # Prevent infinite recursion
    if package_id in visited or max_depth <= 0:
        return 0
    
    visited.add(package_id)
    updated_count = 0
    
    try:
        # Get all children of this package
        child_lineages = get_package_children(db, package_id)
        
        if not child_lineages:
            logger.debug(f"No children found for package {package_id}")
            return 0
        
        logger.info(
            f"Updating tree scores for {len(child_lineages)} children "
            f"of package {package_id}"
        )
        
        for lineage in child_lineages:
            child_id = lineage.child_package_id
            
            # Skip if already visited (cycle detection)
            if child_id in visited:
                logger.debug(
                    f"Skipping package {child_id} (cycle detected)"
                )
                continue
            
            try:
                # Get child package
                child_pkg = get_package_by_id(db, child_id)
                if not child_pkg:
                    continue
                
                # Get child's current scores
                child_scores = get_package_scores(db, child_id)
                if not child_scores:
                    logger.debug(
                        f"No scores found for child package {child_id}"
                    )
                    continue
                
                # Recalculate tree score using MetricScorer
                scorer = MetricScorer()
                scorer.db_session = db  # Set db session for score lookups
                
                # Create minimal context for tree score calculation
                from src.models import ModelContext
                context = ModelContext(
                    model_url=child_pkg.source_url or "",
                    config_data=(
                        child_pkg.lineage_metadata
                        if child_pkg.lineage_metadata else {}
                    ),
                    readme_content=child_pkg.readme_content or ""
                )
                
                # Calculate new tree score
                new_tree_score = scorer._calculate_treescore(
                    context,
                    current_net_score=child_scores.net_score
                )
                
                # Update if changed
                if new_tree_score != child_scores.treescore:
                    old_score = child_scores.treescore
                    child_scores.treescore = new_tree_score
                    db.commit()
                    
                    updated_count += 1
                    logger.info(
                        f"Updated tree score for package {child_id}: "
                        f"{old_score:.4f} -> {new_tree_score:.4f}"
                    )
                    
                    # Recursively update children of this child
                    child_updated = update_dependent_tree_scores(
                        db,
                        child_id,
                        max_depth=max_depth - 1,
                        visited=visited
                    )
                    updated_count += child_updated
                    
            except Exception as e:
                logger.error(
                    f"Failed to update tree score for child {child_id}: {e}"
                )
                continue
        
        return updated_count
        
    except Exception as e:
        logger.error(
            f"Error updating dependent tree scores for {package_id}: {e}"
        )
        return updated_count


# ============================================================================
# Download History CRUD Operations
# ============================================================================

def record_download(
    db: Session,
    package_id: int,
    user_id: Optional[int] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    access_granted: bool = True,
    access_control_result: Optional[dict] = None
) -> DownloadHistory:
    """Record a package download."""
    download = DownloadHistory(
        package_id=package_id,
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        access_granted=access_granted,
        access_control_result=access_control_result
    )
    db.add(download)
    db.commit()
    db.refresh(download)
    return download


def get_download_history(
    db: Session,
    package_id: Optional[int] = None,
    user_id: Optional[int] = None,
    limit: int = 100
) -> List[DownloadHistory]:
    """Get download history with optional filtering."""
    query = db.query(DownloadHistory)
    
    if package_id:
        query = query.filter(DownloadHistory.package_id == package_id)
    if user_id:
        query = query.filter(DownloadHistory.user_id == user_id)
    
    return query.order_by(desc(DownloadHistory.downloaded_at)).limit(limit).all()


# ============================================================================
# System Health Metrics CRUD Operations
# ============================================================================

def record_health_metric(
    db: Session,
    metric_name: str,
    metric_value: float,
    metric_unit: Optional[str] = None,
    tags: Optional[dict] = None
) -> SystemHealthMetric:
    """Record a system health metric."""
    metric = SystemHealthMetric(
        metric_name=metric_name,
        metric_value=metric_value,
        metric_unit=metric_unit,
        tags=tags
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def get_health_metrics(
    db: Session,
    metric_name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000
) -> List[SystemHealthMetric]:
    """Get health metrics with optional filtering."""
    query = db.query(SystemHealthMetric)
    
    if metric_name:
        query = query.filter(SystemHealthMetric.metric_name == metric_name)
    if start_time:
        query = query.filter(SystemHealthMetric.recorded_at >= start_time)
    if end_time:
        query = query.filter(SystemHealthMetric.recorded_at <= end_time)
    
    return query.order_by(desc(SystemHealthMetric.recorded_at)).limit(limit).all()
