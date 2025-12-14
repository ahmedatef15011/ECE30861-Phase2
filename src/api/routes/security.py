"""Security-related endpoints for sensitive module tracking and malicious model detection."""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import json
import logging

from src.api.dependencies import get_db, get_current_user, get_optional_user
from src.database import crud
from src.database.models import User, Package, SensitiveModuleHistory, MaliciousModelReport

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Pydantic Schemas for Security Endpoints
# ============================================================================

class SensitiveHistoryEntryResponse(BaseModel):
    """Response schema for a single sensitive module history entry."""
    id: int
    package_id: int
    package_name: Optional[str] = None
    user_id: Optional[int] = None
    username: Optional[str] = None
    action: str
    field_changed: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    change_summary: Optional[str] = None
    changed_at: datetime
    ip_address: Optional[str] = None
    additional_context: Optional[dict] = None
    
    class Config:
        from_attributes = True


class SensitiveHistoryResponse(BaseModel):
    """Response schema for sensitive module history list."""
    total: int
    entries: List[SensitiveHistoryEntryResponse]


class MaliciousReportRequest(BaseModel):
    """Request schema for reporting a potentially malicious model."""
    package_id: int = Field(..., description="ID of the suspected malicious package")
    reason: str = Field(..., min_length=10, description="Reason for suspecting the model")
    severity: str = Field("medium", description="Severity level: low, medium, high, critical")
    evidence: Optional[dict] = Field(None, description="Additional evidence as JSON")


class MaliciousModelResponse(BaseModel):
    """Response schema for a malicious model report."""
    id: int
    package_id: int
    package_name: Optional[str] = None
    package_version: Optional[str] = None
    detection_method: str
    severity: str
    status: str
    reason: str
    evidence: Optional[dict] = None
    reported_at: datetime
    reported_by_username: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    reviewed_by_username: Optional[str] = None
    resolution_notes: Optional[str] = None
    
    class Config:
        from_attributes = True


class MaliciousModelsListResponse(BaseModel):
    """Response schema for list of suspected malicious models."""
    total: int
    models: List[MaliciousModelResponse]


class UpdateReportStatusRequest(BaseModel):
    """Request schema for updating malicious report status."""
    status: str = Field(..., description="New status: pending, confirmed, dismissed, under_review")
    resolution_notes: Optional[str] = Field(None, description="Notes about the resolution")


# ============================================================================
# Sensitive Module History Endpoints
# ============================================================================

@router.get(
    "/sensitive-modules/{package_id}/history",
    response_model=SensitiveHistoryResponse,
    summary="Get sensitive module history",
    description="Retrieve historical information about changes to a sensitive module."
)
async def get_sensitive_module_history(
    package_id: int,
    action: Optional[str] = Query(None, description="Filter by action type"),
    start_date: Optional[datetime] = Query(None, description="Filter entries after this date"),
    end_date: Optional[datetime] = Query(None, description="Filter entries before this date"),
    limit: int = Query(100, ge=1, le=500, description="Maximum entries to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get historical information about a sensitive module.
    
    Returns what changed, when, and by whom for the specified sensitive package.
    
    Args:
        package_id: ID of the package to get history for
        action: Optional filter by action type
        start_date: Optional filter for entries after this date
        end_date: Optional filter for entries before this date
        limit: Maximum number of entries to return
        
    Returns:
        List of history entries with user and change information
    """
    # Verify package exists and is sensitive
    package = db.query(Package).filter(Package.id == package_id).first()
    if not package:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Package with ID {package_id} not found"
        )
    
    if not package.is_sensitive:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Package {package.name} is not marked as sensitive"
        )
    
    # Get history entries
    history_entries = crud.get_sensitive_module_history(
        db=db,
        package_id=package_id,
        action=action,
        start_time=start_date,
        end_time=end_date,
        limit=limit
    )
    
    # Build response with user information
    entries = []
    for entry in history_entries:
        username = None
        if entry.user_id:
            user = crud.get_user_by_id(db, entry.user_id)
            username = user.username if user else None
        
        entries.append(SensitiveHistoryEntryResponse(
            id=entry.id,
            package_id=entry.package_id,
            package_name=package.name,
            user_id=entry.user_id,
            username=username,
            action=entry.action,
            field_changed=entry.field_changed,
            old_value=entry.old_value,
            new_value=entry.new_value,
            change_summary=entry.change_summary,
            changed_at=entry.changed_at,
            ip_address=entry.ip_address,
            additional_context=entry.additional_context
        ))
    
    return SensitiveHistoryResponse(
        total=len(entries),
        entries=entries
    )


@router.get(
    "/sensitive-modules/history",
    response_model=SensitiveHistoryResponse,
    summary="Get all sensitive module history",
    description="Retrieve historical information about all sensitive module changes."
)
async def get_all_sensitive_history(
    action: Optional[str] = Query(None, description="Filter by action type"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    start_date: Optional[datetime] = Query(None, description="Filter entries after this date"),
    end_date: Optional[datetime] = Query(None, description="Filter entries before this date"),
    limit: int = Query(100, ge=1, le=500, description="Maximum entries to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get historical information about all sensitive module changes.
    
    Only admins can access the full history. Regular users can only see history
    for packages they have access to.
    
    Returns:
        List of history entries across all sensitive modules
    """
    # Only admins can see all history
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required to view all sensitive module history"
        )
    
    # Get history entries
    history_entries = crud.get_sensitive_module_history(
        db=db,
        user_id=user_id,
        action=action,
        start_time=start_date,
        end_time=end_date,
        limit=limit
    )
    
    # Build response with package and user information
    entries = []
    for entry in history_entries:
        username = None
        package_name = None
        
        if entry.user_id:
            user = crud.get_user_by_id(db, entry.user_id)
            username = user.username if user else None
        
        package = db.query(Package).filter(Package.id == entry.package_id).first()
        package_name = package.name if package else None
        
        entries.append(SensitiveHistoryEntryResponse(
            id=entry.id,
            package_id=entry.package_id,
            package_name=package_name,
            user_id=entry.user_id,
            username=username,
            action=entry.action,
            field_changed=entry.field_changed,
            old_value=entry.old_value,
            new_value=entry.new_value,
            change_summary=entry.change_summary,
            changed_at=entry.changed_at,
            ip_address=entry.ip_address,
            additional_context=entry.additional_context
        ))
    
    return SensitiveHistoryResponse(
        total=len(entries),
        entries=entries
    )


# ============================================================================
# Malicious Model Detection Endpoints
# ============================================================================

@router.get(
    "/malicious-models",
    response_model=MaliciousModelsListResponse,
    summary="Get suspected malicious models",
    description="Returns a list of models suspected to be malicious."
)
async def get_suspected_malicious_models(
    status_filter: Optional[str] = Query(
        None, 
        alias="status",
        description="Filter by status: pending, confirmed, dismissed, under_review"
    ),
    severity: Optional[str] = Query(
        None,
        description="Filter by minimum severity: low, medium, high, critical"
    ),
    include_dismissed: bool = Query(
        False,
        description="Whether to include dismissed reports"
    ),
    limit: int = Query(100, ge=1, le=500, description="Maximum models to return"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_optional_user),
):
    """
    Get a list of models suspected to be malicious.
    
    Returns all models that have been flagged as potentially malicious,
    including their detection method, severity, and current status.
    
    Args:
        status_filter: Optional filter by report status
        severity: Optional filter by minimum severity level
        include_dismissed: Whether to include dismissed reports
        limit: Maximum number of results to return
        
    Returns:
        List of suspected malicious models with details
    """
    # Get malicious reports
    if status_filter:
        reports = crud.get_malicious_model_reports(
            db=db,
            status=status_filter,
            severity=severity,
            limit=limit
        )
    else:
        reports = crud.get_suspected_malicious_models(
            db=db,
            include_dismissed=include_dismissed,
            min_severity=severity,
            limit=limit
        )
    
    # Build response with package information
    models = []
    for report in reports:
        package = db.query(Package).filter(Package.id == report.package_id).first()
        
        reported_by_username = None
        reviewed_by_username = None
        
        if report.reported_by_user_id:
            reporter = crud.get_user_by_id(db, report.reported_by_user_id)
            reported_by_username = reporter.username if reporter else None
        
        if report.reviewed_by_user_id:
            reviewer = crud.get_user_by_id(db, report.reviewed_by_user_id)
            reviewed_by_username = reviewer.username if reviewer else None
        
        models.append(MaliciousModelResponse(
            id=report.id,
            package_id=report.package_id,
            package_name=package.name if package else None,
            package_version=package.version if package else None,
            detection_method=report.detection_method,
            severity=report.severity,
            status=report.status,
            reason=report.reason,
            evidence=report.evidence,
            reported_at=report.reported_at,
            reported_by_username=reported_by_username,
            reviewed_at=report.reviewed_at,
            reviewed_by_username=reviewed_by_username,
            resolution_notes=report.resolution_notes
        ))
    
    return MaliciousModelsListResponse(
        total=len(models),
        models=models
    )


@router.post(
    "/malicious-models/report",
    response_model=MaliciousModelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Report a potentially malicious model",
    description="Submit a report for a model suspected to be malicious."
)
async def report_malicious_model(
    request: MaliciousReportRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Report a potentially malicious model.
    
    Allows users to submit reports about models they suspect to be malicious.
    The report will be reviewed by administrators.
    
    Args:
        request: Report details including package ID, reason, severity, and evidence
        
    Returns:
        Created malicious model report
    """
    # Validate severity
    valid_severities = ["low", "medium", "high", "critical"]
    if request.severity not in valid_severities:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid severity. Must be one of: {', '.join(valid_severities)}"
        )
    
    # Verify package exists
    package = db.query(Package).filter(Package.id == request.package_id).first()
    if not package:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Package with ID {request.package_id} not found"
        )
    
    # Create the report
    report = crud.create_malicious_model_report(
        db=db,
        package_id=request.package_id,
        detection_method="USER_REPORT",
        reason=request.reason,
        severity=request.severity,
        evidence=request.evidence,
        reported_by_user_id=current_user.id
    )
    
    logger.info(
        f"Malicious model report created: package_id={request.package_id}, "
        f"severity={request.severity}, reported_by={current_user.username}"
    )
    
    return MaliciousModelResponse(
        id=report.id,
        package_id=report.package_id,
        package_name=package.name,
        package_version=package.version,
        detection_method=report.detection_method,
        severity=report.severity,
        status=report.status,
        reason=report.reason,
        evidence=report.evidence,
        reported_at=report.reported_at,
        reported_by_username=current_user.username,
        reviewed_at=report.reviewed_at,
        reviewed_by_username=None,
        resolution_notes=report.resolution_notes
    )


@router.get(
    "/malicious-models/{package_id}",
    response_model=MaliciousModelsListResponse,
    summary="Get malicious reports for a package",
    description="Get all malicious reports for a specific package."
)
async def get_package_malicious_reports(
    package_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_optional_user),
):
    """
    Get all malicious reports for a specific package.
    
    Args:
        package_id: ID of the package to get reports for
        
    Returns:
        List of malicious reports for the package
    """
    # Verify package exists
    package = db.query(Package).filter(Package.id == package_id).first()
    if not package:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Package with ID {package_id} not found"
        )
    
    reports = crud.get_package_malicious_reports(db, package_id)
    
    models = []
    for report in reports:
        reported_by_username = None
        reviewed_by_username = None
        
        if report.reported_by_user_id:
            reporter = crud.get_user_by_id(db, report.reported_by_user_id)
            reported_by_username = reporter.username if reporter else None
        
        if report.reviewed_by_user_id:
            reviewer = crud.get_user_by_id(db, report.reviewed_by_user_id)
            reviewed_by_username = reviewer.username if reviewer else None
        
        models.append(MaliciousModelResponse(
            id=report.id,
            package_id=report.package_id,
            package_name=package.name,
            package_version=package.version,
            detection_method=report.detection_method,
            severity=report.severity,
            status=report.status,
            reason=report.reason,
            evidence=report.evidence,
            reported_at=report.reported_at,
            reported_by_username=reported_by_username,
            reviewed_at=report.reviewed_at,
            reviewed_by_username=reviewed_by_username,
            resolution_notes=report.resolution_notes
        ))
    
    return MaliciousModelsListResponse(
        total=len(models),
        models=models
    )


@router.patch(
    "/malicious-models/reports/{report_id}/status",
    response_model=MaliciousModelResponse,
    summary="Update malicious report status",
    description="Update the status of a malicious model report (admin only)."
)
async def update_report_status(
    report_id: int,
    request: UpdateReportStatusRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update the status of a malicious model report.
    
    Only administrators can update report status.
    
    Args:
        report_id: ID of the report to update
        request: New status and optional resolution notes
        
    Returns:
        Updated malicious model report
    """
    # Only admins can update status
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required to update report status"
        )
    
    # Validate status
    valid_statuses = ["pending", "confirmed", "dismissed", "under_review"]
    if request.status not in valid_statuses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        )
    
    # Update the report
    report = crud.update_malicious_model_report_status(
        db=db,
        report_id=report_id,
        new_status=request.status,
        reviewed_by_user_id=current_user.id,
        resolution_notes=request.resolution_notes
    )
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report with ID {report_id} not found"
        )
    
    package = db.query(Package).filter(Package.id == report.package_id).first()
    
    reported_by_username = None
    if report.reported_by_user_id:
        reporter = crud.get_user_by_id(db, report.reported_by_user_id)
        reported_by_username = reporter.username if reporter else None
    
    logger.info(
        f"Malicious report status updated: report_id={report_id}, "
        f"new_status={request.status}, reviewed_by={current_user.username}"
    )
    
    return MaliciousModelResponse(
        id=report.id,
        package_id=report.package_id,
        package_name=package.name if package else None,
        package_version=package.version if package else None,
        detection_method=report.detection_method,
        severity=report.severity,
        status=report.status,
        reason=report.reason,
        evidence=report.evidence,
        reported_at=report.reported_at,
        reported_by_username=reported_by_username,
        reviewed_at=report.reviewed_at,
        reviewed_by_username=current_user.username,
        resolution_notes=report.resolution_notes
    )


# ============================================================================
# Helper function to track sensitive module changes
# ============================================================================

def track_sensitive_module_change(
    db: Session,
    package_id: int,
    action: str,
    user_id: Optional[int] = None,
    field_changed: Optional[str] = None,
    old_value: Optional[str] = None,
    new_value: Optional[str] = None,
    change_summary: Optional[str] = None,
    request: Optional[Request] = None
) -> SensitiveModuleHistory:
    """
    Helper function to track changes to sensitive modules.
    
    This should be called whenever a sensitive module is created, updated,
    or accessed in a significant way.
    
    Args:
        db: Database session
        package_id: ID of the package
        action: Type of action
        user_id: ID of the user making the change
        field_changed: Name of the field that changed
        old_value: Previous value (will be JSON serialized if dict)
        new_value: New value (will be JSON serialized if dict)
        change_summary: Human-readable summary
        request: FastAPI request object for extracting IP/user agent
        
    Returns:
        Created history entry
    """
    ip_address = None
    user_agent = None
    
    if request:
        # Extract IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip_address = forwarded.split(",")[0].strip()
        else:
            ip_address = request.client.host if request.client else None
        
        user_agent = request.headers.get("User-Agent")
    
    # Serialize values if they're dicts
    if isinstance(old_value, dict):
        old_value = json.dumps(old_value)
    if isinstance(new_value, dict):
        new_value = json.dumps(new_value)
    
    return crud.create_sensitive_module_history(
        db=db,
        package_id=package_id,
        action=action,
        user_id=user_id,
        field_changed=field_changed,
        old_value=old_value,
        new_value=new_value,
        change_summary=change_summary,
        ip_address=ip_address,
        user_agent=user_agent
    )
