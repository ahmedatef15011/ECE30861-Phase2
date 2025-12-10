"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# ============================================================================
# User Schemas
# ============================================================================

class UserRegister(BaseModel):
    """Schema for user registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    is_admin: bool = False


class UserLogin(BaseModel):
    """Schema for user login request."""
    username: str
    password: str


class UserResponse(BaseModel):
    """Schema for user response."""
    id: int
    username: str
    email: str
    is_admin: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Schema for authentication token response."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ============================================================================
# Package Schemas
# ============================================================================

class PackageCreate(BaseModel):
    """Schema for creating a new package."""
    name: str = Field(..., min_length=1, max_length=255)
    version: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = None
    s3_url: Optional[str] = None


class PackageUpdate(BaseModel):
    """Schema for updating package metadata."""
    description: Optional[str] = None
    s3_url: Optional[str] = None


class PackageResponse(BaseModel):
    """Schema for package response."""
    id: int
    name: str
    version: str
    description: Optional[str] = None
    s3_key: str
    s3_bucket: str
    file_size_bytes: int
    uploaded_by: Optional[int] = None
    uploaded_at: datetime
    download_count: int = 0
    
    class Config:
        from_attributes = True


class PackageListResponse(BaseModel):
    """Schema for paginated package list."""
    total: int
    page: int
    page_size: int
    packages: List[PackageResponse]


# ============================================================================
# Package Score Schemas
# ============================================================================

class PackageScoreCreate(BaseModel):
    """Schema for creating/updating package scores."""
    # Phase 1 metrics
    ramp_up_time: Optional[float] = Field(None, ge=0.0, le=1.0)
    bus_factor: Optional[float] = Field(None, ge=0.0, le=1.0)
    performance_claims: Optional[float] = Field(None, ge=0.0, le=1.0)
    license_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    size_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    dataset_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    dataset_code_linkage: Optional[float] = Field(None, ge=0.0, le=1.0)
    code_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Phase 2 metrics
    reproducibility: Optional[float] = Field(None, ge=0.0, le=1.0)
    reviewedness: Optional[float] = Field(None, ge=-1.0, le=1.0)
    treescore: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Overall score
    net_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class PackageScoreResponse(PackageScoreCreate):
    """Schema for package score response."""
    id: int
    package_id: int
    scored_at: datetime
    scoring_latency_ms: Optional[int] = None
    
    class Config:
        from_attributes = True


# ============================================================================
# Search and Filter Schemas
# ============================================================================

class PackageSearchQuery(BaseModel):
    """Schema for package search query."""
    name_pattern: Optional[str] = Field(None, description="Regex pattern for package name")
    version: Optional[str] = Field(None, description="Version filter (exact, range, tilde, caret)")
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


# ============================================================================
# Health and System Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    timestamp: datetime
    database_status: str
    uptime_seconds: float


class ResetResponse(BaseModel):
    """Schema for system reset response."""
    message: str
    default_admin_created: bool


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorResponse(BaseModel):
    """Schema for error responses."""
    detail: str
    error_code: Optional[str] = None


# ============================================================================
# Audit Trail Schemas
# ============================================================================

class AuditUser(BaseModel):
    """User information in audit trail."""
    name: str
    is_admin: bool


class AuditArtifactMetadata(BaseModel):
    """Artifact metadata in audit trail."""
    name: str
    id: str
    type: str


class ArtifactAuditEntry(BaseModel):
    """One entry in an artifact's audit history."""
    user: AuditUser
    date: datetime
    artifact: AuditArtifactMetadata
    action: str  # CREATE, UPDATE, DOWNLOAD, RATE, AUDIT
