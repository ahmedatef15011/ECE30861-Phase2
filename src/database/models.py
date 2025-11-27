"""SQLAlchemy database models for ML Model Registry."""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    JSON,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    permissions = relationship("Permission", back_populates="user", cascade="all, delete-orphan")
    auth_tokens = relationship("AuthToken", back_populates="user", cascade="all, delete-orphan")
    packages_uploaded = relationship("Package", back_populates="uploader", foreign_keys="Package.uploaded_by")
    download_history = relationship("DownloadHistory", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(username='{self.username}', is_admin={self.is_admin})>"


class Permission(Base):
    """Permission model for fine-grained access control."""
    
    __tablename__ = "permissions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Permission types
    can_upload = Column(Boolean, default=False, nullable=False)
    can_search = Column(Boolean, default=True, nullable=False)  # Default: everyone can search
    can_download = Column(Boolean, default=True, nullable=False)  # Default: everyone can download
    can_rate = Column(Boolean, default=False, nullable=False)
    can_delete = Column(Boolean, default=False, nullable=False)
    
    # Usage limits
    max_uploads_per_day = Column(Integer, default=10, nullable=True)
    max_downloads_per_day = Column(Integer, default=100, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="permissions")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', name='uix_user_permission'),
    )
    
    def __repr__(self):
        return f"<Permission(user_id={self.user_id}, can_upload={self.can_upload})>"


class AuthToken(Base):
    """Authentication token model for JWT-based auth."""
    
    __tablename__ = "auth_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String(500), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    is_revoked = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="auth_tokens")
    
    def __repr__(self):
        return f"<AuthToken(user_id={self.user_id}, expires_at={self.expires_at})>"


class Package(Base):
    """Package model representing an ML model or dataset."""
    
    __tablename__ = "packages"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Package identification
    name = Column(String(200), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    # Artifact type: model, dataset, or code
    artifact_type = Column(String(20), nullable=False, default="model")
    
    # Metadata
    description = Column(Text, nullable=True)
    author = Column(String(100), nullable=True)
    license = Column(String(100), nullable=True)
    readme_content = Column(Text, nullable=True)
    
    # Storage
    s3_key = Column(String(500), nullable=False)  # S3 object key
    s3_bucket = Column(String(100), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False)
    
    # Source information
    source_url = Column(String(500), nullable=True)  # Original URL (e.g., HuggingFace)
    repository_url = Column(String(500), nullable=True)  # GitHub repo URL
    
    # Sensitive model control (Security Track)
    is_sensitive = Column(Boolean, default=False, nullable=False)
    access_control_script = Column(Text, nullable=True)  # JavaScript for sensitive models
    
    # Upload tracking
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Download tracking
    download_count = Column(Integer, default=0, nullable=False)
    
    # Ingest status tracking (Phase 2 enhancement)
    ingest_status = Column(
        String(20),
        nullable=False,
        default="approved",
        index=True
    )
    # Possible values: "approved", "rejected", "pending"
    # - approved: Passed quality gate
    # - rejected: Failed quality gate
    # - pending: Submitted but not yet evaluated (for async workflow)
    
    quality_gate_result = Column(
        JSON,
        nullable=True
    )
    # Stores detailed quality gate evaluation results:
    # {
    #   "passed": true/false,
    #   "evaluated_at": "2025-11-23T19:00:00Z",
    #   "net_score": 0.84,  # if passed
    #   "failing_metrics": [...]  # if failed
    # }
    
    # Relationships
    uploader = relationship("User", back_populates="packages_uploaded", foreign_keys=[uploaded_by])
    scores = relationship("PackageScore", back_populates="package", cascade="all, delete-orphan")
    parent_lineages = relationship(
        "PackageLineage",
        back_populates="child_package",
        foreign_keys="PackageLineage.child_package_id",
        cascade="all, delete-orphan"
    )
    child_lineages = relationship(
        "PackageLineage",
        back_populates="parent_package",
        foreign_keys="PackageLineage.parent_package_id",
        cascade="all, delete-orphan"
    )
    download_history = relationship("DownloadHistory", back_populates="package", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('name', 'version', name='uix_package_name_version'),
        Index('ix_package_name_version', 'name', 'version'),
        Index('ix_package_ingest_status', 'ingest_status'),  # For filtering approved artifacts
    )
    
    def __repr__(self):
        return f"<Package(name='{self.name}', version='{self.version}')>"


class PackageScore(Base):
    """Package scores from various metrics."""
    
    __tablename__ = "package_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    package_id = Column(Integer, ForeignKey("packages.id", ondelete="CASCADE"), nullable=False)
    
    # Phase 1 metrics (8 existing)
    ramp_up_time = Column(Float, nullable=True)
    bus_factor = Column(Float, nullable=True)
    performance_claims = Column(Float, nullable=True)
    license_score = Column(Float, nullable=True)
    size_score = Column(Float, nullable=True)
    # Size score breakdown by device
    size_score_raspberry_pi = Column(Float, nullable=True)
    size_score_jetson_nano = Column(Float, nullable=True)
    size_score_desktop_pc = Column(Float, nullable=True)
    size_score_aws_server = Column(Float, nullable=True)
    dataset_quality = Column(Float, nullable=True)
    dataset_code_linkage = Column(Float, nullable=True)
    code_quality = Column(Float, nullable=True)
    
    # Phase 2 new metrics (3 additional)
    reproducibility = Column(Float, nullable=True)  # LLM-based (0, 0.5, or 1)
    reviewedness = Column(Float, nullable=True)  # PR review fraction
    treescore = Column(Float, nullable=True)  # Average of parent scores
    
    # Overall score (weighted average)
    net_score = Column(Float, nullable=True)
    
    # Metadata
    scored_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    scoring_latency_ms = Column(Integer, nullable=True)  # Time taken to compute scores
    
    # Relationships
    package = relationship("Package", back_populates="scores")
    
    # Constraints
    __table_args__ = (
        Index('ix_package_score_package_id', 'package_id'),
    )
    
    def __repr__(self):
        return f"<PackageScore(package_id={self.package_id}, net_score={self.net_score})>"


class PackageLineage(Base):
    """Package lineage/dependency relationships."""
    
    __tablename__ = "package_lineage"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Parent-child relationship
    parent_package_id = Column(Integer, ForeignKey("packages.id", ondelete="CASCADE"), nullable=False)
    child_package_id = Column(Integer, ForeignKey("packages.id", ondelete="CASCADE"), nullable=False)
    
    # Relationship type
    relationship_type = Column(String(50), nullable=False)  # e.g., "depends_on", "derived_from", "fine_tuned_from"
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    parent_package = relationship("Package", foreign_keys=[parent_package_id], back_populates="child_lineages")
    child_package = relationship("Package", foreign_keys=[child_package_id], back_populates="parent_lineages")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('parent_package_id', 'child_package_id', name='uix_lineage_parent_child'),
        Index('ix_lineage_parent', 'parent_package_id'),
        Index('ix_lineage_child', 'child_package_id'),
    )
    
    def __repr__(self):
        return f"<PackageLineage(parent={self.parent_package_id}, child={self.child_package_id})>"


class DownloadHistory(Base):
    """Download history tracking (Security Track requirement)."""
    
    __tablename__ = "download_history"
    
    id = Column(Integer, primary_key=True, index=True)
    package_id = Column(Integer, ForeignKey("packages.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Download metadata
    downloaded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    user_agent = Column(String(500), nullable=True)
    
    # For sensitive models
    access_granted = Column(Boolean, default=True, nullable=False)
    access_control_result = Column(JSON, nullable=True)  # Result of JS execution
    
    # Relationships
    package = relationship("Package", back_populates="download_history")
    user = relationship("User", back_populates="download_history")
    
    # Constraints
    __table_args__ = (
        Index('ix_download_package_user', 'package_id', 'user_id'),
        Index('ix_download_timestamp', 'downloaded_at'),
    )
    
    def __repr__(self):
        return f"<DownloadHistory(package_id={self.package_id}, user_id={self.user_id})>"


class SystemHealthMetric(Base):
    """System health metrics for observability dashboard."""
    
    __tablename__ = "system_health_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Metric identification
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50), nullable=True)  # e.g., "ms", "count", "bytes"
    
    # Metadata
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    tags = Column(JSON, nullable=True)  # Additional context (e.g., {"endpoint": "/packages", "method": "GET"})
    
    # Constraints
    __table_args__ = (
        Index('ix_health_metric_name_time', 'metric_name', 'recorded_at'),
    )
    
    def __repr__(self):
        return f"<SystemHealthMetric(metric_name='{self.metric_name}', value={self.metric_value})>"
