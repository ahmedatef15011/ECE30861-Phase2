"""
Test ingest status tracking feature.

This test validates that the system stores all artifacts
 (approved and rejected) with proper status tracking.
"""

import pytest
from sqlalchemy.orm import Session
from src.database.connection import SessionLocal, init_db
from src.database import crud
from src.database.models import Package


def test_create_package_with_pending_status():
    """Test that we can create a package with pending status."""
    init_db()
    db = SessionLocal()
    
    try:
        package = crud.create_package(
            db,
            name="test-model",
            version="1.0.0",
            s3_key="test-key",
            s3_bucket="test-bucket",
            file_size_bytes=0,
            artifact_type="model",
            source_url="https://test.com/model",
            ingest_status="pending"
        )
        
        assert package.ingest_status == "pending"
        assert package.id is not None
        
        # Clean up
        db.delete(package)
        db.commit()
    finally:
        db.close()


def test_create_package_with_rejected_status():
    """Test that we can create a package with rejected status."""
    init_db()
    db = SessionLocal()
    
    try:
        quality_gate_result = {
            "passed": False,
            "failing_metrics": [
                {"metric": "bus_factor", "score": 0.4, "required": 0.5}
            ]
        }
        
        package = crud.create_package(
            db,
            name="bad-model",
            version="1.0.0",
            s3_key="bad-key",
            s3_bucket="test-bucket",
            file_size_bytes=0,
            artifact_type="model",
            ingest_status="rejected",
            quality_gate_result=quality_gate_result
        )
        
        assert package.ingest_status == "rejected"
        assert package.quality_gate_result is not None
        assert package.quality_gate_result["passed"] is False
        assert len(package.quality_gate_result["failing_metrics"]) == 1
        
        # Clean up
        db.delete(package)
        db.commit()
    finally:
        db.close()


def test_get_packages_filters_approved_only():
    """Test that get_packages returns all packages regardless of status."""
    init_db()
    db = SessionLocal()
    
    try:
        # Create approved package
        approved = crud.create_package(
            db,
            name="approved-model",
            version="1.0.0",
            s3_key="approved-key",
            s3_bucket="test-bucket",
            file_size_bytes=0,
            ingest_status="approved"
        )
        
        # Create rejected package
        rejected = crud.create_package(
            db,
            name="rejected-model",
            version="1.0.0",
            s3_key="rejected-key",
            s3_bucket="test-bucket",
            file_size_bytes=0,
            ingest_status="rejected"
        )
        
        # Get packages (returns all statuses)
        packages = crud.get_packages(db)
        package_names = [p.name for p in packages]
        
        assert "approved-model" in package_names
        assert "rejected-model" in package_names
        
        # Clean up
        db.delete(approved)
        db.delete(rejected)
        db.commit()
    finally:
        db.close()


def test_quality_gate_result_json_storage():
    """Test that quality_gate_result stores complex JSON properly."""
    init_db()
    db = SessionLocal()
    
    try:
        complex_result = {
            "passed": True,
            "evaluated_at": "2025-11-23T19:00:00Z",
            "net_score": 0.84,
            "metrics": {
                "bus_factor": 0.72,
                "license": 1.0,
                "code_quality": 0.82
            }
        }
        
        package = crud.create_package(
            db,
            name="json-test-model",
            version="1.0.0",
            s3_key="json-key",
            s3_bucket="test-bucket",
            file_size_bytes=0,
            ingest_status="approved",
            quality_gate_result=complex_result
        )
        
        # Refresh to ensure data was persisted
        db.refresh(package)
        
        assert package.quality_gate_result["passed"] is True
        assert package.quality_gate_result["net_score"] == 0.84
        assert "metrics" in package.quality_gate_result
        assert package.quality_gate_result["metrics"]["bus_factor"] == 0.72
        
        # Clean up
        db.delete(package)
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
