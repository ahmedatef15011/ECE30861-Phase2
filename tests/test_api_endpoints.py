"""
API endpoint tests for Phase 2B FastAPI.

Tests all user, package, rating, and system endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.main import create_app
from src.api.dependencies import get_db
from src.database.models import Base
from src.auth.password_hash import hash_password


# Create test database
TEST_DATABASE_URL = "sqlite:///./test_api.db"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestSessionLocal()
        yield db
    finally:
        db.close()


# Create app and override dependencies
app = create_app()
app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)


@pytest.fixture(scope="function", autouse=True)
def setup_test_db():
    """Set up test database before each test."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_user(username="testuser", password="password123", is_admin=False, reset_first=True):
    """Helper to create a test user."""
    if reset_first:
        # Only reset on first call per test
        client.post("/api/v1/system/reset")
    
    # Login as default admin to create user
    admin_login = client.post(
        "/api/v1/user/login",
        json={
            "username": "ece30861defaultadminuser",
            "password": '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''
        }
    )
    
    if admin_login.status_code != 200:
        return None
        
    admin_token = admin_login.json()["access_token"]
    
    # Create user
    response = client.post(
        "/api/v1/user/register",
        headers={"X-Authorization": admin_token},
        json={
            "username": username,
            "email": f"{username}@example.com",
            "password": password,
            "is_admin": is_admin
        }
    )
    return response


def login_user(username="testuser", password="password123"):
    """Helper to login and get token."""
    response = client.post(
        "/api/v1/user/login",
        json={
            "username": username,
            "password": password
        }
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    return None


def create_test_package(token, name="test-model", version="1.0.0"):
    """Helper to create a test package."""
    response = client.post(
        "/api/v1/package",
        headers={"X-Authorization": token},
        json={
            "name": name,
            "version": version,
            "description": f"Test package {name}"
        }
    )
    return response


# ============================================================================
# System Endpoint Tests
# ============================================================================

def test_root_endpoint():
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "docs" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/api/v1/system/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "database_status" in data
    assert "uptime_seconds" in data


def test_system_reset():
    """Test system reset creates default admin."""
    response = client.post("/api/v1/system/reset")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["default_admin_created"] is True
    
    # Verify default admin can login
    login_response = client.post(
        "/api/v1/user/login",
        json={
            "username": "ece30861defaultadminuser",
            "password": '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''
        }
    )
    assert login_response.status_code == 200
    assert "access_token" in login_response.json()


# ============================================================================
# User Endpoint Tests
# ============================================================================

def test_register_user():
    """Test user registration by admin."""
    # Reset to create default admin
    client.post("/api/v1/system/reset")
    
    # Login as admin
    admin_token = login_user(
        "ece30861defaultadminuser",
        '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''
    )
    
    # Register new user
    response = client.post(
        "/api/v1/user/register",
        headers={"X-Authorization": admin_token},
        json={
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "securepass123",
            "is_admin": False
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "newuser"
    assert data["email"] == "newuser@example.com"
    assert data["is_admin"] is False
    assert "id" in data
    assert "created_at" in data


def test_register_duplicate_user():
    """Test registering duplicate username fails."""
    create_test_user("duplicate")
    
    # Try to create another user with same username
    admin_token = login_user(
        "ece30861defaultadminuser",
        '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''
    )
    
    response = client.post(
        "/api/v1/user/register",
        headers={"X-Authorization": admin_token},
        json={
            "username": "duplicate",
            "email": "another@example.com",
            "password": "password123",
            "is_admin": False
        }
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"].lower()


def test_register_without_admin():
    """Test non-admin cannot register users."""
    create_test_user("normaluser")
    token = login_user("normaluser")
    
    response = client.post(
        "/api/v1/user/register",
        headers={"X-Authorization": token},
        json={
            "username": "unauthorized",
            "email": "unauthorized@example.com",
            "password": "password123",
            "is_admin": False
        }
    )
    assert response.status_code == 403


def test_login_success():
    """Test successful user login."""
    create_test_user("logintest", "mypassword")
    
    response = client.post(
        "/api/v1/user/login",
        json={
            "username": "logintest",
            "password": "mypassword"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "user" in data
    assert data["user"]["username"] == "logintest"


def test_login_invalid_password():
    """Test login with invalid password fails."""
    create_test_user("logintest", "correctpass")
    
    response = client.post(
        "/api/v1/user/login",
        json={
            "username": "logintest",
            "password": "wrongpass"
        }
    )
    assert response.status_code == 401


def test_login_nonexistent_user():
    """Test login with nonexistent user fails."""
    response = client.post(
        "/api/v1/user/login",
        json={
            "username": "doesnotexist",
            "password": "password123"
        }
    )
    assert response.status_code == 401


def test_get_current_user():
    """Test getting current user information."""
    create_test_user("metest")
    token = login_user("metest")
    
    response = client.get(
        "/api/v1/user/me",
        headers={"X-Authorization": token}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "metest"


def test_delete_own_user():
    """Test user can delete their own account."""
    create_test_user("deleteme")
    token = login_user("deleteme")
    
    response = client.delete(
        "/api/v1/user/deleteme",
        headers={"X-Authorization": token}
    )
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]


def test_delete_other_user_as_non_admin():
    """Test non-admin cannot delete other users."""
    create_test_user("user1", reset_first=True)
    create_test_user("user2", reset_first=False)
    token = login_user("user1")
    
    response = client.delete(
        "/api/v1/user/user2",
        headers={"X-Authorization": token}
    )
    assert response.status_code == 403


def test_delete_user_as_admin():
    """Test admin can delete any user."""
    create_test_user("victim")
    admin_token = login_user(
        "ece30861defaultadminuser",
        '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''
    )
    
    response = client.delete(
        "/api/v1/user/victim",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 200


# ============================================================================
# Package Endpoint Tests
# ============================================================================

def test_create_package():
    """Test creating a new package."""
    create_test_user("pkguser")
    token = login_user("pkguser")
    
    response = create_test_package(token, "my-model", "1.0.0")
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "my-model"
    assert data["version"] == "1.0.0"
    assert "id" in data
    assert "uploaded_at" in data


def test_create_duplicate_package():
    """Test creating duplicate package fails."""
    create_test_user("pkguser")
    token = login_user("pkguser")
    
    # Create first package
    create_test_package(token, "duplicate-pkg", "1.0.0")
    
    # Try to create duplicate
    response = create_test_package(token, "duplicate-pkg", "1.0.0")
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"].lower()


def test_create_package_without_auth():
    """Test creating package without authentication fails."""
    response = client.post(
        "/api/v1/package",
        json={
            "name": "unauthorized-pkg",
            "version": "1.0.0",
            "description": "Should fail"
        }
    )
    # Accept either 401 (Unauthorized) or 403 (Forbidden)
    assert response.status_code in [401, 403]


def test_get_package_by_id():
    """Test retrieving package by ID."""
    create_test_user("pkguser")
    token = login_user("pkguser")
    
    # Create package
    create_response = create_test_package(token, "get-test", "1.0.0")
    package_id = create_response.json()["id"]
    
    # Get package
    response = client.get(f"/api/v1/package/{package_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == package_id
    assert data["name"] == "get-test"
    assert data["version"] == "1.0.0"


def test_get_nonexistent_package():
    """Test getting nonexistent package returns 404."""
    response = client.get("/api/v1/package/99999")
    assert response.status_code == 404


def test_list_packages():
    """Test listing packages with pagination."""
    create_test_user("pkguser")
    token = login_user("pkguser")
    
    # Create multiple packages
    create_test_package(token, "pkg1", "1.0.0")
    create_test_package(token, "pkg2", "1.0.0")
    create_test_package(token, "pkg3", "1.0.0")
    
    # List packages
    response = client.get("/api/v1/package?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert "packages" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert len(data["packages"]) >= 3


def test_list_packages_pagination():
    """Test package list pagination."""
    create_test_user("pkguser")
    token = login_user("pkguser")
    
    # Create packages
    for i in range(5):
        create_test_package(token, f"pkg{i}", "1.0.0")
    
    # Get page 1 with size 2
    response = client.get("/api/v1/package?page=1&page_size=2")
    assert response.status_code == 200
    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 2


def test_delete_package():
    """Test deleting a package."""
    create_test_user("pkguser")
    token = login_user("pkguser")
    
    # Create package
    create_response = create_test_package(token, "delete-me", "1.0.0")
    package_id = create_response.json()["id"]
    
    # Delete package
    response = client.delete(
        f"/api/v1/package/{package_id}",
        headers={"X-Authorization": token}
    )
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"]
    
    # Verify package is gone
    get_response = client.get(f"/api/v1/package/{package_id}")
    assert get_response.status_code == 404


def test_delete_package_unauthorized():
    """Test non-owner cannot delete package."""
    create_test_user("owner", reset_first=True)
    create_test_user("other", reset_first=False)
    
    owner_token = login_user("owner")
    other_token = login_user("other")
    
    # Owner creates package
    create_response = create_test_package(owner_token, "protected", "1.0.0")
    package_id = create_response.json()["id"]
    
    # Other user tries to delete
    response = client.delete(
        f"/api/v1/package/{package_id}",
        headers={"X-Authorization": other_token}
    )
    assert response.status_code == 403


def test_admin_can_delete_any_package():
    """Test admin can delete any package."""
    create_test_user("owner")
    owner_token = login_user("owner")
    
    # Owner creates package
    create_response = create_test_package(owner_token, "admin-test", "1.0.0")
    package_id = create_response.json()["id"]
    
    # Admin deletes it
    admin_token = login_user(
        "ece30861defaultadminuser",
        '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''
    )
    response = client.delete(
        f"/api/v1/package/{package_id}",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 200


# ============================================================================
# Rating Endpoint Tests
# ============================================================================

def test_rate_package():
    """Test rating a package."""
    create_test_user("rater")
    token = login_user("rater")
    
    # Create package
    create_response = create_test_package(token, "rate-me", "1.0.0")
    package_id = create_response.json()["id"]
    
    # Rate package
    response = client.post(
        f"/api/v1/package/{package_id}/rate",
        headers={"X-Authorization": token},
        json={
            "ramp_up_time": 0.8,
            "bus_factor": 0.7,
            "performance_claims": 0.9,
            "license_score": 1.0,
            "size_score": 0.75,
            "dataset_quality": 0.8,
            "dataset_code_linkage": 0.85,
            "code_quality": 0.82,
            "reproducibility": 1.0,
            "reviewedness": 0.9,
            "treescore": 0.85,
            "net_score": 0.82
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["package_id"] == package_id
    assert data["ramp_up_time"] == 0.8
    assert data["net_score"] == 0.82


def test_rate_nonexistent_package():
    """Test rating nonexistent package fails."""
    create_test_user("rater")
    token = login_user("rater")
    
    response = client.post(
        "/api/v1/package/99999/rate",
        headers={"X-Authorization": token},
        json={
            "ramp_up_score": 0.8,
            "correctness_score": 0.9,
            "bus_factor_score": 0.7,
            "responsive_maintainer_score": 0.85,
            "license_score": 1.0,
            "good_pinning_practice_score": 0.75,
            "pullrequest_score": 0.8,
            "net_score": 0.82
        }
    )
    assert response.status_code == 404


def test_rate_package_invalid_scores():
    """Test rating with invalid score values fails."""
    create_test_user("rater")
    token = login_user("rater")
    
    # Create package
    create_response = create_test_package(token, "rate-me", "1.0.0")
    package_id = create_response.json()["id"]
    
    # Try to rate with invalid score (>1.0)
    response = client.post(
        f"/api/v1/package/{package_id}/rate",
        headers={"X-Authorization": token},
        json={
            "ramp_up_time": 1.5,  # Invalid!
            "bus_factor": 0.7,
            "performance_claims": 0.9,
            "license_score": 1.0,
            "size_score": 0.75,
            "dataset_quality": 0.8,
            "dataset_code_linkage": 0.85,
            "code_quality": 0.82,
            "net_score": 0.82
        }
    )
    assert response.status_code == 422  # Validation error


# ============================================================================
# Authentication & Authorization Tests
# ============================================================================

def test_access_protected_endpoint_without_token():
    """Test accessing protected endpoint without token fails."""
    response = client.post(
        "/api/v1/package",
        json={
            "name": "unauthorized",
            "version": "1.0.0"
        }
    )
    # Accept either 401 (Unauthorized) or 403 (Forbidden)
    assert response.status_code in [401, 403]


def test_access_with_invalid_token():
    """Test accessing with invalid token fails."""
    response = client.post(
        "/api/v1/package",
        headers={"X-Authorization": "invalid_token_here"},
        json={
            "name": "unauthorized",
            "version": "1.0.0"
        }
    )
    assert response.status_code == 401


def test_access_with_expired_token():
    """Test that token verification works."""
    # This is a basic test - full expiration testing would require
    # mocking time or waiting for actual expiration
    response = client.post(
        "/api/v1/package",
        headers={"X-Authorization": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid"},
        json={
            "name": "unauthorized",
            "version": "1.0.0"
        }
    )
    assert response.status_code == 401


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

def test_special_characters_in_password():
    """Test that special characters in password work (like default admin)."""
    # This tests the security issue mentioned in spec
    # First reset to create default admin
    client.post("/api/v1/system/reset")
    
    admin_token = login_user(
        "ece30861defaultadminuser",
        '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''
    )
    assert admin_token is not None


def test_empty_package_list():
    """Test listing packages when none exist."""
    response = client.get("/api/v1/package?page=1&page_size=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["packages"], list)


def test_invalid_page_number():
    """Test invalid page number is rejected."""
    response = client.get("/api/v1/package?page=0&page_size=10")
    assert response.status_code == 422  # Validation error


def test_invalid_page_size():
    """Test invalid page size is rejected."""
    response = client.get("/api/v1/package?page=1&page_size=1000")
    assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
