"""
Tests for lineage API endpoint.

Tests the GET /artifact/model/{id}/lineage endpoint which returns
lineage information from the database (ingested packages only).
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.main import create_app
from src.api.dependencies import get_db
from src.database.models import Base, Package, User
from src.database import crud
from src.auth.password_hash import hash_password


# Create test database
TEST_DATABASE_URL = "sqlite:///./test_lineage_api.db"
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
    
    # Create a test user
    db = TestSessionLocal()
    try:
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=hash_password("testpass123"),
            is_admin=True
        )
        db.add(user)
        db.commit()
    finally:
        db.close()
    
    yield
    
    Base.metadata.drop_all(bind=test_engine)


def get_auth_token():
    """Helper to get authentication token."""
    response = client.put(
        "/authenticate",
        json={
            "user": {"name": "testuser", "is_admin": True},
            "secret": {"password": "testpass123"}
        }
    )
    assert response.status_code == 200
    return response.json()


def test_lineage_endpoint_single_package_no_relationships():
    """Test lineage endpoint with a single package with no relationships."""
    token = get_auth_token()
    
    # Create a package
    db = TestSessionLocal()
    try:
        package = crud.create_package(
            db,
            name="standalone-model",
            version="1.0.0",
            artifact_type="model",
            s3_key="test-key",
            s3_bucket="test-bucket",
            file_size_bytes=1000,
            source_url="https://huggingface.co/test/standalone-model",
            uploaded_by=1
        )
        package_id = package.id
        db.commit()
    finally:
        db.close()
    
    # Get lineage
    response = client.get(
        f"/artifact/model/{package_id}/lineage",
        headers={"X-Authorization": token}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should have exactly 1 node (the package itself) and no edges
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 1
    assert len(data["edges"]) == 0
    
    # Check node structure
    node = data["nodes"][0]
    assert node["artifact_id"] == str(package_id)
    assert node["name"] == "standalone-model"
    assert node["source"] == "database"


def test_lineage_endpoint_with_parent_relationships():
    """Test lineage endpoint with parent (dependency) relationships."""
    token = get_auth_token()
    
    db = TestSessionLocal()
    try:
        # Create parent package
        parent = crud.create_package(
            db,
            name="base-model",
            version="1.0.0",
            artifact_type="model",
            s3_key="parent-key",
            s3_bucket="test-bucket",
            file_size_bytes=1000,
            source_url="https://huggingface.co/test/base-model",
            uploaded_by=1
        )
        db.commit()
        
        # Create child package
        child = crud.create_package(
            db,
            name="fine-tuned-model",
            version="1.0.0",
            artifact_type="model",
            s3_key="child-key",
            s3_bucket="test-bucket",
            file_size_bytes=1000,
            source_url="https://huggingface.co/test/fine-tuned-model",
            uploaded_by=1
        )
        db.commit()
        
        # Create lineage relationship
        crud.create_lineage(
            db,
            parent_package_id=parent.id,
            child_package_id=child.id,
            relationship_type="fine_tuned_from"
        )
        db.commit()
        
        child_id = child.id
        parent_id = parent.id
    finally:
        db.close()
    
    # Get lineage for child package
    response = client.get(
        f"/artifact/model/{child_id}/lineage",
        headers={"X-Authorization": token}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should have 2 nodes (parent + child) and 1 edge
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
    
    # Check nodes
    node_ids = {node["artifact_id"] for node in data["nodes"]}
    assert str(parent_id) in node_ids
    assert str(child_id) in node_ids
    
    # Check edge
    edge = data["edges"][0]
    assert edge["from_node_artifact_id"] == str(parent_id)
    assert edge["to_node_artifact_id"] == str(child_id)
    assert edge["relationship"] == "fine_tuned_from"


def test_lineage_endpoint_with_child_relationships():
    """Test lineage endpoint with child (dependent) relationships."""
    token = get_auth_token()
    
    db = TestSessionLocal()
    try:
        # Create parent package
        parent = crud.create_package(
            db,
            name="base-model",
            version="1.0.0",
            artifact_type="model",
            s3_key="parent-key",
            s3_bucket="test-bucket",
            file_size_bytes=1000,
            source_url="https://huggingface.co/test/base-model",
            uploaded_by=1
        )
        db.commit()
        
        # Create child package
        child = crud.create_package(
            db,
            name="derived-model",
            version="1.0.0",
            artifact_type="model",
            s3_key="child-key",
            s3_bucket="test-bucket",
            file_size_bytes=1000,
            source_url="https://huggingface.co/test/derived-model",
            uploaded_by=1
        )
        db.commit()
        
        # Create lineage relationship
        crud.create_lineage(
            db,
            parent_package_id=parent.id,
            child_package_id=child.id,
            relationship_type="derived_from"
        )
        db.commit()
        
        parent_id = parent.id
        child_id = child.id
    finally:
        db.close()
    
    # Get lineage for parent package (should show child as dependent)
    response = client.get(
        f"/artifact/model/{parent_id}/lineage",
        headers={"X-Authorization": token}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should have 2 nodes and 1 edge
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
    
    # Check edge goes from parent to child
    edge = data["edges"][0]
    assert edge["from_node_artifact_id"] == str(parent_id)
    assert edge["to_node_artifact_id"] == str(child_id)
    assert edge["relationship"] == "derived_from"


def test_lineage_endpoint_complex_graph():
    """Test lineage endpoint with multiple parents and children."""
    token = get_auth_token()
    
    db = TestSessionLocal()
    try:
        # Create a diamond-shaped dependency graph:
        # grandparent -> parent1, parent2
        # parent1, parent2 -> child
        
        grandparent = crud.create_package(
            db,
            name="grandparent-model",
            version="1.0.0",
            artifact_type="model",
            s3_key="gp-key",
            s3_bucket="test-bucket",
            file_size_bytes=1000,
            source_url="https://huggingface.co/test/grandparent",
            uploaded_by=1
        )
        db.commit()
        
        parent1 = crud.create_package(
            db,
            name="parent1-model",
            version="1.0.0",
            artifact_type="model",
            s3_key="p1-key",
            s3_bucket="test-bucket",
            file_size_bytes=1000,
            source_url="https://huggingface.co/test/parent1",
            uploaded_by=1
        )
        db.commit()
        
        parent2 = crud.create_package(
            db,
            name="parent2-model",
            version="1.0.0",
            artifact_type="model",
            s3_key="p2-key",
            s3_bucket="test-bucket",
            file_size_bytes=1000,
            source_url="https://huggingface.co/test/parent2",
            uploaded_by=1
        )
        db.commit()
        
        child = crud.create_package(
            db,
            name="merged-model",
            version="1.0.0",
            artifact_type="model",
            s3_key="child-key",
            s3_bucket="test-bucket",
            file_size_bytes=1000,
            source_url="https://huggingface.co/test/merged",
            uploaded_by=1
        )
        db.commit()
        
        # Create relationships
        crud.create_lineage(db, grandparent.id, parent1.id, "trained_on")
        crud.create_lineage(db, grandparent.id, parent2.id, "trained_on")
        crud.create_lineage(db, parent1.id, child.id, "derived_from")
        crud.create_lineage(db, parent2.id, child.id, "derived_from")
        db.commit()
        
        child_id = child.id
    finally:
        db.close()
    
    # Get lineage for child (merged model)
    response = client.get(
        f"/artifact/model/{child_id}/lineage",
        headers={"X-Authorization": token}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should show child + its 2 parents (grandparent not directly connected)
    assert len(data["nodes"]) == 3  # child + parent1 + parent2
    assert len(data["edges"]) == 2  # parent1->child, parent2->child


def test_lineage_endpoint_nonexistent_package():
    """Test lineage endpoint with non-existent package ID."""
    token = get_auth_token()
    
    response = client.get(
        "/artifact/model/99999/lineage",
        headers={"X-Authorization": token}
    )
    
    assert response.status_code == 404


def test_lineage_endpoint_requires_authentication():
    """Test that lineage endpoint requires authentication."""
    # Create a package first
    db = TestSessionLocal()
    try:
        package = crud.create_package(
            db,
            name="test-model",
            version="1.0.0",
            artifact_type="model",
            s3_key="test-key",
            s3_bucket="test-bucket",
            file_size_bytes=1000,
            source_url="https://huggingface.co/test/model",
            uploaded_by=1
        )
        package_id = package.id
        db.commit()
    finally:
        db.close()
    
    # Try without authentication
    response = client.get(f"/artifact/model/{package_id}/lineage")
    
    assert response.status_code == 401  # Unauthorized


def test_lineage_endpoint_all_source_from_database():
    """Test that all lineage nodes have source='database'."""
    token = get_auth_token()
    
    db = TestSessionLocal()
    try:
        # Create packages with lineage
        parent = crud.create_package(
            db,
            name="parent",
            version="1.0.0",
            artifact_type="model",
            s3_key="p-key",
            s3_bucket="test-bucket",
            file_size_bytes=1000,
            source_url="https://huggingface.co/test/parent",
            uploaded_by=1
        )
        db.commit()
        
        child = crud.create_package(
            db,
            name="child",
            version="1.0.0",
            artifact_type="model",
            s3_key="c-key",
            s3_bucket="test-bucket",
            file_size_bytes=1000,
            source_url="https://huggingface.co/test/child",
            uploaded_by=1
        )
        db.commit()
        
        crud.create_lineage(db, parent.id, child.id, "fine_tuned_from")
        db.commit()
        
        child_id = child.id
    finally:
        db.close()
    
    response = client.get(
        f"/artifact/model/{child_id}/lineage",
        headers={"X-Authorization": token}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # All nodes should have source='database' (not from HuggingFace)
    for node in data["nodes"]:
        assert node["source"] == "database", \
            f"Node {node['artifact_id']} has source={node['source']}, expected 'database'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
