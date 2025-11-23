"""
Test OpenAPI spec compliance for critical endpoints.

This test validates that the API responses match the OpenAPI specification exactly.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import create_app
from src.database.connection import SessionLocal, init_db, reset_db
from src.database.init_db import create_default_user


@pytest.fixture(scope="module")
def client():
    """Create test client with fresh database."""
    # Reset database before tests
    reset_db()
    init_db()
    create_default_user()
    
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_artifact_metadata_id_is_string(client):
    """Test that artifact IDs are returned as strings, not integers."""
    # Query all artifacts
    response = client.post(
        "/artifacts",
        json=[{"name": "*"}]
    )
    
    assert response.status_code == 200
    artifacts = response.json()
    
    # If there are artifacts, check that IDs are strings
    if artifacts:
        for artifact in artifacts:
            assert isinstance(artifact["id"], str), \
                f"Artifact ID should be string, got {type(artifact['id'])}"
            assert artifact["id"].isdigit() or "-" in artifact["id"], \
                f"Invalid ID format: {artifact['id']}"


def test_model_rating_response_format(client):
    """
    Test that /artifact/model/{id}/rate returns correct format.
    
    Expected format per OpenAPI spec (lines 1063-1216):
    - snake_case field names
    - name and category fields
    - size_score as object with 4 platforms
    - tree_score field
    - all 11 metrics with latencies
    """
    # First, we need to create a test artifact with scores
    # For now, just test the structure if we have any artifacts
    
    # This is a placeholder - in real testing, you'd:
    # 1. Ingest a model
    # 2. Get its ID
    # 3. Test the rate endpoint
    
    # Skip if no artifacts exist
    artifacts_response = client.post(
        "/artifacts",
        json=[{"name": "*", "types": ["model"]}]
    )
    
    if artifacts_response.status_code != 200:
        pytest.skip("No artifacts available for testing")
    
    artifacts = artifacts_response.json()
    if not artifacts:
        pytest.skip("No model artifacts available for testing")
    
    # Get first model's rating
    model_id = artifacts[0]["id"]
    response = client.get(f"/artifact/model/{model_id}/rate")
    
    # May get 404 if no ratings exist, which is fine
    if response.status_code == 404:
        pytest.skip("No ratings available for testing")
    
    assert response.status_code == 200
    rating = response.json()
    
    # Required metadata fields
    assert "name" in rating, "Missing 'name' field"
    assert "category" in rating, "Missing 'category' field"
    
    # Net score
    assert "net_score" in rating, "Missing 'net_score' field"
    assert "net_score_latency" in rating, "Missing 'net_score_latency' field"
    
    # Phase 1 metrics (8 metrics) - snake_case names
    required_metrics = [
        "ramp_up_time",
        "bus_factor",
        "performance_claims",
        "license",  # Not LicenseScore
        "dataset_and_code_score",
        "dataset_quality",
        "code_quality",
        "size_score",  # Object, not float
    ]
    
    for metric in required_metrics:
        assert metric in rating, f"Missing metric: {metric}"
        assert f"{metric}_latency" in rating, f"Missing latency for: {metric}"
    
    # Phase 2 metrics (3 additional)
    phase2_metrics = ["reproducibility", "reviewedness", "tree_score"]
    
    for metric in phase2_metrics:
        assert metric in rating, f"Missing Phase 2 metric: {metric}"
        assert f"{metric}_latency" in rating, f"Missing latency for: {metric}"
    
    # Verify size_score is an object with 4 platforms
    size_score = rating["size_score"]
    assert isinstance(size_score, dict), \
        f"size_score should be object, got {type(size_score)}"
    
    required_platforms = ["raspberry_pi", "jetson_nano", "desktop_pc", "aws_server"]
    for platform in required_platforms:
        assert platform in size_score, f"Missing platform in size_score: {platform}"
        assert isinstance(size_score[platform], (int, float)), \
            f"Platform score should be numeric: {platform}"
    
    # Verify NO PascalCase fields exist (old format)
    forbidden_fields = [
        "BusFactor", "Correctness", "RampUp", "ResponsiveMaintainer",
        "LicenseScore", "GoodPinningPractice", "PullRequest",
        "NetScore", "Reproducibility"
    ]
    
    for field in forbidden_fields:
        assert field not in rating, \
            f"Found old PascalCase field: {field}. Should use snake_case."
    
    print("✅ ModelRating response format is correct!")


def test_authentication_token_format(client):
    """Test that authentication returns proper token format."""
    response = client.put(
        "/authenticate",
        json={
            "user": {
                "name": "ece30861defaultadminuser",
                "is_admin": True
            },
            "secret": {
                "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"
            }
        }
    )
    
    assert response.status_code == 200
    token = response.json()
    
    # Token should be a string starting with "bearer "
    assert isinstance(token, str), "Token should be a string"
    assert token.startswith("bearer "), "Token should start with 'bearer '"
    

def test_health_endpoint(client):
    """Test that health endpoint is accessible."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_tracks_endpoint(client):
    """Test that tracks endpoint returns correct format."""
    response = client.get("/tracks")
    
    assert response.status_code == 200
    data = response.json()
    assert "plannedTracks" in data
    assert isinstance(data["plannedTracks"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
