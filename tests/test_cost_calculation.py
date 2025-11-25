"""
Unit tests for artifact cost calculation endpoint.

Tests the three required example scenarios:
1. Content size = 512 bytes, dependency=false
2. Content size = 5120 bytes, dependency=true
3. Content size = 0 bytes, dependency=false (minimum enforced)
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from src.database import models
from src.api.main import app


class TestCostCalculation:
    """Test suite for artifact cost calculation."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_example_1_512_bytes_no_dependency(self, client):
        """
        Example 1: Content size = 512 bytes, dependency=false
        Expected: standalone_cost = max(1.0, 512/1024) = 1.0
                  total_cost = 1.0
        """
        # Create mock package with 512 bytes
        mock_package = Mock(spec=models.Package)
        mock_package.id = 1
        mock_package.name = "test-model-512b"
        mock_package.artifact_type = "model"
        mock_package.file_size_bytes = 512
        
        with patch('src.database.crud.get_package_by_id') as mock_get:
            mock_get.return_value = mock_package
            
            response = client.get(
                "/artifact/model/1/cost"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "1" in data
            assert "standalone_cost" in data["1"]
            assert "total_cost" in data["1"]
            
            # Verify calculations
            # standalone_cost = max(1.0, 512/1024) = max(1.0, 0.5) = 1.0
            assert data["1"]["standalone_cost"] == 1.0
            # total_cost = standalone_cost (no dependency)
            assert data["1"]["total_cost"] == 1.0
    
    def test_example_2_5120_bytes_with_dependency(self, client):
        """
        Example 2: Content size = 5120 bytes, dependency=true
        Expected: standalone_cost = max(1.0, 5120/1024) = 5.0
                  total_cost = 5.0 * 2.0 = 10.0
        """
        # Create mock package with 5120 bytes
        mock_package = Mock(spec=models.Package)
        mock_package.id = 2
        mock_package.name = "test-model-5120b"
        mock_package.artifact_type = "model"
        mock_package.file_size_bytes = 5120
        
        with patch('src.database.crud.get_package_by_id') as mock_get:
            mock_get.return_value = mock_package
            
            response = client.get(
                "/artifact/model/2/cost?dependency=true"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "2" in data
            assert "standalone_cost" in data["2"]
            assert "total_cost" in data["2"]
            
            # Verify calculations
            # standalone_cost = max(1.0, 5120/1024) = max(1.0, 5.0) = 5.0
            assert data["2"]["standalone_cost"] == 5.0
            # total_cost = standalone_cost * 2.0 = 5.0 * 2.0 = 10.0
            assert data["2"]["total_cost"] == 10.0
    
    def test_example_3_0_bytes_minimum_enforced(self, client):
        """
        Example 3: Content size = 0 bytes, dependency=false
        Expected: standalone_cost = max(1.0, 0/1024) = 1.0 (minimum enforced)
                  total_cost = 1.0
        """
        # Create mock package with 0 bytes (empty content)
        mock_package = Mock(spec=models.Package)
        mock_package.id = 3
        mock_package.name = "test-model-empty"
        mock_package.artifact_type = "model"
        mock_package.file_size_bytes = 0
        
        with patch('src.database.crud.get_package_by_id') as mock_get:
            mock_get.return_value = mock_package
            
            response = client.get(
                "/artifact/model/3/cost"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "3" in data
            assert "standalone_cost" in data["3"]
            assert "total_cost" in data["3"]
            
            # Verify calculations - minimum 1.0 KB enforced
            # standalone_cost = max(1.0, 0/1024) = max(1.0, 0.0) = 1.0
            assert data["3"]["standalone_cost"] == 1.0
            # total_cost = standalone_cost (no dependency)
            assert data["3"]["total_cost"] == 1.0
    
    def test_rounding_to_2_decimal_places(self, client):
        """Test that costs are rounded to 2 decimal places."""
        # Create mock package with size that produces decimal result
        mock_package = Mock(spec=models.Package)
        mock_package.id = 4
        mock_package.name = "test-model-decimal"
        mock_package.artifact_type = "model"
        mock_package.file_size_bytes = 1536  # 1536/1024 = 1.5 KB
        
        with patch('src.database.crud.get_package_by_id') as mock_get:
            mock_get.return_value = mock_package
            
            response = client.get(
                "/artifact/model/4/cost"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify rounding
            assert data["4"]["standalone_cost"] == 1.5
            assert data["4"]["total_cost"] == 1.5
    
    def test_artifact_not_found(self, client):
        """Test 404 when artifact doesn't exist."""
        with patch('src.database.crud.get_package_by_id') as mock_get:
            mock_get.return_value = None  # Artifact not found
            
            response = client.get(
                "/artifact/model/999/cost"
            )
            
            assert response.status_code == 404
            assert "does not exist" in response.json()["detail"].lower()
    
    def test_always_returns_both_fields(self, client):
        """Test that both standalone_cost and total_cost are always returned."""
        mock_package = Mock(spec=models.Package)
        mock_package.id = 5
        mock_package.name = "test-model"
        mock_package.artifact_type = "model"
        mock_package.file_size_bytes = 2048
        
        with patch('src.database.crud.get_package_by_id') as mock_get:
            mock_get.return_value = mock_package
            
            # Test without dependency
            response = client.get(
                "/artifact/model/5/cost"
            )
            assert response.status_code == 200
            data = response.json()
            assert "standalone_cost" in data["5"]
            assert "total_cost" in data["5"]
            
            # Test with dependency
            response = client.get(
                "/artifact/model/5/cost?dependency=true"
            )
            assert response.status_code == 200
            data = response.json()
            assert "standalone_cost" in data["5"]
            assert "total_cost" in data["5"]
    
    def test_dependency_doubles_cost(self, client):
        """Test that dependency=true doubles the total_cost."""
        mock_package = Mock(spec=models.Package)
        mock_package.id = 6
        mock_package.name = "test-model-deps"
        mock_package.artifact_type = "model"
        mock_package.file_size_bytes = 3072  # 3 KB
        
        with patch('src.database.crud.get_package_by_id') as mock_get:
            mock_get.return_value = mock_package
            
            # Without dependency
            response = client.get(
                "/artifact/model/6/cost"
            )
            data = response.json()
            standalone_cost = data["6"]["standalone_cost"]
            total_cost_no_deps = data["6"]["total_cost"]
            
            # With dependency
            response = client.get(
                "/artifact/model/6/cost?dependency=true"
            )
            data = response.json()
            total_cost_with_deps = data["6"]["total_cost"]
            
            # Verify total_cost with deps is double the standalone cost
            assert total_cost_with_deps == standalone_cost * 2.0
            assert total_cost_no_deps == standalone_cost
    
    def test_different_artifact_types(self, client):
        """Test cost calculation works for model, dataset, and code types."""
        for artifact_type in ["model", "dataset", "code"]:
            mock_package = Mock(spec=models.Package)
            mock_package.id = 7
            mock_package.name = f"test-{artifact_type}"
            mock_package.artifact_type = artifact_type
            mock_package.file_size_bytes = 2048
            
            with patch('src.database.crud.get_package_by_id') as mock_get:
                mock_get.return_value = mock_package
                
                response = client.get(
                    f"/artifact/{artifact_type}/7/cost"
                )
                
                assert response.status_code == 200
                data = response.json()
                assert "standalone_cost" in data["7"]
                assert "total_cost" in data["7"]
                assert data["7"]["standalone_cost"] == 2.0  # 2048/1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
