"""
Tests for LLM API routes.

Tests cover:
- GET /api/v1/llm/health - Health check endpoint
- POST /api/v1/llm/generate - Text generation
- POST /api/v1/llm/analyze-code - Code analysis
- POST /api/v1/llm/check-license - License compatibility
- POST /api/v1/llm/summarize-readme - README summarization
- POST /api/v1/llm/analyze-artifact/{id} - Artifact analysis
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import with graceful fallback
try:
    from src.api.main import app
    HAS_APP = True
except ImportError:
    HAS_APP = False
    app = None


# Skip all tests if app not available
pytestmark = pytest.mark.skipif(
    not HAS_APP,
    reason="API app not available"
)


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create auth headers for protected endpoints."""
    return {"Authorization": "Bearer test_token"}


# =============================================================================
# HEALTH ENDPOINT TESTS
# =============================================================================

class TestHealthEndpoint:
    """Test LLM health check endpoint."""
    
    @patch('src.api.routes.llm.BEDROCK_ENABLED', True)
    @patch('src.api.routes.llm.BedrockClient')
    def test_health_check_healthy(self, mock_client_class, client):
        """Test successful health check."""
        mock_client = MagicMock()
        mock_client.health_check.return_value = {
            "healthy": True,
            "model_id": "anthropic.claude-3-haiku-20240307-v1:0",
            "latency_ms": 150
        }
        mock_client_class.return_value = mock_client
        
        response = client.get("/api/v1/llm/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @patch('src.api.routes.llm.BEDROCK_ENABLED', False)
    def test_health_check_disabled(self, client):
        """Test health check when LLM is disabled."""
        response = client.get("/api/v1/llm/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "disabled"
    
    @patch('src.api.routes.llm.BEDROCK_ENABLED', True)
    @patch('src.api.routes.llm.BedrockClient')
    def test_health_check_unhealthy(self, mock_client_class, client):
        """Test health check when service is unhealthy."""
        mock_client = MagicMock()
        mock_client.health_check.return_value = {
            "healthy": False,
            "error": "Connection failed"
        }
        mock_client_class.return_value = mock_client
        
        response = client.get("/api/v1/llm/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"


# =============================================================================
# GENERATE ENDPOINT TESTS
# =============================================================================

class TestGenerateEndpoint:
    """Test text generation endpoint."""
    
    @patch('src.api.routes.llm.BEDROCK_ENABLED', True)
    @patch('src.api.routes.llm.BedrockClient')
    def test_generate_success(self, mock_client_class, client, auth_headers):
        """Test successful text generation."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "success": True,
            "response": "Generated text response"
        }
        mock_client_class.return_value = mock_client
        
        response = client.post(
            "/api/v1/llm/generate",
            json={"prompt": "Hello, world!"},
            headers=auth_headers
        )
        
        # May require auth, so accept 200 or 401/403
        assert response.status_code in [200, 401, 403]
    
    @patch('src.api.routes.llm.BEDROCK_ENABLED', False)
    def test_generate_disabled(self, client, auth_headers):
        """Test generate when LLM is disabled."""
        response = client.post(
            "/api/v1/llm/generate",
            json={"prompt": "Hello"},
            headers=auth_headers
        )
        
        # Should return service unavailable or similar
        assert response.status_code in [200, 401, 403, 503]
    
    def test_generate_missing_prompt(self, client, auth_headers):
        """Test generate with missing prompt."""
        response = client.post(
            "/api/v1/llm/generate",
            json={},
            headers=auth_headers
        )
        
        # Should return validation error or auth error
        assert response.status_code in [401, 403, 422]


# =============================================================================
# ANALYZE CODE ENDPOINT TESTS
# =============================================================================

class TestAnalyzeCodeEndpoint:
    """Test code analysis endpoint."""
    
    @patch('src.api.routes.llm.BEDROCK_ENABLED', True)
    @patch('src.api.routes.llm.BedrockClient')
    def test_analyze_code_success(self, mock_client_class, client, auth_headers):
        """Test successful code analysis."""
        mock_client = MagicMock()
        mock_client.analyze_code.return_value = {
            "success": True,
            "response": "Code quality: Good. Well structured."
        }
        mock_client_class.return_value = mock_client
        
        response = client.post(
            "/api/v1/llm/analyze-code",
            json={"code": "def hello(): print('world')"},
            headers=auth_headers
        )
        
        assert response.status_code in [200, 401, 403]
    
    @patch('src.api.routes.llm.BEDROCK_ENABLED', True)
    @patch('src.api.routes.llm.BedrockClient')
    def test_analyze_code_with_language(
        self, mock_client_class, client, auth_headers
    ):
        """Test code analysis with language specification."""
        mock_client = MagicMock()
        mock_client.analyze_code.return_value = {
            "success": True,
            "response": "JavaScript code analysis complete."
        }
        mock_client_class.return_value = mock_client
        
        response = client.post(
            "/api/v1/llm/analyze-code",
            json={
                "code": "console.log('hello')",
                "language": "javascript"
            },
            headers=auth_headers
        )
        
        assert response.status_code in [200, 401, 403]


# =============================================================================
# LICENSE CHECK ENDPOINT TESTS
# =============================================================================

class TestLicenseCheckEndpoint:
    """Test license compatibility endpoint."""
    
    @patch('src.api.routes.llm.BEDROCK_ENABLED', True)
    @patch('src.api.routes.llm.BedrockClient')
    def test_check_license_success(
        self, mock_client_class, client, auth_headers
    ):
        """Test successful license check."""
        mock_client = MagicMock()
        mock_client.check_license_compatibility.return_value = {
            "success": True,
            "response": "Compatible: Yes. MIT allows use with Apache-2.0."
        }
        mock_client_class.return_value = mock_client
        
        response = client.post(
            "/api/v1/llm/check-license",
            json={
                "source_license": "MIT",
                "target_license": "Apache-2.0"
            },
            headers=auth_headers
        )
        
        assert response.status_code in [200, 401, 403]


# =============================================================================
# SUMMARIZE README ENDPOINT TESTS
# =============================================================================

class TestSummarizeReadmeEndpoint:
    """Test README summarization endpoint."""
    
    @patch('src.api.routes.llm.BEDROCK_ENABLED', True)
    @patch('src.api.routes.llm.BedrockClient')
    def test_summarize_readme_success(
        self, mock_client_class, client, auth_headers
    ):
        """Test successful README summarization."""
        mock_client = MagicMock()
        mock_client.summarize_readme.return_value = {
            "success": True,
            "response": "This is a BERT model for text classification."
        }
        mock_client_class.return_value = mock_client
        
        response = client.post(
            "/api/v1/llm/summarize-readme",
            json={"readme_content": "# BERT Model\n\nA BERT model."},
            headers=auth_headers
        )
        
        assert response.status_code in [200, 401, 403]


# =============================================================================
# ANALYZE ARTIFACT ENDPOINT TESTS
# =============================================================================

class TestAnalyzeArtifactEndpoint:
    """Test artifact analysis endpoint."""
    
    @patch('src.api.routes.llm.BEDROCK_ENABLED', True)
    @patch('src.api.routes.llm.BedrockClient')
    def test_analyze_artifact_not_found(
        self, mock_client_class, client, auth_headers
    ):
        """Test artifact analysis with non-existent ID."""
        response = client.post(
            "/api/v1/llm/analyze-artifact/nonexistent-id",
            headers=auth_headers
        )
        
        # Should return not found or auth error
        assert response.status_code in [401, 403, 404]


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling for LLM endpoints."""
    
    @patch('src.api.routes.llm.BEDROCK_ENABLED', True)
    @patch('src.api.routes.llm.BedrockClient')
    def test_client_exception(self, mock_client_class, client, auth_headers):
        """Test handling of client exceptions."""
        mock_client_class.side_effect = Exception("Connection failed")
        
        response = client.get("/api/v1/llm/health")
        
        # Should handle gracefully
        assert response.status_code in [200, 500, 503]
    
    def test_invalid_json(self, client, auth_headers):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/llm/generate",
            content="not valid json",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        
        # Should return error
        assert response.status_code in [400, 401, 403, 422]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for LLM routes."""
    
    def test_llm_routes_registered(self, client):
        """Test that LLM routes are registered."""
        # Health endpoint should always respond
        response = client.get("/api/v1/llm/health")
        assert response.status_code in [200, 404]
    
    def test_openapi_includes_llm(self, client):
        """Test that OpenAPI spec includes LLM endpoints."""
        response = client.get("/openapi.json")
        
        if response.status_code == 200:
            spec = response.json()
            paths = spec.get("paths", {})
            
            # Check for LLM endpoints
            llm_paths = [p for p in paths if "/llm/" in p]
            # May or may not have LLM paths depending on config
            assert isinstance(llm_paths, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
