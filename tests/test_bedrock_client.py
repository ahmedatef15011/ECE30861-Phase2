"""
Tests for AWS Bedrock LLM client.

Tests cover:
- Client initialization
- Text generation
- Code analysis
- License compatibility checking
- README summarization
- Health checks
- Error handling
"""

import pytest
from unittest.mock import patch, MagicMock

# Import with graceful fallback
try:
    from src.llm.bedrock_client import BedrockClient, BedrockConfig
    HAS_CLIENT = True
except ImportError:
    HAS_CLIENT = False
    BedrockClient = None
    BedrockConfig = None


# Skip all tests if client module not available
pytestmark = pytest.mark.skipif(
    not HAS_CLIENT,
    reason="Bedrock client module not available"
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_boto3_client():
    """Create a mock boto3 Bedrock client."""
    with patch('src.llm.bedrock_client.boto3') as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        yield mock_client


@pytest.fixture
def bedrock_config():
    """Create a default Bedrock config."""
    if BedrockConfig:
        return BedrockConfig()
    return None


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestBedrockConfig:
    """Test BedrockConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BedrockConfig()
        
        assert config.region == "us-east-1"
        assert config.model_id == "anthropic.claude-3-haiku-20240307-v1:0"
        assert config.max_tokens == 4096
        assert config.temperature == 0.3
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BedrockConfig(
            region="us-west-2",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=2048,
            temperature=0.5
        )
        
        assert config.region == "us-west-2"
        assert config.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert config.max_tokens == 2048
        assert config.temperature == 0.5


# =============================================================================
# CLIENT INITIALIZATION TESTS
# =============================================================================

class TestClientInitialization:
    """Test BedrockClient initialization."""
    
    @patch('src.llm.bedrock_client.boto3')
    def test_init_with_default_config(self, mock_boto3):
        """Test initialization with default configuration."""
        client = BedrockClient()
        
        assert client is not None
        mock_boto3.client.assert_called_once()
    
    @patch('src.llm.bedrock_client.boto3')
    def test_init_with_custom_config(self, mock_boto3):
        """Test initialization with custom configuration."""
        config = BedrockConfig(region="eu-west-1")
        client = BedrockClient(config=config)
        
        assert client is not None
    
    @patch('src.llm.bedrock_client.boto3')
    def test_init_failure_handling(self, mock_boto3):
        """Test handling of initialization failures."""
        mock_boto3.client.side_effect = Exception("AWS credentials not found")
        
        with pytest.raises(Exception):
            BedrockClient()


# =============================================================================
# TEXT GENERATION TESTS
# =============================================================================

class TestTextGeneration:
    """Test text generation functionality."""
    
    @patch('src.llm.bedrock_client.boto3')
    def test_successful_generation(self, mock_boto3):
        """Test successful text generation."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        # Mock successful response
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = b'{"content": [{"text": "Generated text"}]}'
        mock_client.invoke_model.return_value = mock_response
        
        client = BedrockClient()
        result = client.generate("Test prompt")
        
        assert result["success"] is True
        assert "response" in result
    
    @patch('src.llm.bedrock_client.boto3')
    def test_generation_with_system_prompt(self, mock_boto3):
        """Test generation with system prompt."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = b'{"content": [{"text": "Response"}]}'
        mock_client.invoke_model.return_value = mock_response
        
        client = BedrockClient()
        result = client.generate(
            "Test prompt",
            system_prompt="You are a helpful assistant."
        )
        
        # Verify invoke_model was called
        mock_client.invoke_model.assert_called()
    
    @patch('src.llm.bedrock_client.boto3')
    def test_generation_failure(self, mock_boto3):
        """Test handling of generation failures."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.side_effect = Exception("Model unavailable")
        
        client = BedrockClient()
        result = client.generate("Test prompt")
        
        assert result["success"] is False
        assert "error" in result


# =============================================================================
# CODE ANALYSIS TESTS
# =============================================================================

class TestCodeAnalysis:
    """Test code analysis functionality."""
    
    @patch('src.llm.bedrock_client.boto3')
    def test_analyze_code_success(self, mock_boto3):
        """Test successful code analysis."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = b'{"content": [{"text": "Code analysis: Good quality code."}]}'
        mock_client.invoke_model.return_value = mock_response
        
        client = BedrockClient()
        result = client.analyze_code("def hello(): print('world')")
        
        assert result["success"] is True
    
    @patch('src.llm.bedrock_client.boto3')
    def test_analyze_code_with_language(self, mock_boto3):
        """Test code analysis with language specification."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = b'{"content": [{"text": "Analysis complete."}]}'
        mock_client.invoke_model.return_value = mock_response
        
        client = BedrockClient()
        result = client.analyze_code(
            "console.log('hello')",
            language="javascript"
        )
        
        assert result["success"] is True


# =============================================================================
# LICENSE CHECKING TESTS
# =============================================================================

class TestLicenseChecking:
    """Test license compatibility checking."""
    
    @patch('src.llm.bedrock_client.boto3')
    def test_check_license_compatibility(self, mock_boto3):
        """Test license compatibility checking."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = b'{"content": [{"text": "Compatible: Yes. MIT is permissive."}]}'
        mock_client.invoke_model.return_value = mock_response
        
        client = BedrockClient()
        result = client.check_license_compatibility("MIT", "Apache-2.0")
        
        assert result["success"] is True


# =============================================================================
# README SUMMARIZATION TESTS
# =============================================================================

class TestReadmeSummarization:
    """Test README summarization functionality."""
    
    @patch('src.llm.bedrock_client.boto3')
    def test_summarize_readme(self, mock_boto3):
        """Test README summarization."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = b'{"content": [{"text": "This is a BERT model for text classification."}]}'
        mock_client.invoke_model.return_value = mock_response
        
        client = BedrockClient()
        readme = "# BERT Model\n\nA fine-tuned BERT for classification."
        result = client.summarize_readme(readme)
        
        assert result["success"] is True


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================

class TestHealthCheck:
    """Test health check functionality."""
    
    @patch('src.llm.bedrock_client.boto3')
    def test_health_check_success(self, mock_boto3):
        """Test successful health check."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        mock_response = {
            'body': MagicMock()
        }
        mock_response['body'].read.return_value = b'{"content": [{"text": "OK"}]}'
        mock_client.invoke_model.return_value = mock_response
        
        client = BedrockClient()
        result = client.health_check()
        
        assert result["healthy"] is True
    
    @patch('src.llm.bedrock_client.boto3')
    def test_health_check_failure(self, mock_boto3):
        """Test health check failure handling."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.invoke_model.side_effect = Exception("Service unavailable")
        
        client = BedrockClient()
        result = client.health_check()
        
        assert result["healthy"] is False
        assert "error" in result


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling scenarios."""
    
    @patch('src.llm.bedrock_client.boto3')
    def test_rate_limit_handling(self, mock_boto3):
        """Test handling of rate limit errors."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        # Simulate rate limit error
        from botocore.exceptions import ClientError
        mock_client.invoke_model.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel"
        )
        
        client = BedrockClient()
        result = client.generate("Test")
        
        assert result["success"] is False
    
    @patch('src.llm.bedrock_client.boto3')
    def test_timeout_handling(self, mock_boto3):
        """Test handling of timeout errors."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        
        # Simulate timeout
        mock_client.invoke_model.side_effect = TimeoutError("Request timed out")
        
        client = BedrockClient()
        result = client.generate("Test")
        
        assert result["success"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
