"""
Comprehensive tests for LLM scoring helper functions.

Tests cover:
- Code reproducibility analysis
- README quality analysis
- Dataset analysis from README
- Code analysis from README
- Reviewedness analysis from README
- Graceful fallback when LLM unavailable
"""

import pytest
from unittest.mock import patch, MagicMock

from src.metrics.llm_scoring import (
    analyze_code_reproducibility,
    analyze_readme_quality,
    analyze_dataset_from_readme,
    analyze_code_from_readme,
    analyze_reviewedness_from_readme,
    HAS_LLM,
    LLM_ENABLED,
    get_llm_client,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_code():
    """Sample code snippet for testing."""
    return """
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
inputs = torch.randn(1, 10, 768)
outputs = model(inputs_embeds=inputs)
print(outputs.last_hidden_state.shape)
"""


@pytest.fixture
def sample_readme():
    """Sample README content for testing."""
    return """
# Awesome BERT Model

This is a fine-tuned BERT model for text classification.

## Dataset

The model was trained on the IMDB movie review dataset with 50,000 samples.
Training data was preprocessed using standard tokenization.

## Installation

```bash
pip install awesome-bert
```

## Usage

```python
from awesome_bert import classify
result = classify("This movie was great!")
print(result)
```

## Architecture

- Model: BERT-base (12 layers, 768 hidden size)
- Fine-tuning: Classification head
- Parameters: 110M

## Contributors

- Alice (Lead Developer)
- Bob (Researcher)

Developed at Google Research.

## Citation

```bibtex
@article{awesome2024,
    title={Awesome BERT},
    author={Alice and Bob}
}
```
"""


@pytest.fixture
def minimal_content():
    """Minimal content for edge case testing."""
    return "# Model\n\nA simple model."


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestModuleAvailability:
    """Test module-level availability checks."""
    
    def test_has_llm_is_bool(self):
        """Test HAS_LLM is a boolean."""
        assert isinstance(HAS_LLM, bool)
    
    def test_llm_enabled_is_bool(self):
        """Test LLM_ENABLED is a boolean."""
        assert isinstance(LLM_ENABLED, bool)
    
    def test_get_llm_client_returns_client_or_none(self):
        """Test get_llm_client returns client or None."""
        client = get_llm_client()
        # Should return either a client or None (never raise)
        assert client is None or hasattr(client, 'generate')


# =============================================================================
# CODE REPRODUCIBILITY ANALYSIS TESTS
# =============================================================================

class TestCodeReproducibilityAnalysis:
    """Test analyze_code_reproducibility function."""
    
    @patch('src.metrics.llm_scoring.HAS_LLM', False)
    def test_returns_fallback_when_no_llm(self, sample_code):
        """Test fallback when LLM not available."""
        score, details = analyze_code_reproducibility(sample_code)
        
        assert score == -1.0
        assert "error" in details or "reason" in details
    
    @patch('src.metrics.llm_scoring.HAS_LLM', True)
    @patch('src.metrics.llm_scoring.LLM_ENABLED', True)
    @patch('src.metrics.llm_scoring.BedrockClient')
    def test_successful_analysis(self, mock_client_class, sample_code):
        """Test successful code analysis."""
        # Mock the Bedrock client
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "success": True,
            "response": """
            REPRODUCIBILITY_SCORE: 0.85
            EXPLANATION: Code imports are clear, uses standard libraries.
            ISSUES: None significant.
            """
        }
        mock_client_class.return_value = mock_client
        
        score, details = analyze_code_reproducibility(
            sample_code, model_name="test/model"
        )
        
        # Should either get LLM score or fallback
        assert isinstance(score, float)
        assert isinstance(details, dict)
    
    def test_empty_code_handling(self):
        """Test handling of empty code."""
        score, details = analyze_code_reproducibility("")
        
        assert score == -1.0 or score == 0.0
    
    def test_none_code_handling(self):
        """Test handling of None code."""
        score, details = analyze_code_reproducibility(None)
        
        assert score == -1.0 or score == 0.0


# =============================================================================
# README QUALITY ANALYSIS TESTS
# =============================================================================

class TestReadmeQualityAnalysis:
    """Test analyze_readme_quality function."""
    
    @patch('src.metrics.llm_scoring.HAS_LLM', False)
    def test_returns_fallback_when_no_llm(self, sample_readme):
        """Test fallback when LLM not available."""
        score, details = analyze_readme_quality(sample_readme)
        
        assert score == -1.0
        assert "error" in details or "reason" in details
    
    @patch('src.metrics.llm_scoring.HAS_LLM', True)
    @patch('src.metrics.llm_scoring.LLM_ENABLED', True)
    @patch('src.metrics.llm_scoring.BedrockClient')
    def test_successful_analysis(self, mock_client_class, sample_readme):
        """Test successful README analysis."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "success": True,
            "response": """
            QUALITY_SCORE: 0.9
            SECTIONS: installation, usage, architecture
            """
        }
        mock_client_class.return_value = mock_client
        
        score, details = analyze_readme_quality(sample_readme)
        
        assert isinstance(score, float)
        assert isinstance(details, dict)
    
    def test_empty_readme_handling(self):
        """Test handling of empty README."""
        score, details = analyze_readme_quality("")
        
        assert score == -1.0


# =============================================================================
# DATASET FROM README ANALYSIS TESTS
# =============================================================================

class TestDatasetFromReadmeAnalysis:
    """Test analyze_dataset_from_readme function."""
    
    @patch('src.metrics.llm_scoring.HAS_LLM', False)
    def test_returns_fallback_when_no_llm(self, sample_readme):
        """Test fallback when LLM not available."""
        score, details = analyze_dataset_from_readme(sample_readme)
        
        assert score == -1.0
    
    @patch('src.metrics.llm_scoring.HAS_LLM', True)
    @patch('src.metrics.llm_scoring.LLM_ENABLED', True)
    @patch('src.metrics.llm_scoring.BedrockClient')
    def test_successful_analysis(self, mock_client_class, sample_readme):
        """Test successful dataset analysis."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "success": True,
            "response": """
            DATASET_QUALITY_SCORE: 0.75
            DATASETS_MENTIONED: IMDB
            DATA_DESCRIPTION: Movie reviews for sentiment
            """
        }
        mock_client_class.return_value = mock_client
        
        score, details = analyze_dataset_from_readme(
            sample_readme, model_name="test/model"
        )
        
        assert isinstance(score, float)
        assert isinstance(details, dict)
    
    def test_empty_readme_handling(self):
        """Test handling of empty README."""
        score, details = analyze_dataset_from_readme("")
        
        assert score == -1.0


# =============================================================================
# CODE FROM README ANALYSIS TESTS
# =============================================================================

class TestCodeFromReadmeAnalysis:
    """Test analyze_code_from_readme function."""
    
    @patch('src.metrics.llm_scoring.HAS_LLM', False)
    def test_returns_fallback_when_no_llm(self, sample_readme):
        """Test fallback when LLM not available."""
        score, details = analyze_code_from_readme(sample_readme)
        
        assert score == -1.0
    
    @patch('src.metrics.llm_scoring.HAS_LLM', True)
    @patch('src.metrics.llm_scoring.LLM_ENABLED', True)
    @patch('src.metrics.llm_scoring.BedrockClient')
    def test_successful_analysis(self, mock_client_class, sample_readme):
        """Test successful code documentation analysis."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "success": True,
            "response": """
            CODE_QUALITY_SCORE: 0.8
            HAS_INSTALLATION: True
            HAS_USAGE: True
            """
        }
        mock_client_class.return_value = mock_client
        
        score, details = analyze_code_from_readme(
            sample_readme, model_name="test/model"
        )
        
        assert isinstance(score, float)
        assert isinstance(details, dict)


# =============================================================================
# REVIEWEDNESS FROM README ANALYSIS TESTS
# =============================================================================

class TestReviewednessFromReadmeAnalysis:
    """Test analyze_reviewedness_from_readme function."""
    
    @patch('src.metrics.llm_scoring.HAS_LLM', False)
    def test_returns_fallback_when_no_llm(self, sample_readme):
        """Test fallback when LLM not available."""
        score, details = analyze_reviewedness_from_readme(sample_readme)
        
        assert score == -1.0
    
    @patch('src.metrics.llm_scoring.HAS_LLM', True)
    @patch('src.metrics.llm_scoring.LLM_ENABLED', True)
    @patch('src.metrics.llm_scoring.BedrockClient')
    def test_successful_analysis(self, mock_client_class, sample_readme):
        """Test successful reviewedness analysis."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "success": True,
            "response": """
            REVIEWEDNESS_SCORE: 0.7
            CONTRIBUTORS: 2
            INSTITUTIONAL: Google Research
            HAS_CITATION: True
            """
        }
        mock_client_class.return_value = mock_client
        
        score, details = analyze_reviewedness_from_readme(
            sample_readme, model_name="test/model"
        )
        
        assert isinstance(score, float)
        assert isinstance(details, dict)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling in LLM scoring functions."""
    
    @patch('src.metrics.llm_scoring.HAS_LLM', True)
    @patch('src.metrics.llm_scoring.LLM_ENABLED', True)
    @patch('src.metrics.llm_scoring.BedrockClient')
    def test_client_exception_handling(self, mock_client_class, sample_code):
        """Test handling of client exceptions."""
        mock_client_class.side_effect = Exception("Connection failed")
        
        score, details = analyze_code_reproducibility(sample_code)
        
        # Should return fallback, not raise
        assert score == -1.0
        assert "error" in details
    
    @patch('src.metrics.llm_scoring.HAS_LLM', True)
    @patch('src.metrics.llm_scoring.LLM_ENABLED', True)
    @patch('src.metrics.llm_scoring.BedrockClient')
    def test_api_error_handling(self, mock_client_class, sample_readme):
        """Test handling of API errors."""
        mock_client = MagicMock()
        mock_client.generate.return_value = {
            "success": False,
            "error": "Rate limit exceeded"
        }
        mock_client_class.return_value = mock_client
        
        score, details = analyze_readme_quality(sample_readme)
        
        # Should return fallback on API error
        assert isinstance(score, float)


# =============================================================================
# RETURN TYPE TESTS
# =============================================================================

class TestReturnTypes:
    """Test that all functions return correct types."""
    
    def test_all_functions_return_tuple(self, sample_code, sample_readme):
        """Test all functions return (float, dict) tuple."""
        functions = [
            (analyze_code_reproducibility, sample_code),
            (analyze_readme_quality, sample_readme),
            (analyze_dataset_from_readme, sample_readme),
            (analyze_code_from_readme, sample_readme),
            (analyze_reviewedness_from_readme, sample_readme),
        ]
        
        for func, content in functions:
            result = func(content)
            
            assert isinstance(result, tuple), f"{func.__name__} didn't return tuple"
            assert len(result) == 2, f"{func.__name__} didn't return 2 elements"
            assert isinstance(result[0], float), \
                f"{func.__name__} score is not float"
            assert isinstance(result[1], dict), \
                f"{func.__name__} details is not dict"


# =============================================================================
# SCORE RANGE TESTS
# =============================================================================

class TestScoreRanges:
    """Test that scores are in valid ranges."""
    
    def test_scores_in_valid_range(self, sample_code, sample_readme):
        """Test all scores are between -1.0 and 1.0."""
        functions = [
            (analyze_code_reproducibility, sample_code),
            (analyze_readme_quality, sample_readme),
            (analyze_dataset_from_readme, sample_readme),
            (analyze_code_from_readme, sample_readme),
            (analyze_reviewedness_from_readme, sample_readme),
        ]
        
        for func, content in functions:
            score, _ = func(content)
            
            # -1.0 means not applicable/unavailable, 0.0-1.0 is valid range
            assert -1.0 <= score <= 1.0, \
                f"{func.__name__} returned invalid score: {score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
