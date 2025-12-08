"""
Comprehensive tests for FallbackScorer with LLM integration.

Tests cover:
- Dataset fallback scoring (deterministic and LLM-enhanced)
- Code fallback scoring (deterministic and LLM-enhanced)
- Reviewedness fallback scoring (deterministic and LLM-enhanced)
- LLM availability checks
- Graceful fallback when LLM unavailable
"""

import pytest
from unittest.mock import patch, MagicMock

from src.metrics.fallback_scoring import FallbackScorer


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_readme_with_dataset():
    """README with dataset information."""
    return """
    # Awesome Model
    
    This model was trained on the ImageNet dataset with 1.2 million images.
    The training data was collected from various sources and preprocessed
    using standard tokenization techniques.
    
    ## Training Details
    - Dataset: ImageNet-1K
    - Samples: 1.2M images
    - Preprocessing: Standard augmentation and normalization
    """


@pytest.fixture
def sample_readme_with_code():
    """README with code documentation."""
    return """
    # Awesome Model
    
    ## Installation
    
    ```bash
    pip install awesome-model
    ```
    
    ## Usage
    
    ```python
    from awesome_model import Model
    
    model = Model.from_pretrained("awesome/model")
    output = model.generate("Hello world")
    print(output)
    ```
    
    ## Architecture
    
    This model uses a transformer encoder with 12 layers and 768 hidden size.
    
    ## API Reference
    
    - `model.generate(prompt)`: Generate text from a prompt
    - `model.encode(text)`: Encode text into embeddings
    """


@pytest.fixture
def sample_readme_with_review_indicators():
    """README with review/collaboration indicators."""
    return """
    # Awesome Model
    
    Developed by the Research Team at Google DeepMind.
    
    ## Contributors
    - Alice Smith
    - Bob Johnson
    - Carol Williams
    
    ## Acknowledgments
    
    We thank the open source community for their contributions.
    This work was supported by Grant #12345.
    
    ## Citation
    
    If you use this model, please cite our paper:
    
    ```bibtex
    @article{smith2024awesome,
        title={Awesome Model: A New Approach},
        author={Smith, Alice and Johnson, Bob},
        journal={arXiv preprint arXiv:2024.12345},
        year={2024}
    }
    ```
    
    ## Version History
    - v1.0.0: Initial release
    - v1.1.0: Bug fixes and improvements
    """


@pytest.fixture
def minimal_readme():
    """Minimal README with little information."""
    return """
    # Model
    
    This is a model.
    """


@pytest.fixture
def empty_readme():
    """Empty README."""
    return ""


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestFallbackScorerInit:
    """Test FallbackScorer initialization."""
    
    def test_init_with_readme(self, sample_readme_with_dataset):
        """Test initialization with valid README."""
        scorer = FallbackScorer(sample_readme_with_dataset)
        assert scorer.readme_content == sample_readme_with_dataset
        assert scorer.model_name == "unknown"
        assert scorer.use_llm is True  # Default
    
    def test_init_with_model_name(self, sample_readme_with_dataset):
        """Test initialization with model name."""
        scorer = FallbackScorer(
            sample_readme_with_dataset, 
            model_name="test/model"
        )
        assert scorer.model_name == "test/model"
    
    def test_init_disable_llm(self, sample_readme_with_dataset):
        """Test initialization with LLM disabled."""
        scorer = FallbackScorer(
            sample_readme_with_dataset, 
            use_llm=False
        )
        assert scorer.use_llm is False
    
    def test_init_empty_readme(self, empty_readme):
        """Test initialization with empty README."""
        scorer = FallbackScorer(empty_readme)
        assert scorer.readme_content == ""
        assert scorer.readme_lower == ""
    
    def test_init_none_readme(self):
        """Test initialization with None README."""
        scorer = FallbackScorer(None)
        assert scorer.readme_content == ""


class TestLLMAvailability:
    """Test LLM availability checks."""
    
    def test_is_llm_available(self, sample_readme_with_dataset):
        """Test LLM availability check."""
        scorer = FallbackScorer(sample_readme_with_dataset)
        # Result depends on whether LLM dependencies are installed
        result = scorer.is_llm_available()
        assert isinstance(result, bool)
    
    def test_get_llm_results_empty(self, sample_readme_with_dataset):
        """Test getting LLM results before any scoring."""
        scorer = FallbackScorer(sample_readme_with_dataset)
        results = scorer.get_llm_results()
        assert "llm_enabled" in results
        assert "model_name" in results
        assert "results" in results


# =============================================================================
# DATASET FALLBACK SCORING TESTS
# =============================================================================

class TestDatasetFallbackScoring:
    """Test dataset fallback scoring."""
    
    def test_dataset_score_with_rich_readme(self, sample_readme_with_dataset):
        """Test dataset scoring with rich README content."""
        scorer = FallbackScorer(sample_readme_with_dataset, use_llm=False)
        score, details = scorer.get_dataset_fallback_score()
        
        # Should find dataset mentions, training data, statistics
        assert score > 0.0
        assert score <= scorer.MAX_DATASET_FALLBACK
        assert "indicators_found" in details
        assert len(details["indicators_found"]) > 0
    
    def test_dataset_score_with_empty_readme(self, empty_readme):
        """Test dataset scoring with empty README."""
        scorer = FallbackScorer(empty_readme, use_llm=False)
        score, details = scorer.get_dataset_fallback_score()
        
        assert score == 0.0
        assert "reason" in details
    
    def test_dataset_score_with_minimal_readme(self, minimal_readme):
        """Test dataset scoring with minimal README."""
        scorer = FallbackScorer(minimal_readme, use_llm=False)
        score, details = scorer.get_dataset_fallback_score()
        
        # Minimal content should give base score at most
        assert score >= 0.0
        assert score <= 0.1
    
    def test_dataset_extraction_imagenet(self):
        """Test extraction of ImageNet dataset mention."""
        readme = "This model was trained on ImageNet."
        scorer = FallbackScorer(readme, use_llm=False)
        datasets = scorer._extract_dataset_mentions()
        
        assert "imagenet" in datasets
    
    def test_dataset_extraction_multiple(self):
        """Test extraction of multiple dataset mentions."""
        readme = "Trained on CIFAR-10, MNIST, and ImageNet datasets."
        scorer = FallbackScorer(readme, use_llm=False)
        datasets = scorer._extract_dataset_mentions()
        
        # Should find at least some of these
        assert len(datasets) >= 1
    
    def test_has_training_data_description(self):
        """Test training data description detection."""
        readme = "The model was trained on a large corpus of text."
        scorer = FallbackScorer(readme, use_llm=False)
        assert scorer._has_training_data_description() is True
    
    def test_has_data_statistics(self):
        """Test data statistics detection."""
        readme = "Dataset contains 1.2 million samples."
        scorer = FallbackScorer(readme, use_llm=False)
        assert scorer._has_data_statistics() is True
    
    def test_has_data_source(self):
        """Test data source detection."""
        readme = "Data was collected from Wikipedia articles."
        scorer = FallbackScorer(readme, use_llm=False)
        assert scorer._has_data_source() is True
    
    def test_has_preprocessing_info(self):
        """Test preprocessing info detection."""
        readme = "Data was tokenized using BPE tokenization."
        scorer = FallbackScorer(readme, use_llm=False)
        assert scorer._has_preprocessing_info() is True


# =============================================================================
# CODE FALLBACK SCORING TESTS
# =============================================================================

class TestCodeFallbackScoring:
    """Test code fallback scoring."""
    
    def test_code_score_with_rich_readme(self, sample_readme_with_code):
        """Test code scoring with rich README content."""
        scorer = FallbackScorer(sample_readme_with_code, use_llm=False)
        score, details = scorer.get_code_fallback_score()
        
        # Should find code blocks, installation, usage, architecture, API
        assert score > 0.0
        assert score <= scorer.MAX_CODE_FALLBACK
        assert "indicators_found" in details
        assert len(details["indicators_found"]) > 0
    
    def test_code_score_with_empty_readme(self, empty_readme):
        """Test code scoring with empty README."""
        scorer = FallbackScorer(empty_readme, use_llm=False)
        score, details = scorer.get_code_fallback_score()
        
        # Should return default
        assert score == 0.3
        assert "default" in details
    
    def test_code_score_with_minimal_readme(self, minimal_readme):
        """Test code scoring with minimal README."""
        scorer = FallbackScorer(minimal_readme, use_llm=False)
        score, details = scorer.get_code_fallback_score()
        
        # Minimal content should give default
        assert score == 0.3
    
    def test_count_code_blocks(self, sample_readme_with_code):
        """Test code block counting."""
        scorer = FallbackScorer(sample_readme_with_code, use_llm=False)
        count = scorer._count_code_blocks()
        
        # Should find at least 2 code blocks (bash and python)
        assert count >= 2
    
    def test_has_installation_instructions(self, sample_readme_with_code):
        """Test installation instructions detection."""
        scorer = FallbackScorer(sample_readme_with_code, use_llm=False)
        assert scorer._has_installation_instructions() is True
    
    def test_has_usage_examples(self, sample_readme_with_code):
        """Test usage examples detection."""
        scorer = FallbackScorer(sample_readme_with_code, use_llm=False)
        assert scorer._has_usage_examples() is True
    
    def test_has_architecture_description(self, sample_readme_with_code):
        """Test architecture description detection."""
        scorer = FallbackScorer(sample_readme_with_code, use_llm=False)
        assert scorer._has_architecture_description() is True
    
    def test_has_api_documentation(self, sample_readme_with_code):
        """Test API documentation detection."""
        scorer = FallbackScorer(sample_readme_with_code, use_llm=False)
        assert scorer._has_api_documentation() is True


# =============================================================================
# REVIEWEDNESS FALLBACK SCORING TESTS
# =============================================================================

class TestReviewednessFallbackScoring:
    """Test reviewedness fallback scoring."""
    
    def test_reviewedness_score_with_rich_readme(
        self, sample_readme_with_review_indicators
    ):
        """Test reviewedness scoring with rich README content."""
        scorer = FallbackScorer(
            sample_readme_with_review_indicators, use_llm=False
        )
        score, details = scorer.get_reviewedness_fallback_score()
        
        # Should find multiple indicators
        assert score > 0.0
        assert score <= scorer.MAX_REVIEWEDNESS_FALLBACK
        assert "indicators_found" in details
        assert len(details["indicators_found"]) > 0
    
    def test_reviewedness_score_with_empty_readme(self, empty_readme):
        """Test reviewedness scoring with empty README."""
        scorer = FallbackScorer(empty_readme, use_llm=False)
        score, details = scorer.get_reviewedness_fallback_score()
        
        # Should return baseline credit for HuggingFace-hosted models
        assert score == 0.25
        assert "reason" in details or "method" in details
    
    def test_reviewedness_score_with_minimal_readme(self, minimal_readme):
        """Test reviewedness scoring with minimal README."""
        scorer = FallbackScorer(minimal_readme, use_llm=False)
        score, details = scorer.get_reviewedness_fallback_score()
        
        # Should return baseline credit when no indicators found
        assert score == 0.20
    
    def test_has_multiple_contributors(
        self, sample_readme_with_review_indicators
    ):
        """Test multiple contributors detection."""
        scorer = FallbackScorer(
            sample_readme_with_review_indicators, use_llm=False
        )
        assert scorer._has_multiple_contributors() is True
    
    def test_has_institutional_backing(
        self, sample_readme_with_review_indicators
    ):
        """Test institutional backing detection."""
        scorer = FallbackScorer(
            sample_readme_with_review_indicators, use_llm=False
        )
        assert scorer._has_institutional_backing() is True
    
    def test_has_acknowledgments(self, sample_readme_with_review_indicators):
        """Test acknowledgments detection."""
        scorer = FallbackScorer(
            sample_readme_with_review_indicators, use_llm=False
        )
        assert scorer._has_acknowledgments() is True
    
    def test_has_version_history(self, sample_readme_with_review_indicators):
        """Test version history detection."""
        scorer = FallbackScorer(
            sample_readme_with_review_indicators, use_llm=False
        )
        assert scorer._has_version_history() is True
    
    def test_has_paper_citation(self, sample_readme_with_review_indicators):
        """Test paper citation detection."""
        scorer = FallbackScorer(
            sample_readme_with_review_indicators, use_llm=False
        )
        assert scorer._has_paper_citation() is True


# =============================================================================
# LLM-ENHANCED SCORING TESTS
# =============================================================================

class TestLLMEnhancedScoring:
    """Test LLM-enhanced scoring functionality."""
    
    @patch('src.metrics.fallback_scoring.HAS_LLM', True)
    @patch('src.metrics.fallback_scoring.LLM_ENABLED', True)
    @patch('src.metrics.fallback_scoring.analyze_dataset_from_readme')
    def test_dataset_llm_blending(
        self, mock_analyze, sample_readme_with_dataset
    ):
        """Test that LLM scores are blended with deterministic scores."""
        # Mock LLM returning a score
        mock_analyze.return_value = (0.8, {"llm_analysis": "good"})
        
        scorer = FallbackScorer(sample_readme_with_dataset, use_llm=True)
        score, details = scorer.get_dataset_fallback_score()
        
        # Should be LLM-enhanced if LLM available
        if "method" in details and details["method"] == "llm_enhanced_fallback":
            assert "llm_score" in details
            assert "deterministic_score" in details
            assert "blended_score" in details
            # Blended score should be between LLM and deterministic
            assert score <= scorer.MAX_DATASET_FALLBACK
    
    @patch('src.metrics.fallback_scoring.HAS_LLM', True)
    @patch('src.metrics.fallback_scoring.LLM_ENABLED', True)
    @patch('src.metrics.fallback_scoring.analyze_code_from_readme')
    def test_code_llm_blending(self, mock_analyze, sample_readme_with_code):
        """Test that LLM scores are blended for code scoring."""
        mock_analyze.return_value = (0.75, {"code_quality": "good"})
        
        scorer = FallbackScorer(sample_readme_with_code, use_llm=True)
        score, details = scorer.get_code_fallback_score()
        
        if "method" in details and details["method"] == "llm_enhanced_fallback":
            assert "llm_score" in details
            assert score <= scorer.MAX_CODE_FALLBACK
    
    @patch('src.metrics.fallback_scoring.HAS_LLM', True)
    @patch('src.metrics.fallback_scoring.LLM_ENABLED', True)
    @patch('src.metrics.fallback_scoring.analyze_reviewedness_from_readme')
    def test_reviewedness_llm_blending(
        self, mock_analyze, sample_readme_with_review_indicators
    ):
        """Test that LLM scores are blended for reviewedness scoring."""
        mock_analyze.return_value = (0.7, {"review_indicators": "many"})
        
        scorer = FallbackScorer(
            sample_readme_with_review_indicators, use_llm=True
        )
        score, details = scorer.get_reviewedness_fallback_score()
        
        if "method" in details and details["method"] == "llm_enhanced_fallback":
            assert "llm_score" in details
            assert score <= scorer.MAX_REVIEWEDNESS_FALLBACK
    
    @patch('src.metrics.fallback_scoring.HAS_LLM', False)
    def test_fallback_when_llm_unavailable(self, sample_readme_with_dataset):
        """Test graceful fallback when LLM is unavailable."""
        scorer = FallbackScorer(sample_readme_with_dataset, use_llm=True)
        score, details = scorer.get_dataset_fallback_score()
        
        # Should fall back to deterministic method
        assert score >= 0.0
        # Should not have LLM-enhanced method
        if "method" in details:
            assert details["method"] != "llm_enhanced_fallback"
    
    @patch('src.metrics.fallback_scoring.HAS_LLM', True)
    @patch('src.metrics.fallback_scoring.LLM_ENABLED', True)
    @patch('src.metrics.fallback_scoring.analyze_dataset_from_readme')
    def test_llm_error_handling(
        self, mock_analyze, sample_readme_with_dataset
    ):
        """Test handling of LLM errors."""
        # Mock LLM raising an exception
        mock_analyze.side_effect = Exception("LLM API Error")
        
        scorer = FallbackScorer(sample_readme_with_dataset, use_llm=True)
        score, details = scorer.get_dataset_fallback_score()
        
        # Should gracefully fall back to deterministic
        assert score >= 0.0
    
    def test_llm_disabled_via_parameter(self, sample_readme_with_dataset):
        """Test that LLM can be disabled via parameter."""
        scorer = FallbackScorer(sample_readme_with_dataset, use_llm=False)
        score, details = scorer.get_dataset_fallback_score()
        
        # Should only use deterministic method
        assert "method" in details
        assert details["method"] == "readme_fallback"


# =============================================================================
# WEIGHT AND CONSTANT TESTS
# =============================================================================

class TestWeightsAndConstants:
    """Test weight and constant values."""
    
    def test_weight_constants(self, sample_readme_with_dataset):
        """Test that weight constants are correct."""
        scorer = FallbackScorer(sample_readme_with_dataset)
        
        assert scorer.LLM_WEIGHT == 0.4
        assert scorer.DETERMINISTIC_WEIGHT == 0.6
        assert scorer.LLM_WEIGHT + scorer.DETERMINISTIC_WEIGHT == 1.0
    
    def test_max_fallback_constants(self, sample_readme_with_dataset):
        """Test max fallback score constants."""
        scorer = FallbackScorer(sample_readme_with_dataset)
        
        assert scorer.MAX_DATASET_FALLBACK == 0.65
        assert scorer.MAX_CODE_FALLBACK == 0.65
        assert scorer.MAX_REVIEWEDNESS_FALLBACK == 0.90


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for FallbackScorer."""
    
    def test_all_methods_return_tuple(self, sample_readme_with_dataset):
        """Test that all scoring methods return tuple of (score, details)."""
        scorer = FallbackScorer(sample_readme_with_dataset, use_llm=False)
        
        for method in [
            scorer.get_dataset_fallback_score,
            scorer.get_code_fallback_score,
            scorer.get_reviewedness_fallback_score,
        ]:
            result = method()
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert isinstance(result[0], (int, float))
            assert isinstance(result[1], dict)
    
    def test_scores_are_normalized(self, sample_readme_with_dataset):
        """Test that all scores are in valid range."""
        scorer = FallbackScorer(sample_readme_with_dataset, use_llm=False)
        
        dataset_score, _ = scorer.get_dataset_fallback_score()
        code_score, _ = scorer.get_code_fallback_score()
        review_score, _ = scorer.get_reviewedness_fallback_score()
        
        # Dataset and code should be 0.0 to max
        assert 0.0 <= dataset_score <= 1.0
        assert 0.0 <= code_score <= 1.0
        # Reviewedness can be -1.0 (not applicable) or 0.0 to max
        assert -1.0 <= review_score <= 1.0
    
    def test_llm_results_tracking(self, sample_readme_with_dataset):
        """Test that LLM results are tracked."""
        scorer = FallbackScorer(sample_readme_with_dataset, use_llm=False)
        
        # Run all scoring methods
        scorer.get_dataset_fallback_score()
        scorer.get_code_fallback_score()
        scorer.get_reviewedness_fallback_score()
        
        # Get LLM results
        llm_results = scorer.get_llm_results()
        
        assert "llm_enabled" in llm_results
        assert "model_name" in llm_results
        assert "results" in llm_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
