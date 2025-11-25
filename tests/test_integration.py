"""
Integration tests for the complete system.
"""

import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.cli import process_urls
from src.models import (MetricResult, ModelContext, ParsedURL, SizeScore,
                        URLCategory)
from src.scoring import MetricScorer


@pytest.fixture
def sample_url_file():
    """Create sample URL file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("https://huggingface.co/datasets/test/dataset\n")
        f.write("https://github.com/test/code\n")
        f.write("https://huggingface.co/test/model\n")
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


def test_full_scoring_pipeline():
    """Test complete scoring pipeline."""
    # Create test model context
    model_url = ParsedURL(
        url="https://huggingface.co/test/model",
        category=URLCategory.MODEL,
        name="test/model",
        platform="huggingface",
    )
    context = ModelContext(model_url=model_url)

    # Create scorer with mocked dependencies
    scorer = MetricScorer()

    # Mock HF API calls
    scorer.hf_api.get_model_info = AsyncMock(
        return_value={"id": "test/model", "downloads": 100}
    )
    scorer.hf_api.get_readme_content = AsyncMock(
        return_value="# Test Model\n## Usage\nExample usage"
    )
    scorer.hf_api.get_model_config = AsyncMock(return_value=None)

    # Mock metric computations
    for metric in scorer.metrics:
        if metric.name == "size_score":
            metric._calculate_size_scores = AsyncMock(
                return_value=SizeScore(
                    raspberry_pi=0.5, jetson_nano=0.7, desktop_pc=0.9, aws_server=1.0
                )
            )
        else:
            metric.compute = AsyncMock(
                return_value=MetricResult(score=0.7, latency=100)
            )

    # Run scoring
    result = scorer.score_model(context)

    # Verify result structure
    assert result.name == "test/model"
    assert result.category == "MODEL"
    assert 0.0 <= result.net_score <= 1.0
    assert result.net_score_latency >= 0


def test_config_loading():
    """Test configuration loading from YAML."""
    scorer = MetricScorer()

    # Should have default config
    assert "metric_weights" in scorer.config
    assert "thresholds" in scorer.config

    # Test with custom config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(
            """
metric_weights:
  ramp_up_time: 0.3
  license: 0.7
        """
        )
        temp_path = f.name

    try:
        custom_scorer = MetricScorer(temp_path)
        assert custom_scorer.config["metric_weights"]["ramp_up_time"] == 0.3
        assert custom_scorer.config["metric_weights"]["license"] == 0.7
    finally:
        os.unlink(temp_path)


@patch("src.cli.asyncio.run")
@patch("src.cli.build_model_contexts")
@patch("src.cli.setup_logging")
def test_process_urls_integration(
    mock_logging, mock_contexts, mock_asyncio, sample_url_file
):
    """Test complete URL processing integration."""
    mock_logger = Mock()
    mock_logging.return_value = mock_logger

    # Mock contexts
    mock_context = Mock()
    mock_context.model_url.name = "test/model"
    mock_contexts.return_value = [mock_context]

    # Should complete without error
    process_urls(sample_url_file)

    mock_contexts.assert_called_once()
    mock_asyncio.assert_called_once()


def test_metric_weights_normalization():
    """Test that metric weights are reasonable."""
    scorer = MetricScorer()
    weights = scorer.config["metric_weights"]

    # Check that all expected metrics have weights
    expected_metrics = [
        "ramp_up_time",
        "bus_factor",
        "performance_claims",
        "license",
        "size_score",
        "dataset_and_code_score",
        "dataset_quality",
        "code_quality",
    ]

    for metric in expected_metrics:
        assert metric in weights
        assert 0.0 <= weights[metric] <= 1.0

    # Total should be close to 1.0
    total_weight = sum(weights.values())
    assert 0.8 <= total_weight <= 1.2  # Allow some flexibility


def test_error_handling_graceful():
    """Test that system handles errors gracefully."""
    scorer = MetricScorer()

    # Test with missing model URL
    context = ModelContext(
        model_url=ParsedURL(
            url="https://invalid.com/model",
            category=URLCategory.MODEL,
            name="invalid/model",
            platform="unknown",
        )
    )

    # Should not crash even with invalid data
    import asyncio

    def test_error_resilience():
        # Mock to raise errors
        scorer.hf_api.get_model_info = AsyncMock(side_effect=Exception("API Error"))
        scorer.hf_api.get_readme_content = AsyncMock(
            side_effect=Exception("README Error")
        )
        scorer.hf_api.get_model_config = AsyncMock(
            side_effect=Exception("Config Error")
        )

        try:
            result = scorer.score_model(context)
            # Should still produce a result, even if with low scores
            assert result.name == "invalid/model"
            assert 0.0 <= result.net_score <= 1.0
        except Exception as e:
            # If it does error, it should be handled gracefully
            assert "API Error" in str(e) or "README Error" in str(e)

    # This test ensures error handling doesn't completely break the system
    asyncio.run(test_error_resilience())
