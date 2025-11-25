"""
Tests for ramp-up time metric.
"""

import pytest

from src.metrics.ramp_up import RampUpTimeMetric
from src.models import ModelContext, ParsedURL, URLCategory


@pytest.fixture
def ramp_up_metric():
    """Create RampUpTimeMetric instance."""
    return RampUpTimeMetric()


@pytest.fixture
def model_context():
    """Create model context for testing."""
    model_url = ParsedURL(
        url="https://huggingface.co/test/model",
        category=URLCategory.MODEL,
        name="test/model",
        platform="huggingface",
    )
    return ModelContext(model_url=model_url)


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "thresholds": {
            "ramp_up": {
                "readme_sections": ["usage", "quickstart", "examples"],
                "example_code_bonus": 0.2,
            }
        }
    }


def test_metric_name(ramp_up_metric):
    """Test metric name."""
    assert ramp_up_metric.name == "ramp_up_time"


def test_compute_no_readme(ramp_up_metric, model_context, config):
    """Test computation with no README."""
    result = await ramp_up_metric.compute(model_context, config)

    assert result.score == 0.1  # Low score for missing README
    assert result.latency >= 0


def test_compute_with_readme(ramp_up_metric, model_context, config):
    """Test computation with README content."""
    model_context.readme_content = """
    # Test Model

    ## Usage
    Here's how to use this model...

    ## Examples
    ```python
    import torch
    model = load_model()
    ```
    """

    result = await ramp_up_metric.compute(model_context, config)

    assert result.score > 0.1  # Should get higher score
    assert result.latency >= 0


def test_compute_comprehensive_readme(ramp_up_metric, model_context, config):
    """Test computation with comprehensive README."""
    model_context.readme_content = (
        """
    # Test Model

    ## Usage
    Detailed usage instructions...

    ## Quickstart
    Quick start guide...

    ## Examples
    Multiple examples with code...

    ```python
    from transformers import AutoModel
    model = AutoModel.from_pretrained("test/model")
    ```

    This is a long README with over 1000 characters to test the length bonus.
    It contains comprehensive documentation covering all aspects of the model.
    """
        * 10
    )  # Make it long

    result = await ramp_up_metric.compute(model_context, config)

    assert result.score > 0.7  # Should get high score
    assert result.latency >= 0
