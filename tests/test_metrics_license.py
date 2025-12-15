"""
Tests for license score metric.
"""

import pytest

from src.metrics.license_score import LicenseScoreMetric
from src.models import ModelContext, ParsedURL, URLCategory


@pytest.fixture
def license_metric():
    """Create LicenseScoreMetric instance."""
    return LicenseScoreMetric()


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
            "license": {
                "compatible_licenses": ["apache-2.0", "mit"],
                "restrictive_penalty": 0.3,
                "missing_penalty": 0.7,
            }
        }
    }


def test_metric_name(license_metric):
    """Test metric name."""
    assert license_metric.name == "license"


def test_compute_no_license(license_metric, model_context, config):
    """Test computation with no license information."""
    result = license_metric.compute(model_context, config)

    assert (
        abs(result.score - 0.24) < 0.01
    )  # 0.3 * 0.8 = 0.24 (scaled down)
    assert result.latency >= 0


def test_compute_hf_license(license_metric, model_context, config):
    """Test computation with HF license tag."""
    model_context.hf_info = {"tags": ["license:apache-2.0", "pytorch"]}

    result = license_metric.compute(model_context, config)

    assert abs(result.score - 0.8) < 0.001  # 1.0 * 0.8 (scaled down)
    assert result.latency >= 0


def test_compute_readme_license(license_metric, model_context, config):
    """Test computation with README license."""
    model_context.readme_content = """
    # Test Model

    ## License
    This model is licensed under MIT License.
    """

    result = license_metric.compute(model_context, config)

    assert abs(result.score - 0.8) < 0.001  # 1.0 * 0.8 (scaled down)
    assert result.latency >= 0


def test_compute_restrictive_license(license_metric, model_context, config):
    """Test computation with restrictive license."""
    model_context.readme_content = """
    # Test Model

    ## License
    This model is licensed under GPL v3.
    """

    result = license_metric.compute(model_context, config)

    assert abs(result.score - 0.56) < 0.001  # 0.7 * 0.8 (scaled down)
    assert result.latency >= 0


def test_compute_unknown_license(license_metric, model_context, config):
    """Test computation with unknown license."""
    model_context.readme_content = """
    # Test Model

    ## License
    This model uses a custom license.
    """

    result = license_metric.compute(model_context, config)

    assert abs(result.score - 0.4) < 0.001  # 0.5 * 0.8 (scaled down)
    assert result.latency >= 0
