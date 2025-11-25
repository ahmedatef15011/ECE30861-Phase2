"""
Additional tests to boost coverage to ≥80%.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from src.git_inspect import GitInspector
from src.hf_api import HuggingFaceAPI
from src.metrics.size_score import SizeScoreMetric
from src.models import ModelContext, ParsedURL, URLCategory


def test_cli_error_cases():
    """Test CLI error handling cases."""
    from src.cli import run_tests

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = Exception("Test failed")

        with pytest.raises(SystemExit):
            run_tests()


def test_git_inspector_analysis_edge_cases():
    """Test git inspector with various edge cases."""
    inspector = GitInspector()

    # Test empty analysis
    empty_analysis = inspector._empty_analysis()
    assert empty_analysis["commit_analysis"]["total_commits"] == 0

    # Test file analysis edge cases
    with tempfile.TemporaryDirectory() as temp_dir:
        from pathlib import Path

        # Create some test files
        (Path(temp_dir) / "test.py").write_text("print('hello')")
        (Path(temp_dir) / "test_file.py").write_text("def test(): pass")

        analysis = inspector._analyze_files(temp_dir)
        assert analysis["python_files"] == 2
        assert analysis["test_files"] == 2  # Both files match test pattern


def test_hf_api_error_paths():
    """Test HF API error handling."""
    api = HuggingFaceAPI()

    # Test with invalid repo format
    invalid_url = ParsedURL(
        url="https://huggingface.co/test",
        category=URLCategory.MODEL,
        name="test",
        platform="huggingface",
        owner=None,
        repo=None,
    )

    result = api.get_model_info(invalid_url)
    assert result is None

    result = api.get_model_config(invalid_url)
    assert result is None


def test_size_score_calculations():
    """Test size score calculations with different thresholds."""
    metric = SizeScoreMetric()

    # Test device score calculation - using actual formula with softness = 1.2
    # score = 1.0 / (1.0 + (model_size/limit)^1.2)
    score_well_under = metric._calculate_device_score(1.0, 2.0)
    assert 0.6 < score_well_under < 0.75  # ratio=0.5, score~0.697
    
    score_at_limit = metric._calculate_device_score(2.0, 2.0)
    assert 0.45 < score_at_limit < 0.55  # ratio=1.0, score=0.5
    
    score_double = metric._calculate_device_score(4.0, 2.0)
    assert 0.28 < score_double < 0.35  # ratio=2.0, score~0.312
    
    score_way_over = metric._calculate_device_score(10.0, 2.0)
    assert 0.12 < score_way_over < 0.16  # ratio=5.0, score~0.127


def test_logging_edge_cases():
    """Test logging configuration edge cases."""
    from src.logging_utils import setup_logging

    # Test with invalid log level - should exit with error
    with patch.dict(os.environ, {"LOG_LEVEL": "invalid"}):
        with pytest.raises(SystemExit) as exc_info:
            setup_logging()
        assert exc_info.value.code == 1

    # Test with debug level
    with patch.dict(os.environ, {"LOG_LEVEL": "2"}):
        logger = setup_logging()
        assert logger.level == 10  # DEBUG level


def test_url_parsing_comprehensive():
    """Test comprehensive URL parsing scenarios."""
    from src.urls import _extract_name_parts

    # Test name part extraction
    parts = _extract_name_parts("sentiment-analysis-model-v2")
    assert "sentiment" in parts
    assert "analysis" in parts
    assert "model" in parts

    # Test with short names (should be filtered)
    parts = _extract_name_parts("a-b-c-model")
    assert "model" in parts  # Only long enough part


def test_metrics_error_handling():
    """Test metrics error handling."""
    from src.metrics.bus_factor import BusFactorMetric

    metric = BusFactorMetric()
    context = ModelContext(
        model_url=ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test/model",
            platform="huggingface",
        )
    )

    config = {"thresholds": {"bus_factor": {}}}

    # Should handle missing config gracefully
    result = metric.compute(context, config)
    assert 0.0 <= result.score <= 1.0
    assert result.latency >= 0


def test_scoring_config_defaults():
    """Test scoring system with default configuration."""
    from src.scoring import MetricScorer

    # Test with completely missing config file
    scorer = MetricScorer("nonexistent_path.yaml")

    # Should have default configuration
    assert "metric_weights" in scorer.config
    # Phase 2 added reproducibility metrics, so now we have more than 8
    assert len(scorer.config["metric_weights"]) >= 8


def test_utils_edge_cases():
    """Test utility functions edge cases."""
    from src.utils import check_readme_sections, extract_performance_claims

    # Test with None inputs
    claims = extract_performance_claims(None, [])
    assert claims["benchmarks_mentioned"] == []

    sections = check_readme_sections(None, ["usage"])
    assert sections["usage"] is False

    # Test with special characters in README
    readme = """# Model\n## Usage\nSpecial chars: ±∞"""
    sections = check_readme_sections(readme, ["usage"])
    assert sections["usage"]


def test_model_context_defaults():
    """Test model context default values."""
    from src.models import ModelContext, ParsedURL, URLCategory

    model_url = ParsedURL(
        url="https://test.com/model",
        category=URLCategory.MODEL,
        name="model",
        platform="test",
    )

    context = ModelContext(model_url=model_url)

    # Test all default values
    assert context.datasets == []
    assert context.code_repos == []
    assert context.hf_info is None
    assert context.readme_content is None
    assert context.config_data is None
