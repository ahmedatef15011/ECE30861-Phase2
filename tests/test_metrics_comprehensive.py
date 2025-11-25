"""
Comprehensive tests for all metrics to improve coverage.
"""

from unittest.mock import Mock, patch

import pytest

from src.metrics.bus_factor import BusFactorMetric
from src.metrics.code_quality import CodeQualityMetric
from src.metrics.dataset_and_code import DatasetAndCodeScoreMetric
from src.metrics.dataset_quality import DatasetQualityMetric
from src.metrics.performance_claims import PerformanceClaimsMetric
from src.metrics.size_score import SizeScoreMetric
from src.models import ModelContext, ParsedURL, URLCategory


@pytest.fixture
def config():
    """Standard configuration for metrics testing."""
    return {
        "thresholds": {
            "bus_factor": {
                "min_contributors": 3,
                "recent_commits_window_days": 90,
                "single_contributor_penalty": 0.5,
            },
            "performance": {
                "benchmark_keywords": ["glue", "mmlu"],
                "citation_bonus": 0.2,
                "numeric_results_bonus": 0.3,
            },
            "size_limits": {
                "raspberry_pi": 2.0,
                "jetson_nano": 8.0,
                "desktop_pc": 32.0,
                "aws_server": 128.0,
            },
            "dataset_code": {
                "dataset_link_bonus": 0.4,
                "code_example_bonus": 0.3,
                "both_present_bonus": 0.3,
            },
            "dataset_quality_checklist": ["description", "license", "splits"],
            "code_quality": {
                "max_flake8_issues": 50,
                "min_test_coverage": 0.5,
                "typing_ratio_threshold": 0.3,
            },
        }
    }


# Bus Factor Metric Tests
def test_bus_factor_no_data(config):
    """Test bus factor with no HF info or code repos."""
    metric = BusFactorMetric()
    context = ModelContext(
        model_url=ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test/model",
            platform="huggingface",
        )
    )

    result = metric.compute(context, config)
    assert result.score <= 0.2  # Should get low score for no data


def test_bus_factor_with_hf_engagement(config):
    """Test bus factor with high HF engagement."""
    metric = BusFactorMetric()
    context = ModelContext(
        model_url=ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test/model",
            platform="huggingface",
        ),
    )
    context.hf_info = {"downloads": 50000, "likes": 200, "last_modified": "2023-12-01"}

    result = metric.compute(context, config)
    assert result.score > 0.3  # Should get good score for high engagement


# Performance Claims Metric Tests
@pytest.mark.parametrize(
    "readme_content,expected_range",
    [
        ("No performance data", (0.0, 0.2)),
        ("GLUE benchmark: 82.3% accuracy", (0.3, 0.6)),
        (
            "MMLU score: 0.85\nSee paper: https://arxiv.org/abs/2301.00001",
            (0.6, 1.0),
        ),
        (
            "Performance on GLUE: 85.2%\nMMLU: 0.87\nCitation: "
            "[Paper](https://link.com)",
            (0.7, 1.0),
        ),
    ],
)
def test_performance_claims_variations(readme_content, expected_range, config):
    """Test performance claims with various README content."""
    metric = PerformanceClaimsMetric()
    context = ModelContext(
        model_url=ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test/model",
            platform="huggingface",
        ),
    )
    context.readme_content = readme_content

    result = metric.compute(context, config)
    assert expected_range[0] <= result.score <= expected_range[1]


# Size Score Metric Tests
@pytest.mark.parametrize(
    "model_name,expected_size",
    [
        ("model-7b", 14.0),  # 7B * 2GB
        ("small-model", 0.3),  # 'small' keyword -> 0.3
        ("large-model", 2.5),  # 'large' keyword -> 2.5
        ("unknown-model", 0.5),  # default fallback
    ],
)
def test_size_score_name_patterns(model_name, expected_size, config):
    """Test size score extraction from model names."""
    metric = SizeScoreMetric()
    context = ModelContext(
        model_url=ParsedURL(
            url=f"https://huggingface.co/test/{model_name}",
            category=URLCategory.MODEL,
            name=f"test/{model_name}",
            platform="huggingface",
        ),
    )

    # Test the size estimation method directly
    estimated_size = metric._estimate_model_size(context)
    assert abs(estimated_size - expected_size) < 0.1  # Allow small tolerance


# Dataset and Code Score Tests
@pytest.mark.parametrize(
    "has_datasets,has_code,expected_min",
    [
        (False, False, 0.1),  # Base score only
        (True, False, 0.4),  # Dataset bonus
        (False, True, 0.3),  # Code bonus
        (True, True, 0.7),  # Both bonuses
    ],
)
def test_dataset_code_combinations(has_datasets, has_code, expected_min, config):
    """Test dataset and code score with different combinations."""
    metric = DatasetAndCodeScoreMetric()
    context = ModelContext(
        model_url=ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test/model",
            platform="huggingface",
        ),
    )

    if has_datasets:
        context.datasets = [
            ParsedURL(
                url="https://huggingface.co/datasets/test/data",
                category=URLCategory.DATASET,
                name="test/data",
                platform="huggingface",
            )
        ]
    if has_code:
        context.code_repos = [
            ParsedURL(
                url="https://github.com/test/code",
                category=URLCategory.CODE,
                name="test/code",
                platform="github",
            )
        ]

    result = metric.compute(context, config)
    assert result.score >= expected_min


# Dataset Quality Tests
def test_dataset_quality_no_datasets(config):
    """Test dataset quality with no linked datasets."""
    metric = DatasetQualityMetric()
    context = ModelContext(
        model_url=ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test/model",
            platform="huggingface",
        ),
    )

    result = metric.compute(context, config)
    assert result.score == 0.3  # Default when no datasets


def test_dataset_quality_with_mock_hf_dataset(config):
    """Test dataset quality with mocked HF dataset."""
    metric = DatasetQualityMetric()
    context = ModelContext(
        model_url=ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test/model",
            platform="huggingface",
        ),
    )
    context.datasets = [
        ParsedURL(
            url="https://huggingface.co/datasets/test/data",
            category=URLCategory.DATASET,
            name="test/data",
            platform="huggingface",
        )
    ]

    # Mock the HF API call
    with patch("src.metrics.dataset_quality.HuggingFaceAPI") as MockAPI:
        mock_api_instance = MockAPI.return_value
        mock_api_instance.get_readme_content = Mock(
            return_value="""
        # Dataset Description
        This dataset contains text samples.

        ## License
        Apache 2.0

        ## Splits
        Train/test/validation splits available.
        """
        )
        mock_api_instance.get_dataset_info = Mock(
            return_value={"tags": ["license:apache-2.0"]}
        )

        result = metric.compute(context, config)
        assert result.score > 0.7  # Should get high score for complete info


# Code Quality Tests
def test_code_quality_no_repos(config):
    """Test code quality with no linked repositories."""
    metric = CodeQualityMetric()
    context = ModelContext(
        model_url=ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test/model",
            platform="huggingface",
        ),
    )

    result = metric.compute(context, config)
    assert result.score == 0.5  # Default medium score


def test_code_quality_with_mock_analysis(config):
    """Test code quality with mocked repository analysis."""
    metric = CodeQualityMetric()
    context = ModelContext(
        model_url=ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test/model",
            platform="huggingface",
        ),
    )
    context.code_repos = [
        ParsedURL(
            url="https://github.com/test/code",
            category=URLCategory.CODE,
            name="test/code",
            platform="github",
        )
    ]

    with patch("src.metrics.code_quality.GitInspector") as MockInspector:
        mock_inspector = MockInspector.return_value
        mock_inspector.clone_repo.return_value = "/tmp/test/repo"
        mock_inspector.analyze_repository.return_value = {
            "structure_analysis": {"structure_score": 0.8},
            "documentation_analysis": {"documentation_score": 0.9},
            "file_analysis": {
                "python_files": 20,
                "test_files": 10,
                "total_lines_of_code": 5000,
            },
            "commit_analysis": {
                "total_commits": 50,
                "recent_commits": 10,
                "avg_commit_frequency": 0.5,
            },
        }

        result = metric.compute(context, config)
        assert result.score >= 0.5  # Should get baseline score without HF info
