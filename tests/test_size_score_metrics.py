"""
Tests for size score metrics using sample URLs.
"""

import pytest

from src.metrics.size_score import SizeScoreMetric
from src.models import ModelContext, ParsedURL, URLCategory, SizeScore


@pytest.fixture
def size_metric():
    """Create SizeScoreMetric instance."""
    return SizeScoreMetric()


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "thresholds": {
            "size_limits": {
                "raspberry_pi": 1.0,
                "jetson_nano": 4.0,
                "desktop_pc": 16.0,
                "aws_server": 64.0,
            },
            "softness": 1.2,
        }
    }


def test_metric_name(size_metric):
    """Test metric name."""
    assert size_metric.name == "size_score"


def test_estimate_bert_base_uncased():
    """Test size estimation for bert-base-uncased from sample.txt."""
    model_url = ParsedURL(
        url="https://huggingface.co/google-bert/bert-base-uncased",
        category=URLCategory.MODEL,
        name="google-bert/bert-base-uncased",
        platform="huggingface",
    )
    context = ModelContext(model_url=model_url)
    
    size_metric = SizeScoreMetric()
    estimated_size = size_metric._estimate_model_size(context)
    
    # bert-base-uncased should be ~0.44 GB
    assert estimated_size == pytest.approx(0.44, abs=0.01)


def test_estimate_whisper_tiny():
    """Test size estimation for whisper-tiny from sample.txt."""
    model_url = ParsedURL(
        url="https://huggingface.co/openai/whisper-tiny",
        category=URLCategory.MODEL,
        name="openai/whisper-tiny",
        platform="huggingface",
    )
    context = ModelContext(model_url=model_url)
    
    size_metric = SizeScoreMetric()
    estimated_size = size_metric._estimate_model_size(context)
    
    # whisper-tiny should be ~0.075 GB
    assert estimated_size == pytest.approx(0.075, abs=0.01)


def test_estimate_audience_classifier():
    """Test size estimation for audience classifier from sample.txt."""
    model_url = ParsedURL(
        url="https://huggingface.co/parvk11/audience_classifier_model",
        category=URLCategory.MODEL,
        name="parvk11/audience_classifier_model",
        platform="huggingface",
    )
    context = ModelContext(model_url=model_url)
    
    size_metric = SizeScoreMetric()
    estimated_size = size_metric._estimate_model_size(context)
    
    # Unknown classifier matches "classifier" -> 0.1 GB
    assert estimated_size == pytest.approx(0.1, abs=0.01)


def test_device_score_within_limit(size_metric):
    """Test device score calculation when model is within device limit."""
    score = size_metric._calculate_device_score(
        0.5, 2.0
    )  # 0.5GB on 2GB device
    
    # Uses sigmoid function: should be high but not exactly 1.0
    assert 0.8 < score <= 1.0


def test_device_score_at_limit(size_metric):
    """Test device score calculation when model is at limit."""
    score = size_metric._calculate_device_score(2.0, 2.0)  # 2GB on 2GB device
    
    # Uses sigmoid at ratio=1.0, approximately 0.5
    assert 0.4 < score < 0.6


def test_device_score_exceeds_limit(size_metric):
    """Test device score calculation when model exceeds limit."""
    score = size_metric._calculate_device_score(5.0, 2.0)  # 5GB on 2GB device
    
    # Should be lower score when exceeds limit (ratio=2.5)
    assert 0.1 < score < 0.4


def test_device_score_far_exceeds_limit(size_metric):
    """Test device score calculation when model far exceeds limit."""
    score = size_metric._calculate_device_score(
        20.0, 2.0
    )  # 20GB on 2GB device
    
    # Should be very low score when far exceeds limit
    assert score < 0.2


def test_compute_bert_base_uncased(size_metric, config):
    """Test full compute for bert-base-uncased."""
    model_url = ParsedURL(
        url="https://huggingface.co/google-bert/bert-base-uncased",
        category=URLCategory.MODEL,
        name="google-bert/bert-base-uncased",
        platform="huggingface",
    )
    context = ModelContext(model_url=model_url)
    
    result = size_metric.compute(context, config)
    
    # Should have valid score between 0 and 1
    assert 0 <= result.score <= 1.0
    assert result.latency >= 0
    
    # bert-base (0.44GB) should have good scores on all devices
    # except raspberry_pi
    size_scores = size_metric._calculate_size_scores(context, config)
    assert size_scores.aws_server > 0.99  # Well below 64GB limit
    assert size_scores.desktop_pc > 0.98  # Well below 16GB limit
    assert size_scores.jetson_nano > 0.93  # Below 4GB limit
    assert 0.7 < size_scores.raspberry_pi < 0.8  # Exceeds 1GB limit


def test_compute_whisper_tiny(size_metric, config):
    """Test full compute for whisper-tiny."""
    model_url = ParsedURL(
        url="https://huggingface.co/openai/whisper-tiny",
        category=URLCategory.MODEL,
        name="openai/whisper-tiny",
        platform="huggingface",
    )
    context = ModelContext(model_url=model_url)
    
    result = size_metric.compute(context, config)
    
    # Should have valid score between 0 and 1
    assert 0 <= result.score <= 1.0
    assert result.latency >= 0
    
    # whisper-tiny (0.075GB) should have high scores on all devices
    size_scores = size_metric._calculate_size_scores(context, config)
    assert size_scores.aws_server > 0.99
    assert size_scores.desktop_pc > 0.99
    assert size_scores.jetson_nano > 0.99
    assert size_scores.raspberry_pi > 0.95  # Well below 1GB limit


def test_size_scores_return_type(size_metric, config):
    """Test that size scores returns SizeScore with 4 device scores."""
    model_url = ParsedURL(
        url="https://huggingface.co/google-bert/bert-base-uncased",
        category=URLCategory.MODEL,
        name="google-bert/bert-base-uncased",
        platform="huggingface",
    )
    context = ModelContext(model_url=model_url)
    
    size_scores = size_metric._calculate_size_scores(context, config)
    
    # Verify it's a SizeScore object with all 4 fields
    assert isinstance(size_scores, SizeScore)
    assert hasattr(size_scores, 'raspberry_pi')
    assert hasattr(size_scores, 'jetson_nano')
    assert hasattr(size_scores, 'desktop_pc')
    assert hasattr(size_scores, 'aws_server')
    
    # All scores should be between 0 and 1
    assert 0 <= size_scores.raspberry_pi <= 1.0
    assert 0 <= size_scores.jetson_nano <= 1.0
    assert 0 <= size_scores.desktop_pc <= 1.0
    assert 0 <= size_scores.aws_server <= 1.0


def test_average_score_calculation(size_metric, config):
    """Test that average score is correctly calculated from 4 device scores."""
    model_url = ParsedURL(
        url="https://huggingface.co/google-bert/bert-base-uncased",
        category=URLCategory.MODEL,
        name="google-bert/bert-base-uncased",
        platform="huggingface",
    )
    context = ModelContext(model_url=model_url)
    
    result = size_metric.compute(context, config)
    size_scores = size_metric._calculate_size_scores(context, config)
    
    # Verify average is calculated correctly
    expected_avg = (
        size_scores.raspberry_pi +
        size_scores.jetson_nano +
        size_scores.desktop_pc +
        size_scores.aws_server
    ) / 4.0
    
    assert result.score == pytest.approx(expected_avg, abs=0.001)
