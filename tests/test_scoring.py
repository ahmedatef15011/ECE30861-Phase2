import tempfile
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from src.models import (MetricResult, ModelContext, ParsedURL, SizeScore,
                        URLCategory)
from src.scoring import MetricScorer


@pytest.fixture
def temp_config():
    config_data = {
        "metric_weights": {
            "ramp_up_time": 0.2,
            "bus_factor": 0.2,
            "performance_claims": 0.2,
            "license": 0.2,
            "size_score": 0.2,
        },
        "thresholds": {},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    yield temp_path
    import os

    os.unlink(temp_path)


@pytest.fixture
def model_context():
    model_url = ParsedURL(
        url="https://huggingface.co/test/model",
        category=URLCategory.MODEL,
        name="test/model",
        platform="huggingface",
    )
    return ModelContext(model_url=model_url)


def test_scorer_init_with_config(temp_config):
    scorer = MetricScorer(temp_config)

    assert scorer.config["metric_weights"]["ramp_up_time"] == 0.2
    # Phase 1: 8 metrics + Phase 2: reproducibility + reviewedness = 10
    assert len(scorer.metrics) == 10


def test_scorer_init_no_config():
    scorer = MetricScorer("nonexistent.yaml")

    assert "metric_weights" in scorer.config
    # Phase 1: 8 metrics + Phase 2: reproducibility + reviewedness = 10
    assert len(scorer.metrics) == 10


@patch("src.scoring.HuggingFaceAPI")
def test_scorer_init_hf_api(mock_hf_api, temp_config):
    scorer = MetricScorer(temp_config)
    assert scorer.hf_api is not None


def test_enrich_context(temp_config, model_context):
    scorer = MetricScorer(temp_config)

    scorer.hf_api.get_model_info = AsyncMock(return_value={"id": "test/model"})
    scorer.hf_api.get_readme_content = AsyncMock(return_value="# Test README")
    scorer.hf_api.get_model_config = AsyncMock(return_value={"config.json": {}})

    await scorer._enrich_context(model_context)

    assert model_context.hf_info == {"id": "test/model"}
    assert model_context.readme_content == "# Test README"
    assert model_context.config_data == {"config.json": {}}


def test_calculate_net_score(temp_config):
    scorer = MetricScorer(temp_config)

    metric_results = {
        "ramp_up_time": MetricResult(score=0.8, latency=100),
        "bus_factor": MetricResult(score=0.6, latency=200),
        "performance_claims": MetricResult(score=0.7, latency=150),
        "license": MetricResult(score=1.0, latency=50),
        "size_score": SizeScore(
            raspberry_pi=0.5, jetson_nano=0.7, desktop_pc=0.9, aws_server=1.0
        ),
        "size_score_latency": 300,
    }

    net_score = scorer._calculate_net_score(metric_results)

    # weighted average
    expected = 0.8 * 0.2 + 0.6 * 0.2 + 0.7 * 0.2 + 1.0 * 0.2 + 0.775 * 0.2
    assert abs(net_score - expected) < 0.01


def test_score_model_integration(temp_config, model_context):
    scorer = MetricScorer(temp_config)

    scorer.hf_api.get_model_info = AsyncMock(return_value={"id": "test/model"})
    scorer.hf_api.get_readme_content = AsyncMock(return_value="# Test README")
    scorer.hf_api.get_model_config = AsyncMock(return_value=None)

    for metric in scorer.metrics:
        if metric.name == "size_score":
            scorer._compute_size_metric_with_latency = AsyncMock(
                return_value=(
                    SizeScore(
                        raspberry_pi=0.5,
                        jetson_nano=0.7,
                        desktop_pc=0.9,
                        aws_server=1.0,
                    ),
                    300,
                )
            )
        else:
            metric.compute = AsyncMock(
                return_value=MetricResult(score=0.7, latency=100)
            )

    result = scorer.score_model(model_context)

    assert result.name == "test/model"
    assert result.category == "MODEL"
    assert 0.0 <= result.net_score <= 1.0
    assert result.net_score_latency >= 0
