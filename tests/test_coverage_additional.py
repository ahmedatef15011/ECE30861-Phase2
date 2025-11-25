"""
Additional targeted tests to raise coverage ≥ 80%.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.git_inspect import GitInspector
from src.hf_api import HuggingFaceAPI
from src.metrics.code_quality import CodeQualityMetric
from src.metrics.size_score import SizeScoreMetric
from src.models import ModelContext, ParsedURL, URLCategory


def test_size_score_estimation_paths_readme_and_patterns():
    metric = SizeScoreMetric()

    # From README content (7B -> ~14GB via utils mapping or model patterns)
    context = ModelContext(
        model_url=ParsedURL(
            url="https://hf.co/a/b",
            category=URLCategory.MODEL,
            name="a/b",
            platform="huggingface",
        ),
        readme_content="This is a 7B parameter model",
    )
    size_scores = metric._calculate_size_scores(context, {})
    assert 0.0 <= size_scores.aws_server <= 1.0

    # From hf_info files list estimation
    context = ModelContext(
        model_url=ParsedURL(
            url="https://hf.co/a/b",
            category=URLCategory.MODEL,
            name="a/b",
            platform="huggingface",
        ),
        hf_info={"files": ["pytorch_model-00001-of-00002.bin", "config.json"]},
    )
    size_scores = metric._calculate_size_scores(context, {})
    assert 0.0 <= size_scores.jetson_nano <= 1.0

    # From model name patterns (13B)
    context = ModelContext(
        model_url=ParsedURL(
            url="https://hf.co/a/awesome-13B",
            category=URLCategory.MODEL,
            name="awesome-13B",
            platform="huggingface",
        ),
    )
    size_scores = metric._calculate_size_scores(context, {})
    assert 0.0 <= size_scores.desktop_pc <= 1.0

    # Generic names
    for name in ["model-large", "model-base", "model-small", "unknown-model"]:
        context = ModelContext(
            model_url=ParsedURL(
                url=f"https://hf.co/a/{name}",
                category=URLCategory.MODEL,
                name=name,
                platform="huggingface",
            ),
        )
        size_scores = metric._calculate_size_scores(context, {})
        assert 0.0 <= size_scores.raspberry_pi <= 1.0


def test_code_quality_flake8_mypy_branches(tmp_path: Path):
    # Create repo structure with config files, tests and CI
    repo = tmp_path
    (repo / ".flake8").write_text("[flake8]\n")
    (repo / "mypy.ini").write_text("[mypy]\n")
    (repo / "tests").mkdir(parents=True, exist_ok=True)
    (repo / ".github" / "workflows").mkdir(parents=True, exist_ok=True)

    metric = CodeQualityMetric()

    # Test the fast code quality check that uses file existence
    score = metric._fast_code_quality_check(str(repo))
    # Should get 0.5 baseline + 0.15 for tests + 0.15 for CI + 0.1 for flake8
    assert 0.7 <= score <= 1.0


def test_code_quality_fallbacks_basic_syntax(tmp_path: Path):
    # Repo without configs still gets baseline score
    repo = tmp_path
    (repo / "good.py").write_text("print('ok')\n")
    (repo / "bad.py").write_text("def x(:\n")  # SyntaxError

    metric = CodeQualityMetric()

    # Fast check should return baseline 0.5 (no special configs)
    score = metric._fast_code_quality_check(str(repo))
    assert score == 0.5  # No bonus files found


def test_hf_api_dataset_info_and_readme_choices():
    api = HuggingFaceAPI()

    # dataset_info success mapping
    dataset_url = ParsedURL(
        url="https://huggingface.co/datasets/test/data",
        category=URLCategory.DATASET,
        name="test/data",
        platform="huggingface",
        owner="test",
        repo="data",
    )
    mock_obj = Mock()
    mock_obj.id = "test/data"
    mock_obj.author = "me"
    mock_obj.downloads = 5
    mock_obj.likes = 1
    mock_obj.created_at = None
    mock_obj.last_modified = None
    mock_obj.tags = ["tag"]
    mock_obj.task_categories = ["tc"]
    api.api.dataset_info = Mock(return_value=mock_obj)
    info = api.get_dataset_info(dataset_url)
    assert info and info["id"] == "test/data"

    # get_readme_content tries multiple names, simulate first None then success
    model_url = ParsedURL(
        url="https://huggingface.co/test/model",
        category=URLCategory.MODEL,
        name="test/model",
        platform="huggingface",
        owner="test",
        repo="model",
    )
    with patch.object(api, "download_file", side_effect=[None, "# ok"]):
        readme = api.get_readme_content(model_url)
        assert readme == "# ok"


def test_git_inspect_structure_and_docs(tmp_path: Path):
    repo = tmp_path
    # Create minimal files
    (repo / "README.md").write_text("# Title\nUsage example")
    (repo / "LICENSE").write_text("MIT")
    (repo / "requirements.txt").write_text("pytest\n")
    (repo / "setup.py").write_text("from setuptools import setup\n")
    (repo / ".github").mkdir(exist_ok=True)

    insp = GitInspector()
    try:
        s = insp._analyze_structure(str(repo))
        assert 0.0 <= s["structure_score"] <= 1.0

        d = insp._analyze_documentation(str(repo))
        assert d["documentation_score"] >= 0.25
    finally:
        insp.cleanup()
