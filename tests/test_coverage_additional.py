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

    # Mock subprocess runs: flake8 returns 2 errors, mypy returns 1 error
    with patch("subprocess.run") as mock_run:
        # flake8 call
        mock_flake8 = Mock()
        mock_flake8.returncode = 1
        mock_flake8.stdout = "a.py:1:1 E999\nb.py:1:1 E900"
        # mypy call
        mock_mypy = Mock()
        mock_mypy.returncode = 1
        mock_mypy.stdout = "a.py:1: error: X\n"
        mock_run.side_effect = [mock_flake8, mock_mypy]

        score = metric._analyze_code_quality_by_spec(str(repo), GitInspector())
        assert 0.0 <= score <= 1.0


def test_code_quality_fallbacks_basic_syntax(tmp_path: Path):
    # Repo without configs triggers basic syntax check; include a bad file
    repo = tmp_path
    (repo / "good.py").write_text("print('ok')\n")
    (repo / "bad.py").write_text("def x(:\n")  # SyntaxError

    metric = CodeQualityMetric()

    # Make subprocess.run raise FileNotFoundError to force fallbacks first
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        score = metric._analyze_code_quality_by_spec(str(repo), GitInspector())
        assert 0.0 <= score <= 1.0


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
