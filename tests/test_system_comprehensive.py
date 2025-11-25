"""
Comprehensive system-level tests to improve coverage.
"""

import os
import tempfile
from unittest.mock import Mock, Mock, patch

import pytest

from src.git_inspect import GitInspector
from src.hf_api import HuggingFaceAPI
from src.logging_utils import get_logger, setup_logging
from src.models import ParsedURL, URLCategory
from src.utils import extract_model_size_from_text, parse_license_from_readme


# Git Inspector comprehensive tests
def test_git_inspector_analyze_empty_repo():
    """Test analyzing empty repository."""
    inspector = GitInspector()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create empty git repo
        import subprocess

        subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True)

        analysis = inspector.analyze_repository(temp_dir)

        assert analysis["commit_analysis"]["total_commits"] == 0
        assert analysis["contributor_analysis"]["unique_authors"] == 0
        assert analysis["file_analysis"]["total_files"] >= 0


def test_git_inspector_cache_reuse():
    """Test that git inspector reuses cached clones."""
    inspector = GitInspector()

    github_url = ParsedURL(
        url="https://github.com/test/repo",
        category=URLCategory.CODE,
        name="test/repo",
        platform="github",
        owner="test",
        repo="repo",
    )

    with patch("src.git_inspect.porcelain.clone") as mock_clone, patch(
        "os.path.exists"
    ) as mock_exists:

        # First call - directory doesn't exist, should clone
        mock_exists.return_value = False
        inspector.clone_repo(github_url)
        assert mock_clone.called

        # Second call - directory exists, should use cache
        mock_clone.reset_mock()
        mock_exists.return_value = True
        inspector.clone_repo(github_url)
        assert not mock_clone.called


# HF API comprehensive tests
def test_hf_api_repository_not_found():
    """Test HF API with repository not found error."""
    api = HuggingFaceAPI()

    model_url = ParsedURL(
        url="https://huggingface.co/nonexistent/model",
        category=URLCategory.MODEL,
        name="nonexistent/model",
        platform="huggingface",
        owner="nonexistent",
        repo="model",
    )

    with patch.object(api.api, "model_info") as mock_model_info:
        from huggingface_hub.errors import RepositoryNotFoundError
        from unittest.mock import Mock
        
        # Create a mock response object
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        
        mock_model_info.side_effect = RepositoryNotFoundError("Not found", response=mock_response)

        result = api.get_model_info(model_url)
        assert result is None


def test_hf_api_file_list_success():
    """Test HF API file listing success."""
    api = HuggingFaceAPI()

    model_url = ParsedURL(
        url="https://huggingface.co/test/model",
        category=URLCategory.MODEL,
        name="test/model",
        platform="huggingface",
        owner="test",
        repo="model",
    )

    with patch.object(api.api, "model_info") as mock_model_info, patch(
        "src.hf_api.list_repo_files"
    ) as mock_list_files:

        mock_model_info.return_value = Mock(id="test/model", downloads=1000, likes=50)
        mock_list_files.return_value = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
        ]

        result = api.get_model_info(model_url)

        assert result is not None
        assert result["file_count"] == 3
        assert "config.json" in result["files"]


# Logging utilities tests
def test_logging_setup_silent():
    """Test logging setup with silent level."""
    with patch.dict(os.environ, {"LOG_LEVEL": "0"}):
        logger = setup_logging()
        assert logger.level > 50  # Should be effectively silent


def test_logging_setup_with_file():
    """Test logging setup with file output."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        with patch.dict(os.environ, {"LOG_LEVEL": "1", "LOG_FILE": temp_path}):
            logger = setup_logging()
            logger.info("Test message")

            # Verify file was written
            with open(temp_path, "r") as f:
                content = f.read()
                assert "Test message" in content
    finally:
        os.unlink(temp_path)


def test_logging_get_logger():
    """Test get_logger function."""
    logger1 = get_logger()
    logger2 = get_logger()

    # Should return same logger instance
    assert logger1 is logger2
    assert logger1.name == "src"


# Utility function comprehensive tests
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Model has 13B parameters", 26.0),  # 13 * 2
        ("270M parameter model", 0.54),  # 270 * 0.002
        ("Size: 1.5GB", 1.5),  # Direct GB
        ("Model file: 512MB", 0.5),  # MB to GB
        ("Large language model", None),  # No size info
        ("The model contains 7 billion params", 14.0),  # Written out
        ("Small 125M model", 0.25),  # Small M model
    ],
)
def test_extract_model_size_comprehensive(text, expected):
    """Test model size extraction with various text patterns."""
    result = extract_model_size_from_text(text)
    if expected is None:
        assert result is None
    else:
        assert abs(result - expected) < 0.1


@pytest.mark.parametrize(
    "readme,expected_license",
    [
        ("""# Model\n## License\nMIT License""", "MIT License"),
        ("""# Model\n**License**: Apache 2.0""", "Apache 2.0"),
        ("""License: BSD-3-Clause""", "BSD-3-Clause"),
        ("""# Model\nNo license section""", None),
        ("", None),
        ("""## License\n[Apache 2.0](https://license.url)""", "Apache 2.0"),
    ],
)
def test_parse_license_comprehensive(readme, expected_license):
    """Test license parsing with various README formats."""
    result = parse_license_from_readme(readme)
    if expected_license is None:
        assert result is None
    else:
        assert expected_license in result


# CLI edge cases
def test_cli_no_models_found():
    """Test CLI behavior when no model URLs are found."""
    from src.urls import build_model_contexts

    # URLs with no models (only datasets and code)
    urls = ["https://huggingface.co/datasets/test/data", "https://github.com/test/code"]

    contexts = build_model_contexts(urls)
    assert len(contexts) == 0


def test_url_parsing_edge_cases():
    """Test URL parsing edge cases."""
    from src.urls import parse_url

    # Test with unknown platform - defaults to DATASET category
    url = parse_url("https://example.com/some/model")
    assert url.category == URLCategory.DATASET  # Unknown URLs default to DATASET
    assert url.platform == "unknown"

    # Single path part on HF
    url = parse_url("https://huggingface.co/model")
    assert url.name == "model"
    assert url.owner == "model"


def test_model_context_enrichment():
    """Test model context enrichment process."""
    from src.models import ModelContext, ParsedURL, URLCategory
    from src.scoring import MetricScorer

    scorer = MetricScorer()
    context = ModelContext(
        model_url=ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test/model",
            platform="huggingface",
        )
    )

    # Mock API methods
    scorer.hf_api.get_model_info = Mock(return_value={"id": "test/model"})
    scorer.hf_api.get_readme_content = Mock(return_value="# README")
    scorer.hf_api.get_model_config = Mock(return_value={"config.json": {}})

    # Test enrichment
    import asyncio

    scorer._enrich_context(context)

    assert context.hf_info == {"id": "test/model"}
    assert context.readme_content == "# README"
    assert context.config_data == {"config.json": {}}
