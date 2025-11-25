"""
Tests for Hugging Face API integration.
"""

from unittest.mock import Mock, patch

import pytest

from src.hf_api import HuggingFaceAPI
from src.models import ParsedURL, URLCategory


@pytest.fixture
def hf_api():
    """Create HuggingFaceAPI instance for testing."""
    return HuggingFaceAPI()


@pytest.fixture
def model_url():
    """Create model URL for testing."""
    return ParsedURL(
        url="https://huggingface.co/test/model",
        category=URLCategory.MODEL,
        name="test/model",
        platform="huggingface",
        owner="test",
        repo="model",
    )


@pytest.fixture
def dataset_url():
    """Create dataset URL for testing."""
    return ParsedURL(
        url="https://huggingface.co/datasets/test/dataset",
        category=URLCategory.DATASET,
        name="test/dataset",
        platform="huggingface",
        owner="test",
        repo="dataset",
    )


@patch("src.hf_api.list_repo_files")
def test_get_model_info_success(mock_list_files, hf_api, model_url):
    """Test successful model info retrieval."""
    # Mock API responses
    mock_model_info = Mock()
    mock_model_info.id = "test/model"
    mock_model_info.downloads = 100
    mock_model_info.likes = 10
    mock_model_info.tags = ["pytorch"]

    mock_list_files.return_value = ["config.json", "pytorch_model.bin"]
    hf_api.api.model_info = Mock(return_value=mock_model_info)

    # Test async function
    import asyncio

    result = asyncio.run(hf_api.get_model_info(model_url))

    assert result is not None
    assert result["id"] == "test/model"
    assert result["downloads"] == 100
    assert result["likes"] == 10
    assert result["files"] == ["config.json", "pytorch_model.bin"]


def test_get_model_info_non_hf_url(hf_api):
    """Test get_model_info with non-HF URL."""
    non_hf_url = ParsedURL(
        url="https://github.com/test/repo",
        category=URLCategory.CODE,
        name="test/repo",
        platform="github",
        owner="test",
        repo="repo",
    )

    import asyncio

    result = asyncio.run(hf_api.get_model_info(non_hf_url))
    assert result is None


@patch("builtins.open")
@patch("src.hf_api.hf_hub_download")
def test_download_file_success(mock_download, mock_open, hf_api):
    """Test successful file download."""
    mock_download.return_value = "/tmp/test_file"
    mock_open.return_value.__enter__.return_value.read.return_value = "test content"

    result = hf_api.download_file("test/model", "README.md")
    assert result == "test content"


def test_download_file_failure(hf_api):
    """Test file download failure."""
    with patch("src.hf_api.hf_hub_download", side_effect=Exception("Download failed")):
        result = hf_api.download_file("test/model", "README.md")
        assert result is None


def test_get_readme_content(hf_api, model_url):
    """Test README content retrieval."""
    with patch.object(hf_api, "download_file", return_value="# Test README"):
        result = hf_api.get_readme_content(model_url)
        assert result == "# Test README"


def test_get_model_config(hf_api, model_url):
    """Test model config retrieval."""
    with patch.object(hf_api, "download_file") as mock_download:
        mock_download.side_effect = lambda repo, filename, is_dataset: (
            '{"test": "config"}' if filename == "config.json" else None
        )

        result = hf_api.get_model_config(model_url)
        assert result is not None
        assert "config.json" in result
        assert result["config.json"]["test"] == "config"
