"""
Tests for CLI functionality.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from src.cli import process_urls, run_tests


@pytest.fixture
def temp_url_file():
    """Create temporary URL file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("https://huggingface.co/google/gemma-3-270m\n")
        f.write("https://huggingface.co/datasets/xlangai/AgentNet\n")
        f.write("https://github.com/SkyworkAI/Matrix-Game\n")
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


def test_process_urls_file_not_found():
    """Test process_urls with non-existent file."""
    with pytest.raises(SystemExit) as exc_info:
        process_urls("nonexistent_file.txt")
    assert exc_info.value.code == 1


@patch("src.cli._process_contexts")
@patch("src.cli.build_model_contexts")
def test_process_urls_success(mock_build_contexts, mock_asyncio_run, temp_url_file):
    """Test successful URL processing."""
    # Mock model contexts
    mock_context = Mock()
    mock_context.model_url.name = "test-model"
    mock_build_contexts.return_value = [mock_context]

    # Should not raise exception
    process_urls(temp_url_file)

    mock_build_contexts.assert_called_once()
    mock_asyncio_run.assert_called_once()


def test_process_urls_empty_file():
    """Test process_urls with empty file."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        temp_path = f.name

    try:
        with pytest.raises(SystemExit) as exc_info:
            process_urls(temp_path)
        assert exc_info.value.code == 1
    finally:
        os.unlink(temp_path)


@patch("subprocess.run")
def test_run_tests_success(mock_run):
    """Test successful test run."""
    mock_run.return_value = Mock(
        returncode=0, stdout="5 passed, 0 failed\nTOTAL 85%", stderr=""
    )

    with pytest.raises(SystemExit) as exc_info:
        run_tests()
    assert exc_info.value.code == 0


@patch("subprocess.run")
def test_run_tests_failure(mock_run):
    """Test test run with failures."""
    mock_run.return_value = Mock(
        returncode=1, stdout="3 passed, 2 failed\nTOTAL 75%", stderr=""
    )

    with pytest.raises(SystemExit) as exc_info:
        run_tests()
    assert exc_info.value.code == 1
