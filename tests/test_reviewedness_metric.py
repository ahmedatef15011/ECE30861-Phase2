"""Tests for reviewedness metric.

Tests cover:
- Models with GitHub repositories (various review levels)
- Models without GitHub repositories
- Edge cases (empty repos, all reviewed, none reviewed)
- PR detection patterns
- File filtering (code vs weights)
"""

import pytest
from unittest.mock import Mock, patch

from src.metrics.reviewedness import ReviewednessMetric
from src.models import ModelContext, ParsedURL, URLCategory, MetricResult


@pytest.fixture
def reviewedness_metric():
    """Create reviewedness metric instance."""
    return ReviewednessMetric()


@pytest.fixture
def model_context_with_github():
    """Create model context with GitHub repository."""
    model_url = ParsedURL(
        url="https://huggingface.co/test/model",
        category=URLCategory.MODEL,
        name="test-model",
        platform="huggingface",
        owner="test",
        repo="model",
    )

    github_repo = ParsedURL(
        url="https://github.com/test-org/test-repo",
        category=URLCategory.CODE,
        name="test-repo",
        platform="github",
        owner="test-org",
        repo="test-repo",
    )

    return ModelContext(
        model_url=model_url,
        code_repos=[github_repo]
    )


@pytest.fixture
def model_context_without_github():
    """Create model context without GitHub repository."""
    model_url = ParsedURL(
        url="https://huggingface.co/test/model",
        category=URLCategory.MODEL,
        name="test-model",
        platform="huggingface",
        owner="test",
        repo="model",
    )

    return ModelContext(
        model_url=model_url,
        code_repos=[]
    )


class TestReviewednessMetric:
    """Test cases for ReviewednessMetric."""

    def test_metric_name(self, reviewedness_metric):
        """Test that metric has correct name."""
        assert reviewedness_metric.name == "reviewedness"

    def test_no_github_repo_returns_minus_one(
        self, reviewedness_metric, model_context_without_github
    ):
        """Test that models without GitHub repos return -1."""
        result = reviewedness_metric.compute(
            model_context_without_github, {}
        )

        assert isinstance(result, MetricResult)
        assert result.score == -1.0
        assert result.latency >= 0

    def test_empty_code_repos_returns_minus_one(
        self, reviewedness_metric
    ):
        """Test that empty code_repos list returns -1."""
        model_url = ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test-model",
            platform="huggingface",
            owner="test",
            repo="model",
        )
        context = ModelContext(model_url=model_url, code_repos=[])

        result = reviewedness_metric.compute(context, {})

        assert result.score == -1.0

    @patch('src.metrics.reviewedness.GitInspector')
    def test_clone_failure_returns_minus_one(
        self, mock_git_inspector, reviewedness_metric,
        model_context_with_github
    ):
        """Test that clone failure returns -1."""
        # Mock GitInspector to return None (clone failed)
        mock_instance = Mock()
        mock_instance.clone_repo.return_value = None
        mock_instance.cleanup = Mock()
        mock_git_inspector.return_value = mock_instance

        result = reviewedness_metric.compute(
            model_context_with_github, {}
        )

        assert result.score == -1.0

    @patch('src.metrics.reviewedness.GitInspector')
    @patch('src.metrics.reviewedness.subprocess.run')
    def test_all_code_reviewed(
        self, mock_subprocess, mock_git_inspector,
        reviewedness_metric, model_context_with_github
    ):
        """Test repository with 100% reviewed code."""
        # Mock successful clone
        mock_instance = Mock()
        mock_instance.clone_repo.return_value = "/tmp/test-repo"
        mock_instance.cleanup = Mock()
        mock_git_inspector.return_value = mock_instance

        # Mock git log output - all commits are PRs
        git_output = """abc123|Merge pull request #1
100\t0\tsrc/main.py

def456|PR #2: Add feature
50\t10\tsrc/utils.py

789ghi|Fixes #3
30\t5\ttests/test_main.py
"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = git_output
        mock_subprocess.return_value = mock_result

        result = reviewedness_metric.compute(
            model_context_with_github, {}
        )

        assert isinstance(result, MetricResult)
        assert result.score == 1.0  # 100% reviewed
        assert result.latency > 0

    @patch('src.metrics.reviewedness.GitInspector')
    @patch('src.metrics.reviewedness.subprocess.run')
    def test_no_code_reviewed(
        self, mock_subprocess, mock_git_inspector,
        reviewedness_metric, model_context_with_github
    ):
        """Test repository with 0% reviewed code."""
        # Mock successful clone
        mock_instance = Mock()
        mock_instance.clone_repo.return_value = "/tmp/test-repo"
        mock_instance.cleanup = Mock()
        mock_git_inspector.return_value = mock_instance

        # Mock git log output - no PR references
        git_output = """abc123|Initial commit
100\t0\tsrc/main.py

def456|Update code
50\t10\tsrc/utils.py

789ghi|Fix bug
30\t5\ttests/test_main.py
"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = git_output
        mock_subprocess.return_value = mock_result

        result = reviewedness_metric.compute(
            model_context_with_github, {}
        )

        assert isinstance(result, MetricResult)
        assert result.score == 0.0  # 0% reviewed
        assert result.latency > 0

    @patch('src.metrics.reviewedness.GitInspector')
    @patch('src.metrics.reviewedness.subprocess.run')
    def test_partial_code_reviewed(
        self, mock_subprocess, mock_git_inspector,
        reviewedness_metric, model_context_with_github
    ):
        """Test repository with partial code review."""
        # Mock successful clone
        mock_instance = Mock()
        mock_instance.clone_repo.return_value = "/tmp/test-repo"
        mock_instance.cleanup = Mock()
        mock_git_inspector.return_value = mock_instance

        # Mock git log - 60 lines reviewed, 40 not reviewed
        git_output = """abc123|Merge pull request #1
60\t0\tsrc/main.py

def456|Direct commit
40\t0\tsrc/utils.py
"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = git_output
        mock_subprocess.return_value = mock_result

        result = reviewedness_metric.compute(
            model_context_with_github, {}
        )

        assert isinstance(result, MetricResult)
        assert result.score == 0.6  # 60% reviewed
        assert result.latency > 0

    @patch('src.metrics.reviewedness.GitInspector')
    @patch('src.metrics.reviewedness.subprocess.run')
    def test_filters_weight_files(
        self, mock_subprocess, mock_git_inspector,
        reviewedness_metric, model_context_with_github
    ):
        """Test that weight files are excluded from analysis."""
        # Mock successful clone
        mock_instance = Mock()
        mock_instance.clone_repo.return_value = "/tmp/test-repo"
        mock_instance.cleanup = Mock()
        mock_git_inspector.return_value = mock_instance

        # Include weight files that should be ignored
        git_output = """abc123|Merge pull request #1
100\t0\tsrc/main.py
1000\t0\tmodels/weights.pt
500\t0\tmodels/model.bin

def456|Direct commit
50\t0\tsrc/utils.py
"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = git_output
        mock_subprocess.return_value = mock_result

        result = reviewedness_metric.compute(
            model_context_with_github, {}
        )

        # Only .py files counted: 100 reviewed, 50 not reviewed
        # Score should be 100/(100+50) = 0.667
        assert isinstance(result, MetricResult)
        assert abs(result.score - 0.667) < 0.01

    @patch('src.metrics.reviewedness.GitInspector')
    @patch('src.metrics.reviewedness.subprocess.run')
    def test_pr_pattern_detection(
        self, mock_subprocess, mock_git_inspector,
        reviewedness_metric, model_context_with_github
    ):
        """Test various PR reference patterns are detected."""
        # Mock successful clone
        mock_instance = Mock()
        mock_instance.clone_repo.return_value = "/tmp/test-repo"
        mock_instance.cleanup = Mock()
        mock_git_inspector.return_value = mock_instance

        # Test different PR patterns
        git_output = """abc123|Merge pull request #123 from branch
10\t0\tsrc/file1.py

def456|PR #456: Feature
10\t0\tsrc/file2.py

789ghi|Fixes (#789)
10\t0\tsrc/file3.py

jkl012|Issue #1001
10\t0\tsrc/file4.py

mno345|Direct commit no PR
10\t0\tsrc/file5.py
"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = git_output
        mock_subprocess.return_value = mock_result

        result = reviewedness_metric.compute(
            model_context_with_github, {}
        )

        # 4 commits with PR refs, 1 without
        # Score should be 40/50 = 0.8
        assert isinstance(result, MetricResult)
        assert result.score == 0.8

    @patch('src.metrics.reviewedness.GitInspector')
    @patch('src.metrics.reviewedness.subprocess.run')
    def test_git_command_timeout(
        self, mock_subprocess, mock_git_inspector,
        reviewedness_metric, model_context_with_github
    ):
        """Test handling of git command timeout."""
        # Mock successful clone
        mock_instance = Mock()
        mock_instance.clone_repo.return_value = "/tmp/test-repo"
        mock_instance.cleanup = Mock()
        mock_git_inspector.return_value = mock_instance

        # Mock timeout
        import subprocess
        mock_subprocess.side_effect = subprocess.TimeoutExpired(
            cmd="git log", timeout=60
        )

        result = reviewedness_metric.compute(
            model_context_with_github, {}
        )

        # Should return -1 on error
        assert result.score == -1.0

    @patch('src.metrics.reviewedness.GitInspector')
    @patch('src.metrics.reviewedness.subprocess.run')
    def test_git_command_failure(
        self, mock_subprocess, mock_git_inspector,
        reviewedness_metric, model_context_with_github
    ):
        """Test handling of git command failure."""
        # Mock successful clone
        mock_instance = Mock()
        mock_instance.clone_repo.return_value = "/tmp/test-repo"
        mock_instance.cleanup = Mock()
        mock_git_inspector.return_value = mock_instance

        # Mock failed git command
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "fatal: not a git repository"
        mock_subprocess.return_value = mock_result

        result = reviewedness_metric.compute(
            model_context_with_github, {}
        )

        # Should return -1 on error
        assert result.score == -1.0

    @patch('src.metrics.reviewedness.GitInspector')
    @patch('src.metrics.reviewedness.subprocess.run')
    def test_empty_repository(
        self, mock_subprocess, mock_git_inspector,
        reviewedness_metric, model_context_with_github
    ):
        """Test handling of empty repository (no commits)."""
        # Mock successful clone
        mock_instance = Mock()
        mock_instance.clone_repo.return_value = "/tmp/test-repo"
        mock_instance.cleanup = Mock()
        mock_git_inspector.return_value = mock_instance

        # Mock empty git log
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_subprocess.return_value = mock_result

        result = reviewedness_metric.compute(
            model_context_with_github, {}
        )

        # Empty repo should return 0.0
        assert result.score == 0.0

    @patch('src.metrics.reviewedness.GitInspector')
    @patch('src.metrics.reviewedness.subprocess.run')
    def test_only_weight_files_in_repo(
        self, mock_subprocess, mock_git_inspector,
        reviewedness_metric, model_context_with_github
    ):
        """Test repo with only weight files (no code)."""
        # Mock successful clone
        mock_instance = Mock()
        mock_instance.clone_repo.return_value = "/tmp/test-repo"
        mock_instance.cleanup = Mock()
        mock_git_inspector.return_value = mock_instance

        # Only weight files
        git_output = """abc123|Add weights
1000\t0\tmodel.pt
500\t0\tweights.bin
"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = git_output
        mock_subprocess.return_value = mock_result

        result = reviewedness_metric.compute(
            model_context_with_github, {}
        )

        # No code files, should return 0.0
        assert result.score == 0.0

    def test_is_code_file_python(self, reviewedness_metric):
        """Test Python files are recognized as code."""
        assert reviewedness_metric._is_code_file(
            "src/main.py", "/tmp/repo"
        )

    def test_is_code_file_javascript(self, reviewedness_metric):
        """Test JavaScript files are recognized as code."""
        assert reviewedness_metric._is_code_file(
            "src/app.js", "/tmp/repo"
        )

    def test_is_code_file_config(self, reviewedness_metric):
        """Test config files are recognized as code."""
        assert reviewedness_metric._is_code_file(
            "config.json", "/tmp/repo"
        )
        assert reviewedness_metric._is_code_file(
            "settings.yaml", "/tmp/repo"
        )

    def test_is_code_file_weight_excluded(self, reviewedness_metric):
        """Test weight files are excluded."""
        assert not reviewedness_metric._is_code_file(
            "model.pt", "/tmp/repo"
        )
        assert not reviewedness_metric._is_code_file(
            "weights.bin", "/tmp/repo"
        )
        assert not reviewedness_metric._is_code_file(
            "checkpoint.safetensors", "/tmp/repo"
        )

    def test_is_code_file_data_excluded(self, reviewedness_metric):
        """Test data files are excluded."""
        assert not reviewedness_metric._is_code_file(
            "data.pkl", "/tmp/repo"
        )
        assert not reviewedness_metric._is_code_file(
            "dataset.npz", "/tmp/repo"
        )

    def test_is_code_file_no_extension(self, reviewedness_metric):
        """Test files without extension are excluded."""
        assert not reviewedness_metric._is_code_file(
            "README", "/tmp/repo"
        )

    def test_is_reviewed_commit_pr_number(self, reviewedness_metric):
        """Test PR number patterns are detected."""
        assert reviewedness_metric._is_reviewed_commit(
            "Fix bug #123", "/tmp/repo", "abc"
        )
        assert reviewedness_metric._is_reviewed_commit(
            "PR #456: Add feature", "/tmp/repo", "def"
        )
        assert reviewedness_metric._is_reviewed_commit(
            "Fixes (#789)", "/tmp/repo", "ghi"
        )

    def test_is_reviewed_commit_merge(self, reviewedness_metric):
        """Test merge commit patterns are detected."""
        assert reviewedness_metric._is_reviewed_commit(
            "Merge pull request #123", "/tmp/repo", "abc"
        )
        assert reviewedness_metric._is_reviewed_commit(
            "Merge branch 'feature'", "/tmp/repo", "def"
        )

    def test_is_reviewed_commit_no_pr(self, reviewedness_metric):
        """Test commits without PR are not marked as reviewed."""
        assert not reviewedness_metric._is_reviewed_commit(
            "Direct commit", "/tmp/repo", "abc"
        )
        assert not reviewedness_metric._is_reviewed_commit(
            "Update file", "/tmp/repo", "def"
        )

    def test_metric_returns_metric_result(
        self, reviewedness_metric, model_context_without_github
    ):
        """Test that compute returns MetricResult object."""
        result = reviewedness_metric.compute(
            model_context_without_github, {}
        )

        assert isinstance(result, MetricResult)
        assert hasattr(result, 'score')
        assert hasattr(result, 'latency')
        assert isinstance(result.score, float)
        assert isinstance(result.latency, int)

    def test_score_in_valid_range(
        self, reviewedness_metric, model_context_without_github
    ):
        """Test that score is always in valid range."""
        result = reviewedness_metric.compute(
            model_context_without_github, {}
        )

        assert -1.0 <= result.score <= 1.0

    def test_latency_is_positive(
        self, reviewedness_metric, model_context_without_github
    ):
        """Test that latency is always positive."""
        result = reviewedness_metric.compute(
            model_context_without_github, {}
        )

        assert result.latency >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
