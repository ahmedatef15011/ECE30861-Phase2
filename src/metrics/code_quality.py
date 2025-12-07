from typing import Any, Dict

from ..git_inspect import GitInspector
from ..models import MetricResult, ModelContext
from ..utils import measure_time
from .base import BaseMetric
from .fallback_scoring import FallbackScorer


class CodeQualityMetric(BaseMetric):
    """Metric for evaluating quality of linked code repositories."""

    @property
    def name(self) -> str:
        return "code_quality"

    def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        # compute code quality score
        with measure_time() as get_latency:
            score = self._calculate_code_quality_score(context, config)

        return MetricResult(score=score, latency=get_latency())

    def _calculate_code_quality_score(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> float:
        """Calculate code quality using fast heuristics.
        
        Avoids slow linting tools - uses structure analysis instead.
        """
        # First: try to estimate from HF metadata (fastest path)
        if context.hf_info:
            github_url = context.hf_info.get("github_url")
            file_count = context.hf_info.get("file_count", 0)
            
            # If model has linked GitHub + reasonable file count, assume quality
            if github_url and file_count > 5:
                return 0.7  # Good baseline for linked code
            elif file_count > 10:
                return 0.6  # Has files, likely some code structure
        
        if not context.code_repos:
            # Use enhanced fallback scoring based on README
            fallback = FallbackScorer(context.readme_content)
            score, details = fallback.get_code_fallback_score()
            return score

        git_inspector = GitInspector()
        try:
            code_repo = context.code_repos[0]
            if code_repo.platform == "github":
                repo_path = git_inspector.clone_repo(code_repo)
                if repo_path:
                    return self._fast_code_quality_check(repo_path)
        finally:
            git_inspector.cleanup()

        # Fallback if cloning failed
        fallback = FallbackScorer(context.readme_content)
        score, details = fallback.get_code_fallback_score()
        return score

    def _fast_code_quality_check(self, repo_path: str) -> float:
        """Fast code quality estimation without running linters."""
        import os

        score = 0.5  # baseline

        # Check for tests folder (+0.15)
        has_tests = (
            os.path.exists(os.path.join(repo_path, "tests")) or
            os.path.exists(os.path.join(repo_path, "test"))
        )
        if has_tests:
            score += 0.15

        # Check for CI config (+0.15)
        ci_indicators = [
            os.path.join(repo_path, ".github", "workflows"),
            os.path.join(repo_path, ".travis.yml"),
            os.path.join(repo_path, ".circleci"),
            os.path.join(repo_path, "azure-pipelines.yml"),
        ]
        if any(os.path.exists(p) for p in ci_indicators):
            score += 0.15

        # Check for linting config (+0.1) - presence suggests quality focus
        lint_configs = [".flake8", "mypy.ini", ".mypy.ini", ".pylintrc"]
        if any(os.path.exists(os.path.join(repo_path, c)) for c in lint_configs):
            score += 0.1

        # Check for requirements.txt or pyproject.toml (+0.05)
        dep_files = ["requirements.txt", "pyproject.toml", "setup.py"]
        if any(os.path.exists(os.path.join(repo_path, f)) for f in dep_files):
            score += 0.05

        # Check for README (+0.05)
        readme_files = ["README.md", "README.rst", "README.txt", "README"]
        if any(os.path.exists(os.path.join(repo_path, f)) for f in readme_files):
            score += 0.05

        return min(1.0, score)

    def _analyze_code_repository(
        self, repo_path: str, inspector: GitInspector, thresholds: Dict[str, Any]
    ) -> float:
        """Analyze a code repository for quality indicators."""
        analysis = inspector.analyze_repository(repo_path)

        score = 0.0

        # repository structure quality (25% of score)
        structure_analysis = analysis.get("structure_analysis", {})
        structure_score = structure_analysis.get("structure_score", 0.0)
        score += structure_score * 0.25

        # documentation quality (25% of score)
        doc_analysis = analysis.get("documentation_analysis", {})
        doc_score = doc_analysis.get("documentation_score", 0.0)
        score += doc_score * 0.25

        # file organization quality (25% of score)
        file_analysis = analysis.get("file_analysis", {})
        file_score = self._calculate_file_quality_score(file_analysis, thresholds)
        score += file_score * 0.25

        # commit quality (25% of score)
        commit_analysis = analysis.get("commit_analysis", {})
        commit_score = self._calculate_commit_quality_score(commit_analysis)
        score += commit_score * 0.25

        return min(1.0, score)

    def _calculate_file_quality_score(
        self, file_analysis: Dict[str, Any], thresholds: Dict[str, Any]
    ) -> float:
        """Calculate score based on file structure and organization."""
        score = 0.0

        python_files = file_analysis.get("python_files", 0)
        test_files = file_analysis.get("test_files", 0)
        total_lines = file_analysis.get("total_lines_of_code", 0)

        # test coverage estimate (based on test file ratio)
        if python_files > 0:
            test_ratio = test_files / python_files
            min_coverage = thresholds.get("min_test_coverage", 0.5)

            if test_ratio >= min_coverage:
                score += 0.4
            elif test_ratio >= min_coverage * 0.5:
                score += 0.2

        # code organization (reasonable file count and LOC)
        if 10 <= python_files <= 100:
            score += 0.3
        elif python_files > 0:
            score += 0.1

        # lines of code 
        if 1000 <= total_lines <= 50000: 
            score += 0.3
        elif total_lines > 0:
            score += 0.1

        return score

    def _calculate_commit_quality_score(self, commit_analysis: Dict[str, Any]) -> float:
        """Calculate score based on commit history quality."""
        score = 0.0

        total_commits = commit_analysis.get("total_commits", 0)
        recent_commits = commit_analysis.get("recent_commits", 0)
        avg_frequency = commit_analysis.get("avg_commit_frequency", 0)

        # regular commit activity
        if total_commits >= 20:
            score += 0.3
        elif total_commits >= 5:
            score += 0.2
        elif total_commits >= 1:
            score += 0.1

        # recent activity
        if recent_commits >= 5:
            score += 0.3
        elif recent_commits >= 1:
            score += 0.2

        # commits per day
        if avg_frequency >= 0.1:  # 1 commit per 10 days
            score += 0.4
        elif avg_frequency > 0:
            score += 0.2

        return score
