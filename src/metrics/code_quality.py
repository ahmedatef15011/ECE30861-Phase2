from typing import Any, Dict

from ..git_inspect import GitInspector
from ..models import MetricResult, ModelContext
from ..utils import measure_time
from .base import BaseMetric

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
        """Calculate code quality using lints, tests folder, and CI config."""
        if not context.code_repos:
            return 0.5

        total_score = 0.0
        repos_analyzed = 0

        git_inspector = GitInspector()

        try:
            # limit to first 2 repositories
            for code_repo in context.code_repos[:2]:
                if code_repo.platform == "github":
                    repo_path = git_inspector.clone_repo(code_repo)
                    if repo_path:
                        repo_score = self._analyze_code_quality_by_spec(
                            repo_path, git_inspector
                        )
                        total_score += repo_score
                        repos_analyzed += 1
                        break  # first avail repo
        finally:
            git_inspector.cleanup()

        if repos_analyzed == 0:
            return 0.5  # default

        return total_score / repos_analyzed

    def _analyze_code_quality_by_spec(
        self, repo_path: str, inspector: GitInspector
    ) -> float:
        """Analyze code quality using existing static-analysis hooks."""
        import glob
        import os
        import subprocess

        total_errors = 0

        # try to run existing static analysis tools
        try:
            # check if flake8 config exists and try to run
            flake8_configs = [".flake8", "setup.cfg", "tox.ini", "pyproject.toml"]
            has_flake8_config = any(
                os.path.exists(os.path.join(repo_path, config))
                for config in flake8_configs
            )

            if has_flake8_config:
                try:
                    result = subprocess.run(
                        ["flake8", repo_path],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=repo_path,
                    )
                    if result.returncode == 0:
                        total_errors += 0
                    else:
                        error_lines = [
                            line for line in result.stdout.split("\n") if line.strip()
                        ]
                        total_errors += len(error_lines)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    # fallback: estimate errors from file analysis
                    python_files = glob.glob(
                        os.path.join(repo_path, "**", "*.py"), recursive=True
                    )
                    total_errors += len(python_files) // 5  # conservative estimate

            # check for mypy config and try to run
            mypy_configs = ["mypy.ini", ".mypy.ini", "setup.cfg", "pyproject.toml"]
            has_mypy_config = any(
                os.path.exists(os.path.join(repo_path, config))
                for config in mypy_configs
            )

            if has_mypy_config:
                try:
                    # try to run mypy on the repository
                    result = subprocess.run(
                        ["mypy", repo_path],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=repo_path,
                    )
                    if result.returncode != 0:
                        # count mypy errors
                        error_lines = [
                            line
                            for line in result.stdout.split("\n")
                            if "error:" in line.lower()
                        ]
                        total_errors += len(error_lines)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    # fallback: estimate type errors
                    python_files = glob.glob(
                        os.path.join(repo_path, "**", "*.py"), recursive=True
                    )
                    total_errors += len(python_files) // 8  # Conservative estimate

            # if no linting configs found, use basic syntax checking
            if not has_flake8_config and not has_mypy_config:
                python_files = glob.glob(
                    os.path.join(repo_path, "**", "*.py"), recursive=True
                )
                for py_file in python_files[:20]:  # limit for performance
                    try:
                        with open(py_file, "r") as f:
                            content = f.read()
                            # syntax check
                            compile(content, py_file, "exec")
                    except SyntaxError:
                        total_errors += 1
                    except BaseException:
                        pass  # other errors

        except Exception:
            # last fallback: conservative estimation
            python_files = glob.glob(
                os.path.join(repo_path, "**", "*.py"), recursive=True
            )
            total_errors = max(1, len(python_files) // 4) 

        base_score = max(0.0, min(1.0, 1.0 - total_errors / 50.0))

        # bump: tests folder exists (tests/ or test/) → +0.1
        has_tests = os.path.exists(os.path.join(repo_path, "tests")) or os.path.exists(
            os.path.join(repo_path, "test")
        )
        if has_tests:
            base_score += 0.1

        # bump: CI config present (.github/workflows/*.yml or ci/*.yml) → +0.1
        has_ci = (
            os.path.exists(os.path.join(repo_path, ".github", "workflows"))
            or os.path.exists(os.path.join(repo_path, "ci"))
            or any(
                os.path.exists(os.path.join(repo_path, ci_file))
                for ci_file in [".travis.yml", ".circleci", "azure-pipelines.yml"]
            )
        )
        if has_ci:
            base_score += 0.1

        # cap at 1.0
        return min(1.0, base_score)

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
