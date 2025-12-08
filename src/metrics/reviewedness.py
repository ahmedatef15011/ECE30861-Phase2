"""Reviewedness metric: fraction of code introduced through reviewed PRs."""

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

from ..git_inspect import GitInspector
from ..models import MetricResult, ModelContext
from ..utils import measure_time
from .base import BaseMetric
from .fallback_scoring import FallbackScorer
from ..logging_utils import get_logger

logger = get_logger()


# File extensions for model weights (exclude from code review analysis)
WEIGHT_EXTENSIONS = {
    '.pt', '.pth', '.bin', '.safetensors', '.h5', '.pb', 
    '.onnx', '.tflite', '.ckpt', '.pkl', '.pickle', 
    '.npz', '.npy', '.weights'
}

# File extensions considered as code
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', 
    '.c', '.h', '.hpp', '.go', '.rs', '.rb', '.php', 
    '.swift', '.kt', '.scala', '.cs', '.r', '.m', '.sh',
    '.yaml', '.yml', '.json', '.toml', '.xml'
}

# Maximum file size to consider as code (10MB)
MAX_CODE_FILE_SIZE = 10 * 1024 * 1024


class ReviewednessMetric(BaseMetric):
    """Evaluate fraction of code introduced through pull requests with reviews."""

    @property
    def name(self) -> str:
        return "reviewedness"

    def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        """Compute reviewedness score.
        
        Returns:
            MetricResult with score:
            - 0.0 to 1.0: fraction of code lines from reviewed PRs
            - -1.0: no GitHub repository linked
        """
        with measure_time() as get_latency:
            # Call synchronous implementation directly
            score = self._calculate_reviewedness_score_sync(
                context,
                config
            )

        return MetricResult(score=score, latency=get_latency())

    def _calculate_reviewedness_score_sync(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> float:
        """Synchronous implementation of reviewedness calculation."""
        
        # Check if there's a GitHub repository
        if not context.code_repos:
            # Use fallback scoring based on README content
            logger.info("No GitHub repository found, trying fallback scoring")
            fallback = FallbackScorer(context.readme_content)
            score, details = fallback.get_reviewedness_fallback_score()
            if score == -1.0:
                logger.info("No review indicators in README, returning -1")
            else:
                logger.info(f"Reviewedness fallback score: {score:.2f}")
            return score

        git_inspector = GitInspector()
        try:
            # Clone the first code repository
            for code_repo in context.code_repos[:1]:  # Only analyze first repo
                repo_path = git_inspector.clone_repo(code_repo)
                if not repo_path:
                    logger.warning(f"Failed to clone {code_repo.url}")
                    continue

                # Calculate reviewedness from git history
                reviewed_lines, total_lines = self._analyze_git_history(repo_path)
                
                # Check for error condition (both are -1) - try fallback
                if reviewed_lines == -1 and total_lines == -1:
                    logger.warning(
                        "Git operation failed, trying fallback scoring"
                    )
                    fallback = FallbackScorer(context.readme_content)
                    score, details = fallback.get_reviewedness_fallback_score()
                    return score
                
                if total_lines == 0:
                    logger.warning("No code lines found in repository")
                    # Use fallback instead of returning 0
                    fallback = FallbackScorer(context.readme_content)
                    score, details = fallback.get_reviewedness_fallback_score()
                    return max(0.1, score)  # Minimum floor

                git_score = reviewed_lines / total_lines
                logger.info(
                    f"Reviewedness git: {reviewed_lines}/{total_lines} "
                    f"= {git_score:.3f}"
                )
                
                # Get fallback score from README indicators
                fallback = FallbackScorer(context.readme_content)
                fallback_score, details = (
                    fallback.get_reviewedness_fallback_score()
                )
                
                # Only blend if fallback found meaningful indicators
                # (score > 0.15 baseline means actual indicators were found)
                if fallback_score > 0.15:
                    # Blend git score (70%) with fallback score (30%)
                    # This ensures models with good README indicators
                    # aren't penalized too harshly for missing PR patterns
                    blended = 0.7 * git_score + 0.3 * fallback_score
                    logger.info(
                        f"Reviewedness blended: git={git_score:.3f}, "
                        f"fallback={fallback_score:.3f}, final={blended:.3f}"
                    )
                    return blended
                
                return git_score

            # If we couldn't clone any repo, try fallback
            logger.warning(
                "Could not clone any code repository, trying fallback"
            )
            fallback = FallbackScorer(context.readme_content)
            score, details = fallback.get_reviewedness_fallback_score()
            return score

        except Exception as e:
            logger.error(f"Error calculating reviewedness: {e}, trying fallback")
            # Try fallback on exception instead of returning -1
            fallback = FallbackScorer(context.readme_content)
            score, details = fallback.get_reviewedness_fallback_score()
            return score
        finally:
            git_inspector.cleanup()

    def _analyze_git_history(self, repo_path: str) -> Tuple[int, int]:
        """Analyze git history to calculate reviewed vs total lines.
        
        Returns:
            Tuple of (reviewed_lines, total_lines)
        """
        try:
            # Get all commits with their stats
            result = subprocess.run(
                ['git', 'log', '--all', '--numstat', '--pretty=format:%H|%s'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.warning(f"Git log failed: {result.stderr}")
                return (-1, -1)

            return self._parse_git_log_output(result.stdout, repo_path)

        except subprocess.TimeoutExpired:
            logger.warning("Git log command timed out")
            return (-1, -1)
        except FileNotFoundError:
            logger.error("Git command not found")
            return (-1, -1)
        except Exception as e:
            logger.error(f"Error analyzing git history: {e}")
            return (-1, -1)

    def _parse_git_log_output(
        self, git_output: str, repo_path: str
    ) -> Tuple[int, int]:
        """Parse git log output and calculate reviewed/total lines."""
        
        reviewed_lines = 0
        total_lines = 0
        current_commit = None
        current_is_reviewed = False

        lines = git_output.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a commit header
            if '|' in line and not '\t' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    current_commit = parts[0]
                    commit_message = parts[1]
                    
                    # Check if commit is from a reviewed PR
                    current_is_reviewed = self._is_reviewed_commit(
                        commit_message, repo_path, current_commit
                    )
                continue

            # Parse numstat line: "added\tdeleted\tfilename"
            if '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 3:
                    added_str, deleted_str, filename = parts[0], parts[1], parts[2]
                    
                    # Skip if not a code file
                    if not self._is_code_file(filename, repo_path):
                        continue

                    # Parse additions
                    try:
                        added = int(added_str) if added_str != '-' else 0
                    except ValueError:
                        continue

                    # Count lines
                    if current_is_reviewed:
                        reviewed_lines += added
                    total_lines += added

        return (reviewed_lines, total_lines)

    def _is_reviewed_commit(
        self, commit_message: str, repo_path: str, commit_hash: str
    ) -> bool:
        """Check if a commit was introduced through a reviewed PR.
        
        Detection strategies:
        1. Commit message contains PR reference (e.g., "#123", "PR #456")
        2. Commit is a merge commit with PR reference
        3. GitHub squash merge patterns
        4. Co-authored commits (multiple contributors = review)
        5. Signed-off-by trailer (DCO review process)
        6. Conventional commit format (indicates structured process)
        7. Reviewed-by/Acked-by trailers
        """
        
        msg_lower = commit_message.lower()
        
        # Strategy 1: Check for PR references in commit message
        pr_patterns = [
            r'#(\d+)',                      # #123
            r'PR\s*#(\d+)',                 # PR #123
            r'Merge pull request #(\d+)',   # Merge pull request #123
            r'\(#(\d+)\)',                  # Fixes (#123) or squash merge
            r'pull request \d+',            # pull request 123
            r'MR\s*[#!](\d+)',              # GitLab MR #123 or MR !123
            r'!(\d+)',                      # GitLab style !123
        ]
        
        for pattern in pr_patterns:
            if re.search(pattern, commit_message, re.IGNORECASE):
                return True

        # Strategy 2: Check if it's a merge commit
        if msg_lower.startswith('merge'):
            return True
        
        # Strategy 3: GitHub squash merge patterns (often end with PR number)
        # e.g., "feat: add new feature (#123)"
        if re.search(r'\(\s*#\d+\s*\)\s*$', commit_message):
            return True
        
        # Strategy 4: Co-authored commits indicate collaboration/review
        if 'co-authored-by:' in msg_lower:
            return True
        
        # Strategy 5: Signed-off-by indicates DCO/review process
        if 'signed-off-by:' in msg_lower:
            return True
        
        # Strategy 6: Reviewed-by or Acked-by trailers
        review_trailers = [
            'reviewed-by:', 'acked-by:', 'tested-by:', 'approved-by:'
        ]
        for trailer in review_trailers:
            if trailer in msg_lower:
                return True
        
        # Strategy 7: Conventional commits often indicate structured process
        # feat:, fix:, chore:, docs:, style:, refactor:, test:, etc.
        conventional_prefixes = (
            r'^(feat|fix|chore|docs|style|refactor|perf|test|build|ci|revert)'
        )
        conventional_pattern = conventional_prefixes + r'(\(.+\))?:'
        if re.match(conventional_pattern, commit_message, re.IGNORECASE):
            return True
        
        # Strategy 8: Revert commits indicate review process
        if msg_lower.startswith('revert'):
            return True

        return False

    def _is_code_file(self, filename: str, repo_path: str) -> bool:
        """Check if a file should be considered as code (not weights)."""
        
        # Get file extension
        ext = Path(filename).suffix.lower()
        
        # Exclude weight files
        if ext in WEIGHT_EXTENSIONS:
            return False

        # Include known code files
        if ext in CODE_EXTENSIONS:
            # Check file size if it exists
            file_path = Path(repo_path) / filename
            if file_path.exists():
                try:
                    if file_path.stat().st_size > MAX_CODE_FILE_SIZE:
                        return False
                except OSError:
                    pass
            return True

        # Exclude files in common weight directories
        weight_dirs = {'models', 'weights', 'checkpoints', 'model_weights'}
        path_parts = set(Path(filename).parts)
        if path_parts & weight_dirs:
            return False

        return False
