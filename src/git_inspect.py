import os
import shutil
import sys
import tempfile
import threading
import atexit
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

from dulwich import porcelain
from dulwich.errors import NotGitRepository
from dulwich.repo import Repo

from .logging_utils import get_logger
from .models import ParsedURL

logger = get_logger()

# Global per-process cache directory (shared within this process only)
_PROCESS_CACHE_DIR = None

def _get_process_cache_dir():
    """Get or create the per-process cache directory"""
    global _PROCESS_CACHE_DIR
    if _PROCESS_CACHE_DIR is None:
        # Create a unique temp directory for this process
        _PROCESS_CACHE_DIR = tempfile.mkdtemp(prefix="ece30861_git_")
        logger.debug(f"Created per-process git cache: {_PROCESS_CACHE_DIR}")
        
        # Register cleanup on exit
        def cleanup_cache():
            if _PROCESS_CACHE_DIR and os.path.exists(_PROCESS_CACHE_DIR):
                try:
                    shutil.rmtree(_PROCESS_CACHE_DIR)
                    logger.debug(f"Cleaned up git cache: {_PROCESS_CACHE_DIR}")
                except:
                    pass
        atexit.register(cleanup_cache)
    
    return _PROCESS_CACHE_DIR


class GitInspector:
    # git repository inspector using Dulwich

    def __init__(self, cache_dir: Optional[str] = None, timeout: int = 30):
        # Use per-process cache (shared within run, cleaned between runs)
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            # Get the shared per-process cache directory
            self.cache_dir = _get_process_cache_dir()
            
        self.timeout = timeout
        os.makedirs(self.cache_dir, exist_ok=True)

    def clone_repo(self, repo_url: ParsedURL) -> Optional[str]:
        
        # clone a repository to cache directory with timeout handling

        if repo_url.platform != "github":
            return None

        repo_name = (
            f"{repo_url.owner}_{repo_url.repo}"
            if repo_url.owner and repo_url.repo
            else "unknown"
        )
        clone_path = os.path.join(self.cache_dir, repo_name)

        # check if already cloned
        if os.path.exists(clone_path):
            # Check if it's a valid repo (has .git)
            if os.path.exists(os.path.join(clone_path, ".git")):
                logger.debug(f"Using cached clone at {clone_path}")
                return clone_path
            else:
                # Invalid/partial clone, remove it
                shutil.rmtree(clone_path, ignore_errors=True)

        try:
            logger.info(f"Cloning {repo_url.url} to {clone_path}")
            
            # use threading to implement timeout
            result = [None]
            exception = [None]
            
            def clone_worker():
                try:
                    # suppress dulwich's verbose clone output
                    import io
                    old_stderr = sys.stderr
                    sys.stderr = io.StringIO()
                    
                    try:
                        # use smaller depth for faster cloning
                        porcelain.clone(
                            repo_url.url,
                            clone_path,
                            depth=5,
                            errstream=io.BytesIO()  # suppress error stream
                        )
                        result[0] = clone_path
                    finally:
                        # restore stderr
                        sys.stderr = old_stderr
                except Exception as e:
                    # restore stderr before logging exception
                    sys.stderr = old_stderr
                    exception[0] = e
            
            # start clone in separate thread
            thread = threading.Thread(target=clone_worker)
            thread.daemon = True
            thread.start()
            
            # wait for completion or timeout
            thread.join(timeout=self.timeout)
            
            if thread.is_alive():
                logger.warning(f"Timeout cloning {repo_url.url} after {self.timeout}s")
                # clean up partial clone
                if os.path.exists(clone_path):
                    shutil.rmtree(clone_path, ignore_errors=True)
                return None
            
            if exception[0]:
                raise exception[0]
                
            return result[0]
                
        except Exception as e:
            logger.warning(f"Failed to clone {repo_url.url}: {e}")
            # clean up partial clone
            if os.path.exists(clone_path):
                shutil.rmtree(clone_path, ignore_errors=True)
            return None

    def analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        # analyze a cloned repository for various quality metrics

        try:
            repo = Repo(repo_path)
        except NotGitRepository:
            logger.error(f"Not a git repository: {repo_path}")
            return self._empty_analysis()

        analysis = {
            "commit_analysis": self._analyze_commits(repo),
            "contributor_analysis": self._analyze_contributors(repo),
            "file_analysis": self._analyze_files(repo_path),
            "structure_analysis": self._analyze_structure(repo_path),
            "documentation_analysis": self._analyze_documentation(repo_path),
        }

        return analysis

    def _analyze_commits(self, repo: Repo) -> Dict[str, Any]:
        # analyze commit history
        try:
            commits = list(repo.get_walker(max_entries=100))  # limit for performance

            if not commits:
                return {
                    "total_commits": 0,
                    "recent_commits": 0,
                    "avg_commit_frequency": 0,
                }

            # get recent commits
            now = datetime.now(timezone.utc)
            recent_threshold = now.timestamp() - (90 * 24 * 60 * 60)  # 90 days ago

            recent_commits = []
            commit_dates = []

            for entry in commits:
                commit = entry.commit
                commit_time = commit.commit_time
                commit_dates.append(commit_time)

                if commit_time >= recent_threshold:
                    recent_commits.append(commit)

            # calculate average commit frequency
            if len(commit_dates) > 1:
                time_span = max(commit_dates) - min(commit_dates)
                days_span = max(1, time_span / (24 * 60 * 60))
                avg_frequency = len(commit_dates) / days_span
            else:
                avg_frequency = 0

            return {
                "total_commits": len(commits),
                "recent_commits": len(recent_commits),
                "avg_commit_frequency": avg_frequency,
                "time_span_days": (
                    (max(commit_dates) - min(commit_dates)) / (24 * 60 * 60)
                    if len(commit_dates) > 1
                    else 0
                ),
            }

        except Exception as e:
            #keeping the rest of the analysis alive even if this part fails
            logger.warning(f"Error analyzing commits: {e}")
            return {"total_commits": 0, "recent_commits": 0, "avg_commit_frequency": 0}

    def _analyze_contributors(self, repo: Repo) -> Dict[str, Any]:
        # analyze contributor diversity
        try:
            commits = list(repo.get_walker(max_entries=100))

            authors: Set[str] = set()
            committers: Set[str] = set()

            for entry in commits:
                commit = entry.commit
                #author/commiteer are bytes
                authors.add(commit.author.decode("utf-8", errors="ignore"))
                committers.add(commit.committer.decode("utf-8", errors="ignore"))

            # simple bus factor calculation
            total_commits = len(commits)
            unique_authors = len(authors)

            if total_commits == 0:
                bus_factor_score = 0.0
            elif unique_authors == 1:
                bus_factor_score = 0.3  # low diversity
            elif unique_authors < 3:
                bus_factor_score = 0.5  # medium diversity
            else:
                bus_factor_score = 0.8  # high diversity

            return {
                "unique_authors": unique_authors,
                "unique_committers": len(committers),
                "bus_factor_score": bus_factor_score,
                "authors": list(authors)[:10],  # limit for privacy
            }

        except Exception as e:
            logger.warning(f"Error analyzing contributors: {e}")
            return {
                "unique_authors": 0,
                "unique_committers": 0,
                "bus_factor_score": 0.0,
            }

    def _analyze_files(self, repo_path: str) -> Dict[str, Any]:
        # analyze file structure and types
        try:
            repo_root = Path(repo_path)

            #consider any 8.py as code, tests are identified by name or parent folder

            python_files = list(repo_root.glob("**/*.py"))
            test_files = [
                f
                for f in python_files
                if "test" in f.name.lower()
                or f.parent.name.lower() in ["test", "tests"]
            ]

            # count lines of code
            total_lines = 0
            for py_file in python_files:
                try:
                    with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                        total_lines += len(f.readlines())
                except Exception:
                    #ignore unreadable files 
                    continue

            return {
                "total_files": len(list(repo_root.glob("**/*"))),
                "python_files": len(python_files),
                "test_files": len(test_files),
                "total_lines_of_code": total_lines,
                "test_coverage_estimate": (
                    len(test_files) / max(1, len(python_files)) if python_files else 0
                ),
            }

        except Exception as e:
            logger.warning(f"Error analyzing files: {e}")
            return {
                "total_files": 0,
                "python_files": 0,
                "test_files": 0,
                "total_lines_of_code": 0,
            }

    def _analyze_structure(self, repo_path: str) -> Dict[str, Any]:
        # analyze repository structure quality
        try:
            repo_root = Path(repo_path)

            # check for standard files
            has_readme = any(repo_root.glob("README.*"))
            has_license = any(repo_root.glob("LICENSE*")) or any(
                repo_root.glob("LICENCE*")
            )
            has_requirements = (repo_root / "requirements.txt").exists() or (
                repo_root / "pyproject.toml"
            ).exists()
            has_setup = (repo_root / "setup.py").exists() or (
                repo_root / "setup.cfg"
            ).exists()
            has_ci = (repo_root / ".github").exists() or (
                repo_root / ".gitlab-ci.yml"
            ).exists()

#simple equal-weight score, eadsy to extend with weights if needed
            structure_score = (
                sum([has_readme, has_license, has_requirements, has_setup, has_ci])
                / 5.0
            )

            return {
                "has_readme": has_readme,
                "has_license": has_license,
                "has_requirements": has_requirements,
                "has_setup": has_setup,
                "has_ci": has_ci,
                "structure_score": structure_score,
            }

        except Exception as e:
            logger.warning(f"Error analyzing structure: {e}")
            return {"structure_score": 0.0}

    def _analyze_documentation(self, repo_path: str) -> Dict[str, Any]:
        # analyze documentation quality
        try:
            repo_root = Path(repo_path)

            # find README content
            readme_content = ""
            for readme_file in repo_root.glob("README.*"):
                try:
                    with open(readme_file, "r", encoding="utf-8", errors="ignore") as f:
                        readme_content = f.read()
                    break
                except Exception:
                    continue

            # basic documentation analysis
            has_usage = "usage" in readme_content.lower()
            has_installation = "install" in readme_content.lower()
            has_examples = "example" in readme_content.lower()

            doc_score = (
                sum([bool(readme_content), has_usage, has_installation, has_examples])
                / 4.0
            )

            return {
                "readme_length": len(readme_content),
                "has_usage_section": has_usage,
                "has_installation_section": has_installation,
                "has_examples": has_examples,
                "documentation_score": doc_score,
                "readme_content": readme_content[:2000],  # Limit size to keep small
            }

        except Exception as e:
            logger.warning(f"Error analyzing documentation: {e}")
            return {"documentation_score": 0.0, "readme_content": ""}

    def _empty_analysis(self) -> Dict[str, Any]:
        # return empty analysis structure
        return {
            "commit_analysis": {
                "total_commits": 0,
                "recent_commits": 0,
                "avg_commit_frequency": 0,
            },
            "contributor_analysis": {"unique_authors": 0, "bus_factor_score": 0.0},
            "file_analysis": {"total_files": 0, "python_files": 0, "test_files": 0},
            "structure_analysis": {"structure_score": 0.0},
            "documentation_analysis": {
                "documentation_score": 0.0,
                "readme_content": "",
            },
        }

    def cleanup(self):
        # clean up cache directory
        # Delete the cache directory (useful for tests)
        if os.path.exists(self.cache_dir):
            try:
                shutil.rmtree(self.cache_dir)
                logger.debug(f"Cleaned up cache directory: {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup cache directory: {e}")

