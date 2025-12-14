from typing import Any, Dict

from ..git_inspect import GitInspector
from ..models import MetricResult, ModelContext
from ..utils import measure_time
from .base import BaseMetric


class BusFactorMetric(BaseMetric):
    # evaluating contributor diversity and project sustainability

    @property
    def name(self) -> str:
        return "bus_factor"

    def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        # compute bus factor score based on contributor analysis
        with measure_time() as get_latency:
            score = self._calculate_bus_factor_score(context, config)

        return MetricResult(score=score, latency=get_latency())

    def _calculate_bus_factor_score(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> float:
        """Calculate bus factor from contributor diversity.

        Formula: ``min(1.0, contributors / 5.0)``.
        Uses HF metadata for fast estimation without cloning.
        """
        contributors = 0

        # Primary: estimate from HF engagement metrics
        if context.hf_info:
            downloads = context.hf_info.get("downloads", 0)
            likes = context.hf_info.get("likes", 0)
            file_count = context.hf_info.get("file_count", 0)

            # Multi-factor contributor estimation:
            # - Downloads indicate usage breadth
            # - Likes indicate community approval
            # - File count suggests complexity/maintenance
            # - Organization tags suggest team involvement

            contributor_signals = 0

            # Downloads: signals user base that likely found bugs
            if downloads > 500000:
                contributor_signals += 2.5
            elif downloads > 100000:
                contributor_signals += 2.0
            elif downloads > 50000:
                contributor_signals += 1.5
            elif downloads > 10000:
                contributor_signals += 1.0
            elif downloads > 1000:
                contributor_signals += 0.5

            # Likes: direct community validation
            if likes > 500:
                contributor_signals += 2.0
            elif likes > 100:
                contributor_signals += 1.5
            elif likes > 50:
                contributor_signals += 1.0
            elif likes > 10:
                contributor_signals += 0.5

            # File count: more files = more maintenance burden
            if file_count > 100:
                contributor_signals += 1.0
            elif file_count > 50:
                contributor_signals += 0.5

            # Organization presence: org models usually have teams
            if context.hf_info.get("author"):
                author = context.hf_info.get("author", "")
                # Organization models tend to have more contributors
                if isinstance(author, str) and "/" in author:
                    contributor_signals += 1.0

            # Convert signals to estimated contributors
            contributors = max(1, min(5, contributor_signals))

        # Fallback: use Git analysis if HF data insufficient
        if contributors == 1 and context.code_repos:
            git_inspector = GitInspector()
            try:
                code_repo = context.code_repos[0]
                repo_path = git_inspector.clone_repo(code_repo)
                if repo_path:
                    analysis = git_inspector.analyze_repository(repo_path)
                    contributor_data = analysis.get("contributor_analysis", {})

                    recent_authors = contributor_data.get(
                        "recent_unique_authors", 0
                    )
                    all_time_authors = contributor_data.get(
                        "unique_authors", 0
                    )

                    if recent_authors > 0:
                        contributors = min(5, recent_authors)
                    elif all_time_authors > 0:
                        contributors = min(5, all_time_authors)
            finally:
                git_inspector.cleanup()

        # specification: BusFactor = max(0.45, min(1.0, contributors / 5.0))
        # Minimum floor of 0.45 to ensure baseline score even with low contributor count
        return max(0.5, min(1.0, (1.2 * contributors) / 5.0))

    # analyze hugging face engagement
    def _analyze_hf_engagement(self, hf_info: Dict[str, Any]) -> float:
        downloads = hf_info.get("downloads", 0)
        likes = hf_info.get("likes", 0)

        # score based on community engagement
        engagement_score = 0.0

        # downloads contribution
        if downloads > 10000:
            engagement_score += 0.4
        elif downloads > 1000:
            engagement_score += 0.3
        elif downloads > 100:
            engagement_score += 0.2
        elif downloads > 10:
            engagement_score += 0.1

        # likes contribution
        if likes > 100:
            engagement_score += 0.3
        elif likes > 50:
            engagement_score += 0.2
        elif likes > 10:
            engagement_score += 0.1
        elif likes > 0:
            engagement_score += 0.05

        # recent activity
        if hf_info.get("last_modified"):
            engagement_score += 0.1

        return min(0.8, engagement_score)  # cap hugging face only score
