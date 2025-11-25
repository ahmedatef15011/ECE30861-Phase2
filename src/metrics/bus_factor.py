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
        """Calculate bus factor from unique commit authors.

        Formula: ``min(1.0, contributors / 5.0)``.
        """
        contributors = 0

        # get contributor count from git repository analysis
        if context.code_repos:
            git_inspector = GitInspector()
            try:
                # limit to first 2 repositories to prevent excessive processing
                for code_repo in context.code_repos[:2]:
                    repo_path = git_inspector.clone_repo(code_repo)
                    if repo_path:
                        analysis = git_inspector.analyze_repository(repo_path)
                        contributor_data = analysis.get("contributor_analysis", {})

                        # get unique authors (prefer last 12 months, otherwise all-time)
                        recent_authors = contributor_data.get(
                            "recent_unique_authors", 0
                        )
                        all_time_authors = contributor_data.get("unique_authors", 0)

                        contributors = (
                            recent_authors if recent_authors > 0 else all_time_authors
                        )
                        break  # use first available repo
            finally:
                git_inspector.cleanup()

        # otherwise, try to estimate from HF metadata
        if contributors == 0 and context.hf_info:
            # estimate contributors based on HF engagement as fallback
            downloads = context.hf_info.get("downloads", 0)
            likes = context.hf_info.get("likes", 0)

            # rough estimation: high engagement suggests more contributors
            if downloads > 100000 and likes > 100:
                contributors = 5 # assumption
            elif downloads > 10000 and likes > 50:
                contributors = 3
            elif downloads > 1000 and likes > 10:
                contributors = 2
            else:
                contributors = 1

        # specification formula: BusFactor = min(1.0, contributors / 5.0)
        return min(1.0, contributors / 5.0)

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
