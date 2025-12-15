from typing import Any, Dict

from ..models import MetricResult, ModelContext
from ..utils import measure_time
from .base import BaseMetric


class PerformanceClaimsMetric(BaseMetric):
    """Metric for evaluating documented performance claims and benchmarks."""

    @property
    def name(self) -> str:
        return "performance_claims"

    def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        """Compute performance claims score."""
        with measure_time() as get_latency:
            score = self._calculate_performance_score(context, config)

        return MetricResult(score=score, latency=get_latency())

    def _calculate_performance_score(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> float:
        """Calculate performance claims score.

        Heuristic: reproducible steps → 1.0, vague claims → 0.5, none → 0.0.
        """
        if not context.readme_content:
            return 0.0

        readme_lower = context.readme_content.lower()

        # check for explicit negative statements
        negative_indicators = [
            "no performance",
            "no benchmark",
            "no evaluation",
            "no results",
        ]
        has_negative = any(neg in readme_lower for neg in negative_indicators)

        if has_negative:
            return 0.05  # no performance data

        # specific benchmark scores
        benchmark_patterns = [
            "glue",
            "squad",
            "mmlu",
            "accuracy",
            "f1",
            "bleu",
            "rouge",
            "score",
            "benchmark",
            "%",
            "percent",
        ]
        has_benchmarks = any(pattern in readme_lower for pattern in benchmark_patterns)

        # paper references (arxiv, papers, citations)
        paper_indicators = [
            "arxiv",
            "paper",
            "citation",
            "https://arxiv.org",
            "methodology",
            "experiment",
            "evaluation protocol",
            "see paper",
        ]
        has_paper_ref = any(indicator in readme_lower for indicator in paper_indicators)

        # reproducible elements (code, scripts, notebooks)
        reproducible_indicators = [
            "reproduce",
            "reproducible",
            "script",
            "code",
            "github.com/",
            "colab",
            "notebook",
            "eval.py",
            "evaluation.py",
            "steps to reproduce",
        ]
        has_reproducible = any(
            indicator in readme_lower for indicator in reproducible_indicators
        )

        # multiple benchmarks or detailed results
        detailed_indicators = ["|", "table", "multiple", "various", "several"]
        has_detailed_results = any(
            indicator in readme_lower for indicator in detailed_indicators
        )

        # scoring logic
        # multiple benchmarks + citations - 0.7-1.0
        if has_benchmarks and has_paper_ref and has_detailed_results:
            return 0.85 * 0.8  # Scale down by 0.8

        # benchmark + paper reference - 0.6-1.0
        elif has_benchmarks and has_paper_ref:
            return 0.75 * 0.8  # Scale down by 0.8

        # reproducible steps + benchmarks - 1.0
        elif has_reproducible and has_benchmarks:
            return 1.0 * 0.8  # Scale down by 0.8

        # just benchmark scores - 0.3-0.6 (vague claims)
        elif has_benchmarks:
            return 0.5 * 0.8  # Scale down by 0.8

        # no performance data - 0.0-0.2
        else:
            return 0.1 * 0.8  # Scale down by 0.8
