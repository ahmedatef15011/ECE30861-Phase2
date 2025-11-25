from typing import Any, Dict, Optional

from ..hf_api import HuggingFaceAPI
from ..models import MetricResult, ModelContext
from ..utils import measure_time
from .base import BaseMetric


class DatasetQualityMetric(BaseMetric):
    """Metric for evaluating quality of linked datasets."""

    @property
    def name(self) -> str:
        return "dataset_quality"

    def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        """Compute dataset quality score."""
        with measure_time() as get_latency:
            score = self._calculate_dataset_quality_score(context, config)

        return MetricResult(score=score, latency=get_latency())

    def _calculate_dataset_quality_score(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> float:
        """Calculate dataset quality.

        Score = (#fields_filled / 4) for description, size/#samples, license,
        and benchmark references.
        """
        #iIf no datasets are linked, check README for dataset information
        if not context.datasets:
            if context.readme_content:
                return self._analyze_readme_dataset_quality(context.readme_content)
            else:
                return 0.3  # default when no datasets

        total_score = 0.0
        datasets_analyzed = 0

        hf_api = HuggingFaceAPI()

        for dataset_url in context.datasets:
            if dataset_url.platform == "huggingface":
                dataset_score = self._analyze_hf_dataset_quality(
                    dataset_url, hf_api
                )
                total_score += dataset_score
                datasets_analyzed += 1

        if datasets_analyzed == 0:
            # non HF datasets - check README fallback
            if context.readme_content:
                return self._analyze_readme_dataset_quality(context.readme_content)
            else:
                return 0.3  # default

        return total_score / datasets_analyzed

    def _analyze_hf_dataset_quality(
        self, dataset_url, hf_api: HuggingFaceAPI
    ) -> float:
        """Analyze a HF dataset for description, size/#samples, license, benchmarks."""
        # get dataset README
        readme_content = hf_api.get_readme_content(dataset_url)
        if not readme_content:
            return 0.0 

        # get dataset info from API
        dataset_info = hf_api.get_dataset_info(dataset_url)

        return self._analyze_dataset_content(readme_content, dataset_info)

    def _analyze_readme_dataset_quality(self, readme_content: str) -> float:
        """Analyze README content for dataset quality indicators."""
        return self._analyze_dataset_content(readme_content, None)

    def _analyze_dataset_content(
        self, readme_content: str, dataset_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Analyze content for 4 specific dataset quality fields."""
        score = 0.0
        readme_lower = readme_content.lower()

        # description (25% of points)
        if (
            "description" in readme_lower
            or "overview" in readme_lower
            or "dataset" in readme_lower
            or len(readme_content) > 300
        ):
            score += 0.25

        # size/#samples (25% of points)
        size_indicators = [
            "size",
            "samples",
            "examples",
            "instances",
            "records",
            "entries",
            "rows",
            "datapoints",
            "mb",
            "gb",
            "kb",
            "million",
            "thousand",
        ]
        if any(indicator in readme_lower for indicator in size_indicators):
            score += 0.25

        # license (25% of points)
        license_found = False
        if "license" in readme_lower:
            license_found = True
        elif dataset_info and dataset_info.get("tags"):
            license_found = any("license:" in tag for tag in dataset_info["tags"])

        if license_found:
            score += 0.25

        # benchmark references (25% of points)
        benchmark_indicators = [
            "benchmark",
            "evaluation",
            "baseline",
            "performance",
            "accuracy",
            "f1",
            "bleu",
            "rouge",
            "glue",
            "squad",
            "superglue",
            "results",
        ]
        if any(indicator in readme_lower for indicator in benchmark_indicators):
            score += 0.25

        return score
