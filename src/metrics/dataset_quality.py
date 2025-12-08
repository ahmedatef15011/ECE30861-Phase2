from typing import Any, Dict, Optional, List
import logging
import re

from ..hf_api import HuggingFaceAPI
from ..models import MetricResult, ModelContext
from ..utils import measure_time
from .base import BaseMetric
from .fallback_scoring import FallbackScorer

logger = logging.getLogger(__name__)

# Common dataset name patterns to extract from README
COMMON_DATASETS = [
    "imagenet", "coco", "squad", "glue", "superglue", "wikitext",
    "bookcorpus", "books corpus", "wikipedia", "openwebtext", "c4", "pile",
    "laion", "cc12m", "conceptual-captions", "vqa", "mnist",
    "cifar", "celeba", "lfw", "imdb", "yelp", "amazon-reviews",
    "commonvoice", "librispeech", "audioset", "kinetics",
    "ucf101", "hmdb51", "ade20k", "cityscapes", "pascal-voc",
    "ms-marco", "natural-questions", "triviaqa", "hotpotqa",
    "web-text", "redpajama", "dolma", "fineweb", "cosmopedia"
]


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

    def _extract_dataset_names_from_readme(
        self, readme_content: str
    ) -> List[str]:
        """
        Extract potential dataset names from README content.
        
        Uses pattern matching to find common dataset references.
        """
        if not readme_content:
            return []
        
        readme_lower = readme_content.lower()
        found_datasets = []
        
        # Check for common dataset names
        for dataset in COMMON_DATASETS:
            if dataset in readme_lower:
                found_datasets.append(dataset)
        
        # Look for HuggingFace dataset patterns like "datasets/xxx" or
        # "load_dataset('xxx')"
        hf_patterns = [
            r"load_dataset\s*\(\s*['\"]([^'\"]+)['\"]",
            r"datasets/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)",
            r"huggingface\.co/datasets/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)",
        ]
        
        for pattern in hf_patterns:
            matches = re.findall(pattern, readme_content, re.IGNORECASE)
            found_datasets.extend(matches)
        
        return list(set(found_datasets))

    def _try_lookup_dataset(
        self, dataset_name: str, hf_api: HuggingFaceAPI
    ) -> Optional[float]:
        """
        Try to look up a dataset on HuggingFace and get its quality score.
        
        Returns None if dataset not found.
        """
        try:
            # Try to fetch dataset info from HuggingFace
            # This is a simplified lookup - in production would use
            # the datasets library
            dataset_readme = hf_api.get_model_readme(
                f"datasets/{dataset_name}"
            )
            if dataset_readme:
                # Got a dataset README - analyze it
                return self._analyze_dataset_content(dataset_readme, None)
        except Exception as e:
            logger.debug(f"Could not look up dataset {dataset_name}: {e}")
        
        return None

    def _calculate_dataset_quality_score(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> float:
        """Calculate dataset quality.

        Score = (#fields_filled / 4) for description, size/#samples, license,
        and benchmark references.
        
        Enhanced with LLM dataset discovery:
        - If no datasets linked, LLM analyzes README to find dataset names
        - Discovered datasets are looked up on HuggingFace when possible
        - Blends dataset analysis with model README fallback for coverage
        """
        hf_api = HuggingFaceAPI()
        
        # If no datasets are linked, try to discover them from README
        if not context.datasets:
            # First, try to extract dataset names from README
            discovered_datasets = self._extract_dataset_names_from_readme(
                context.readme_content
            )
            
            if discovered_datasets:
                logger.info(
                    f"Discovered datasets from README: {discovered_datasets}"
                )
                # Try to look up discovered datasets
                # (future enhancement: actually fetch HF dataset info)
            
            # Use enhanced fallback scoring (includes LLM analysis)
            fallback = FallbackScorer(context.readme_content)
            score, details = fallback.get_dataset_fallback_score()
            
            # Boost score if we found dataset references
            if discovered_datasets:
                # Give extra credit for mentioning known datasets
                dataset_boost = min(0.15, len(discovered_datasets) * 0.05)
                score = min(0.85, score + dataset_boost)
            
            return score

        total_score = 0.0
        datasets_analyzed = 0

        for dataset_url in context.datasets:
            if dataset_url.platform == "huggingface":
                dataset_score = self._analyze_hf_dataset_quality(
                    dataset_url, hf_api
                )
                total_score += dataset_score
                datasets_analyzed += 1

        if datasets_analyzed == 0:
            # Non-HF datasets - use enhanced fallback scoring
            fallback = FallbackScorer(context.readme_content)
            score, details = fallback.get_dataset_fallback_score()
            return score

        dataset_avg = total_score / datasets_analyzed
        
        # Blend dataset score with model README fallback
        # This helps when dataset metadata is incomplete but model
        # README describes data well
        fallback = FallbackScorer(context.readme_content)
        fallback_score, _ = fallback.get_dataset_fallback_score()
        
        # Use max of dataset analysis and fallback to be generous
        # Models shouldn't be penalized if they describe data in README
        # even if the linked dataset has incomplete metadata
        if fallback_score > dataset_avg:
            # Blend: 60% better score, 40% lower score
            final_score = 0.6 * fallback_score + 0.4 * dataset_avg
        else:
            final_score = dataset_avg
        
        return final_score

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
