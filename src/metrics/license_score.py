from typing import Any, Dict

from ..models import MetricResult, ModelContext
from ..utils import measure_time, parse_license_from_readme
from .base import BaseMetric


class LicenseScoreMetric(BaseMetric):
    """Metric for evaluating license compatibility and clarity."""

    @property
    def name(self) -> str:
        return "license"

    def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        """Compute license score."""
        with measure_time() as get_latency:
            score = self._calculate_license_score(context, config)

        return MetricResult(score=score, latency=get_latency())

    def _calculate_license_score(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> float:
        """Calculate license score based on exact specification mapping."""
        # get license from HF info first
        license_info = None
        if context.hf_info and context.hf_info.get("tags"):
            # license in tags
            for tag in context.hf_info["tags"]:
                if tag.startswith("license:"):
                    license_info = tag.replace("license:", "")
                    break

        # no HF license
        if not license_info and context.readme_content:
            license_info = parse_license_from_readme(context.readme_content)

        # no license present
        if not license_info:
            return 0.3 * 0.8  # Scale down by 0.8

        license_lower = (
            license_info.lower().replace("-", "").replace("_", "").replace(" ", "")
        )

        # MIT/Apache 2.0/BSD → 1.0
        compatible_licenses = [
            "mit",
            "apache2.0",
            "apache20",
            "apache",
            "bsd3clause",
            "bsd2clause",
            "bsd",
            "bsd3",
            "bsd2",
            "bsdnew",
            "bsdmodified",
            "bsdsimplified",
        ]
        for compat in compatible_licenses:
            if compat in license_lower:
                return 1.0 * 0.8  # Scale down by 0.8

        # LGPL (any variant) 
        lgpl_variants = ["lgpl", "lessergpl", "lgplv2", "lgplv3", "lgpl2.1", "lgpl3.0"]
        for lgpl in lgpl_variants:
            if lgpl in license_lower:
                return 0.8 * 0.8  # Scale down by 0.8

        # GPL (any variant)
        gpl_variants = ["gpl", "gplv2", "gplv3", "gpl2.0", "gpl3.0", "agpl", "agplv3"]
        for gpl in gpl_variants:
            if gpl in license_lower:
                return 0.7 * 0.8  # Scale down by 0.8

        # unknown license format
        return 0.5 * 0.8  # Scale down by 0.8
