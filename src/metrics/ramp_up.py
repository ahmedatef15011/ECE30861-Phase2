from typing import Any, Dict

from ..models import MetricResult, ModelContext
from ..utils import measure_time
from .base import BaseMetric


class RampUpTimeMetric(BaseMetric):
    """Metric for evaluating ease of getting started with the model."""

    @property
    def name(self) -> str:
        return "ramp_up_time"

    def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        """Compute ramp-up time score based on documentation quality."""
        with measure_time() as get_latency:
            score = self._calculate_ramp_up_score(context, config)

        # Scale down by 0.8 to make ramp-up time less dominant
        scaled_score = min(1.0, score * 0.8)
        
        # Map to [0.6, 0.85] range: ensure all models get decent ramp-up credit
        # Formula: 0.6 + (scaled_score / 0.8) * 0.25
        final_score = 0.6 + (scaled_score / 0.8) * 0.25
        
        return MetricResult(score=final_score, latency=get_latency())

    def _calculate_ramp_up_score(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> float:
        """Calculate ramp-up score based on 4 specific criteria."""
        score = 0.0
        criteria_count = 4

        # 1 - README present
        if context.readme_content:
            score += 1.0 / criteria_count
            readme_lower = context.readme_content.lower()
        else:
            return 0.1

        # 2 - install instructions
        install_indicators = [
            "install",
            "pip install",
            "conda install",
            "npm install",
            "yarn install",
            "setup",
            "installation",
            "getting started",
            "requirements",
            "dependencies",
        ]
        if any(indicator in readme_lower for indicator in install_indicators):
            score += 1.0 / criteria_count

        # 3 - training/evaluation examples
        training_indicators = [
            "training",
            "train",
            "fine-tuning",
            "fine tuning",
            "finetune",
            "evaluation",
            "eval",
            "benchmark",
            "test",
            "validate",
        ]
        if any(indicator in readme_lower for indicator in training_indicators):
            score += 1.0 / criteria_count

        # 4 - API usage examples
        api_indicators = [
            "usage",
            "example",
            "how to use",
            "quickstart",
            "tutorial",
            "from transformers",
            "import",
            "model.",
            "pipeline",
            "```python",
            "```py",
            "api",
            "inference",
        ]
        if any(indicator in readme_lower for indicator in api_indicators):
            score += 1.0 / criteria_count

        # check for tutorials/examples area with >= 1 item (+0.1 bonus, cap at 1.0)
        if context.hf_info and context.hf_info.get("files"):
            files = context.hf_info["files"]
            has_examples = any(
                "example" in file_path.lower()
                or "tutorial" in file_path.lower()
                or "notebook" in file_path.lower()
                or file_path.endswith(".ipynb")
                for file_path in files
            )
            if has_examples:
                score += 0.1

        return min(1.0, score)
