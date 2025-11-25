from typing import Any, Dict

from ..models import MetricResult, ModelContext
from ..utils import measure_time
from .base import BaseMetric


class DatasetAndCodeScoreMetric(BaseMetric):
    """Metric for evaluating linked datasets and code quality."""

    @property
    def name(self) -> str:
        return "dataset_and_code_score"

    def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        """Compute dataset and code linkage score."""
        with measure_time() as get_latency:
            score = self._calculate_dataset_code_score(context, config)

        return MetricResult(score=score, latency=get_latency())

    def _calculate_dataset_code_score(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> float:
        """Calculate score based on dataset link and example train/test code.

        The metric gives 1.0 when both are present, 0.5 when only one,
        and 0.1 when neither is present.
        """
        has_dataset_link = False
        has_example_code = False

        # check for dataset link
        if context.datasets:
            has_dataset_link = True
        elif context.readme_content:
            readme_lower = context.readme_content.lower()
            dataset_indicators = [
                "dataset:",
                "training data",
                "train on",
                "trained on",
                "huggingface.co/datasets/",
                "dataset link",
                "data source",
            ]
            has_dataset_link = any(
                indicator in readme_lower for indicator in dataset_indicators
            )

        # check for model_index.json dataset references
        if (
            not has_dataset_link
            and context.hf_info
            and context.hf_info.get("model_index")
        ):
            has_dataset_link = True

        # check for example train/test code
        if context.code_repos:
            has_example_code = True
        elif context.readme_content:
            readme_lower = context.readme_content.lower()
            code_indicators = [
                "training script",
                "train.py",
                "fine-tune",
                "finetune",
                "example code",
                "training code",
                "github.com/",
                "colab",
                "jupyter",
                "notebook",
                "script",
                "example:",
                "tutorial",
            ]
            has_example_code = any(
                indicator in readme_lower for indicator in code_indicators
            )

        # check files for training/example scripts
        if not has_example_code and context.hf_info and context.hf_info.get("files"):
            files = context.hf_info["files"]
            script_files = any(
                file_path.endswith(".py")
                or file_path.endswith(".ipynb")
                or "train" in file_path.lower()
                or "example" in file_path.lower()
                for file_path in files
            )
            has_example_code = script_files

        # specification scoring rules
        if has_dataset_link and has_example_code:
            return 1.0  # both present
        elif has_dataset_link or has_example_code:
            return 0.5  # one present
        else:
            return 0.1  # neither present
