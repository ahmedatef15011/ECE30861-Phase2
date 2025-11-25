from abc import ABC, abstractmethod
from typing import Any, Dict

from ..models import MetricResult, ModelContext


# base class for all metrics
class BaseMetric(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        """
        Compute the metric score for a model context.
        Arguments:
            context: Model context with associated data
            config: Configuration dictionary with weights and thresholds

        Returns:
            MetricResult with score (0-1) and latency (ms)
        """
