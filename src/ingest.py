"""Model ingest module for quality gate validation.

Handles validation of HuggingFace models against quality metrics.
Models must have net_score >= 0.5 to be ingestible.
"""

import logging
from typing import Any, Dict, List, Tuple
from datetime import datetime
import uuid

from .scoring import MetricScorer
from .models import ModelContext, ParsedURL, URLCategory, AuditResult
from .hf_api import HuggingFaceAPI
from .utils import measure_time

logger = logging.getLogger(__name__)

# Quality gate threshold for ingest
# Net score must be >= 0.5 to pass
INGEST_NET_SCORE_THRESHOLD = 0.5


class IngestValidator:
    """Validates HuggingFace models for ingest using quality gate."""

    def __init__(self, config_path: str = "config/weights.yaml"):
        """Initialize ingest validator.
        
        Args:
            config_path: Path to config file with metric weights
        """
        self.scorer = MetricScorer(config_path)
        self.hf_api = HuggingFaceAPI()

    def validate_ingest_candidate(
        self, model_name: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a HuggingFace model for ingest.

        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b")

        Returns:
            Tuple of (is_ingestible, details_dict)
            - is_ingestible: bool indicating if model passes quality gate
            - details_dict: Contains scores, failing metrics, artifact_id
        """
        with measure_time() as get_latency:
            try:
                logger.info(f"Starting ingest validation for {model_name}")

                # Step 1: Parse model URL
                model_url = ParsedURL(
                    url=f"https://huggingface.co/{model_name}",
                    category=URLCategory.MODEL,
                    name=model_name,
                    platform="huggingface",
                    owner=model_name.split("/")[0],
                    repo=model_name.split("/")[1] if "/" in model_name else model_name,
                )

                # Step 2: Create model context
                context = ModelContext(model_url=model_url)

                # Step 3: Score the model
                logger.debug(f"Running metrics for {model_name}")
                audit_result = self.scorer.score_model(context)

                # Step 4: Apply quality gate
                passes, failing_metrics = self._apply_quality_gate(audit_result)

                # Step 5: Build response
                latency = get_latency()
                response = {
                    "model_name": model_name,
                    "is_ingestible": passes,
                    "latency_ms": latency,
                    "artifact_id": str(uuid.uuid4()) if passes else None,
                    "timestamp": datetime.utcnow().isoformat(),
                    "all_scores": self._extract_scores(audit_result),
                }

                if not passes:
                    response["failing_metrics"] = failing_metrics
                    logger.warning(
                        f"Model {model_name} failed ingest quality gate: "
                        f"{[m['metric'] for m in failing_metrics]}"
                    )
                else:
                    logger.info(f"Model {model_name} passed ingest quality gate")

                return passes, response

            except Exception as e:
                logger.error(
                    f"Error validating ingest for {model_name}: {e}",
                    exc_info=True
                )
                return False, {
                    "model_name": model_name,
                    "is_ingestible": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }

    def _apply_quality_gate(
        self, audit_result: AuditResult
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Apply quality gate check to audit results.

        Net score must be >= 0.5 to pass the quality gate.

        Args:
            audit_result: Scoring results from MetricScorer

        Returns:
            Tuple of (passes_gate, failing_metrics_list)
        """
        failing_metrics = []
        net_score = audit_result.net_score

        # Check net score against threshold
        if net_score < INGEST_NET_SCORE_THRESHOLD:
            failing_metrics.append({
                "metric": "net_score",
                "score": round(net_score, 3),
                "required": INGEST_NET_SCORE_THRESHOLD,
                "gap": round(net_score - INGEST_NET_SCORE_THRESHOLD, 3)
            })
            logger.debug(
                f"Net score failed: {net_score} < {INGEST_NET_SCORE_THRESHOLD}"
            )

        # Passes if no failing metrics
        passes = len(failing_metrics) == 0

        return passes, failing_metrics

    def _extract_scores(self, audit_result: AuditResult) -> Dict[str, float]:
        """
        Extract all scores from audit result.

        Args:
            audit_result: Scoring results

        Returns:
            Dictionary of metric scores
        """
        scores = {}

        # Extract all metric scores
        metric_names = [
            "reproducibility",
            "code_quality",
            "license",
            "dataset_quality",
            "ramp_up_time",
            "bus_factor",
            "performance_claims",
            "dataset_and_code_score",
            "reviewedness"
        ]
        
        for metric_name in metric_names:
            score = getattr(audit_result, metric_name, None)
            if score is not None:
                scores[metric_name] = round(score, 3)

        # Add net score
        scores["net_score"] = round(audit_result.net_score, 3)

        return scores


# Singleton instance
_validator = None


def get_validator(
    config_path: str = "config/weights.yaml",
) -> IngestValidator:
    """Get or create validator singleton."""
    global _validator
    if _validator is None:
        _validator = IngestValidator(config_path)
    return _validator


def validate_and_ingest(
    model_name: str,
    config_path: str = "config/weights.yaml",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to validate a model for ingest.

    Args:
        model_name: HuggingFace model name
        config_path: Path to config file

    Returns:
        Tuple of (passes_quality_gate, details)
    """
    validator = get_validator(config_path)
    return validator.validate_ingest_candidate(model_name)
