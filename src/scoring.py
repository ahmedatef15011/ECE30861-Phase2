from pathlib import Path
from typing import Any, Dict, List

import yaml

from .hf_api import HuggingFaceAPI
from .logging_utils import get_logger
from .metrics.bus_factor import BusFactorMetric
from .metrics.code_quality import CodeQualityMetric
from .metrics.dataset_and_code import DatasetAndCodeScoreMetric
from .metrics.dataset_quality import DatasetQualityMetric
from .metrics.license_score import LicenseScoreMetric
from .metrics.performance_claims import PerformanceClaimsMetric
from .metrics.ramp_up import RampUpTimeMetric
from .metrics.Reproducibility import ReproducibilityMetric
from .metrics.reviewedness import ReviewednessMetric
from .metrics.size_score import SizeScoreMetric
from .models import AuditResult, MetricResult, ModelContext, ParsedURL, URLCategory
from .utils import measure_time

logger = get_logger()


class MetricScorer:
    # handles parallel metric computation and scoring

    def __init__(self, config_path: str = "config/weights.yaml"): 
        self.config = self._load_config(config_path) #loads weights+ thresholds
        self.metrics = [
            #grouping order here
            RampUpTimeMetric(),
            BusFactorMetric(),
            PerformanceClaimsMetric(),
            LicenseScoreMetric(),
            SizeScoreMetric(),
            DatasetAndCodeScoreMetric(),
            DatasetQualityMetric(),
            CodeQualityMetric(),
            ReproducibilityMetric(),
            ReviewednessMetric(),
        ]
        self.hf_api = HuggingFaceAPI() #hf client for enhancing

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        # load configuration from YAML file
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file {config_path} not found, using defaults")
                return self._get_default_config() #fallsback if file might be missing

            with open(config_file, "r") as f:
                return yaml.safe_load(f) #simple yaml -> dict
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config() #fallback on parse errors

    def _get_default_config(self) -> Dict[str, Any]:
        # return default config
        return {
            "metric_weights": {
                "ramp_up_time": 0.15,
                "bus_factor": 0.10,
                "performance_claims": 0.15,
                "license": 0.10,
                "size_score": 0.15,
                "dataset_and_code_score": 0.15,
                "dataset_quality": 0.10,
                "code_quality": 0.10,
                "reproducibility": 0.10,
                "reviewedness": 0.10,
                "treescore": 0.0,  # Not implemented yet
            },
            "thresholds": {
                "size_limits": {
                    "raspberry_pi": 1.0,
                    "jetson_nano": 4.0,
                    "desktop_pc": 16.0,
                    "aws_server": 64.0,
                }
            },
        }
    
    # score a model using all metrics sequentially
    def score_model(self, context: ModelContext) -> AuditResult:
        self._enrich_context(context)

        # compute all metrics sequentially
        with measure_time() as get_net_latency:
            metric_results = self._compute_metrics(context)

            # calculate net score
            net_score = self._calculate_net_score(metric_results)

        # build audit result - recalculate size_score breakdown for AuditResult
        size_score_metric = SizeScoreMetric()
        size_score_config = self.config.get("metrics", {}).get("size_score", {})
        size_score_breakdown = size_score_metric._calculate_size_scores(
            context, size_score_config
        )

        # Extract default MetricResult to avoid duplication
        reproducibility_default = MetricResult(score=0.0, latency=0)
        reviewedness_default = MetricResult(score=-1.0, latency=0)
        
        return AuditResult(
            name=context.model_url.name,
            category="MODEL",
            net_score=net_score,
            net_score_latency=get_net_latency(),
            ramp_up_time=metric_results["ramp_up_time"].score,
            ramp_up_time_latency=metric_results["ramp_up_time"].latency,
            bus_factor=metric_results["bus_factor"].score,
            bus_factor_latency=metric_results["bus_factor"].latency,
            performance_claims=metric_results["performance_claims"].score,
            performance_claims_latency=metric_results["performance_claims"].latency,
            license=metric_results["license"].score,
            license_latency=metric_results["license"].latency,
            size_score=size_score_breakdown,
            size_score_latency=metric_results["size_score"].latency,
            dataset_and_code_score=metric_results["dataset_and_code_score"].score,
            dataset_and_code_score_latency=metric_results["dataset_and_code_score"].latency,
            dataset_quality=metric_results["dataset_quality"].score,
            dataset_quality_latency=metric_results["dataset_quality"].latency,
            code_quality=metric_results["code_quality"].score,
            code_quality_latency=metric_results["code_quality"].latency,
            reproducibility=metric_results.get("reproducibility", reproducibility_default).score,
            reproducibility_latency=metric_results.get("reproducibility", reproducibility_default).latency,
            reviewedness=metric_results.get("reviewedness", reviewedness_default).score,
            reviewedness_latency=metric_results.get("reviewedness", reviewedness_default).latency,
            treescore=0.0,  # Placeholder for now (not implemented yet)
            treescore_latency=0,
        )

    def _enrich_context(self, context: ModelContext):
        # enrich context with data from APIs - SEQUENTIAL VERSION
        import re
        
        # Fetch HF data synchronously
        try:
            context.hf_info = self.hf_api.get_model_info(context.model_url)
        except Exception as e:
            logger.error(f"Failed to fetch hf_info: {e}")
            context.hf_info = None
            
        try:
            context.readme_content = self.hf_api.get_readme_content(context.model_url)
        except Exception as e:
            logger.error(f"Failed to fetch readme: {e}")
            context.readme_content = None
            
        try:
            context.config_data = self.hf_api.get_model_config(context.model_url)
        except Exception as e:
            logger.error(f"Failed to fetch config: {e}")
            context.config_data = None
        
        # Extract GitHub repository from README (if available)
        if context.readme_content:
            try:
                # Look for GitHub URLs in README
                github_patterns = [
                    r'https?://github\.com/([^/\s]+)/([^/\s\)]+)',
                    r'github\.com/([^/\s]+)/([^/\s\)]+)',
                ]
                
                for pattern in github_patterns:
                    matches = re.findall(pattern, context.readme_content)
                    if matches:
                        # Take the first match
                        owner, repo = matches[0]
                        # Clean up repo name (remove trailing punctuation)
                        repo = re.sub(r'[.,;:\)]$', '', repo)
                        
                        github_url = f"https://github.com/{owner}/{repo}"
                        github_parsed = ParsedURL(
                            url=github_url,
                            category=URLCategory.CODE,
                            name=f"{owner}/{repo}",
                            platform="github",
                            owner=owner,
                            repo=repo,
                        )
                        context.code_repos.append(github_parsed)
                        logger.info(f"Found GitHub repo in README: {github_url}")
                        break
            except Exception as e:
                logger.error(f"Error extracting GitHub repo from README: {e}")
        
        logger.info(f"Enriched context for {context.model_url.name}")

    # compute all metrics sequentially
    def _compute_metrics(
        self, context: ModelContext
    ) -> Dict[str, Any]:
        # execute all tasks sequentially
        results = {}
        
        for metric in self.metrics:
            metric_name = metric.name
            try:
                result = metric.compute(context, self.config)
                if isinstance(result, MetricResult):
                    results[metric_name] = result
                else:
                    logger.warning(
                        f"Unexpected result type for {metric_name}: {type(result)}"
                    )
                    results[metric_name] = MetricResult(score=0.0, latency=0)
            except Exception as e:
                logger.error(f"Error computing {metric_name}: {e}")
                results[metric_name] = MetricResult(score=0.0, latency=0)
        
        return results


    def _calculate_net_score(self, metric_results: Dict[str, Any]) -> float:
        # calculate weighted net score from individual metrics
        weights = self.config.get("metric_weights", {})

        total_score = 0.0
        total_weight = 0.0

        for metric_name, result in metric_results.items():
            if metric_name.endswith("_latency"):
                continue  # skip latency fields

            weight = weights.get(metric_name, 0.0)
            if isinstance(result, MetricResult):
                # Skip metrics with negative scores (not applicable)
                if result.score < 0:
                    continue
                    
                total_score += result.score * weight
                total_weight += weight

        # return average of present scores
        if total_weight == 0.0:
            present_scores: List[float] = []
            for metric_name, result in metric_results.items():
                if metric_name.endswith("_latency"):
                    continue
                if isinstance(result, MetricResult) and result.score >= 0:
                    present_scores.append(result.score)
            
            net_score = sum(present_scores) / max(len(present_scores), 1)
        else:
            net_score = total_score / total_weight

        return max(0.0, min(1.0, net_score))
