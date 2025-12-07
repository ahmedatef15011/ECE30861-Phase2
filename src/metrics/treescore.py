"""
TreeScore metric implementation.

TreeScore represents the average of the total model scores of all parent
models according to the lineage graph. This provides a measure of the
trustworthiness of a model based on its ancestry.
"""

import time
from typing import Dict, Optional

from ..logging_utils import get_logger
from ..lineage import LineageExtractor, LineageGraph
from ..models import MetricResult, ModelContext
from ..hf_api import HuggingFaceAPI

logger = get_logger()


class TreeScoreMetric:
    """
    Calculates TreeScore based on parent model scores.
    
    TreeScore = Average of net_scores of all parent models in the lineage.
    
    If no parents are found, returns 0.0.
    If parents exist but no scores are available, returns 0.0.
    """
    
    def __init__(
        self,
        hf_api: Optional[HuggingFaceAPI] = None,
        score_fetcher: Optional[callable] = None,
    ):
        """
        Initialize TreeScoreMetric.
        
        Args:
            hf_api: HuggingFace API instance for fetching parent data
            score_fetcher: Optional callable that takes a model_id and returns
                           the net_score for that model. If not provided,
                           will try to compute scores or look up in database.
        """
        self.hf_api = hf_api or HuggingFaceAPI()
        self.lineage_extractor = LineageExtractor(hf_api=self.hf_api)
        self.score_fetcher = score_fetcher
        self._score_cache: Dict[str, float] = {}
    
    def calculate(
        self,
        context: ModelContext,
        lineage_graph: Optional[LineageGraph] = None,
        parent_scores: Optional[Dict[str, float]] = None,
    ) -> MetricResult:
        """
        Calculate TreeScore for a model.
        
        Args:
            context: Model context with config_data and readme_content
            lineage_graph: Pre-computed lineage graph (optional)
            parent_scores: Pre-fetched parent scores {model_id: net_score}
            
        Returns:
            MetricResult with treescore and latency
        """
        start_time = time.time()
        
        try:
            # Get or compute lineage graph
            if lineage_graph is None:
                lineage_graph = self.lineage_extractor.extract_lineage(
                    model_url=context.model_url,
                    config_data=context.config_data,
                    readme_content=context.readme_content,
                    max_depth=3,
                    recursive=True,
                )
            
            # Get parent model IDs
            parent_ids = lineage_graph.get_parent_ids()
            
            if not parent_ids:
                logger.info("No parent models found, TreeScore = 0.0")
                latency = int((time.time() - start_time) * 1000)
                return MetricResult(score=0.0, latency=latency)
            
            logger.info(f"Found {len(parent_ids)} parent models: {parent_ids}")
            
            # Get scores for parent models
            scores = []
            for parent_id in parent_ids:
                score = self._get_parent_score(parent_id, parent_scores)
                if score is not None and score >= 0:
                    scores.append(score)
                    logger.debug(f"Parent {parent_id} score: {score}")
            
            # Calculate average
            if not scores:
                logger.info("No valid parent scores found, TreeScore = 0.0")
                treescore = 0.0
            else:
                treescore = sum(scores) / len(scores)
                logger.info(
                    f"TreeScore = avg({scores}) = {treescore:.4f} "
                    f"({len(scores)} parents with scores)"
                )
            
            latency = int((time.time() - start_time) * 1000)
            return MetricResult(score=treescore, latency=latency)
            
        except Exception as e:
            logger.error(f"Error calculating TreeScore: {e}")
            latency = int((time.time() - start_time) * 1000)
            return MetricResult(score=0.0, latency=latency)
    
    def _get_parent_score(
        self,
        model_id: str,
        parent_scores: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """
        Get the net_score for a parent model.
        
        Tries in order:
        1. Pre-provided parent_scores dict
        2. Score cache
        3. Score fetcher callable
        4. Returns None if not available
        """
        # Check pre-provided scores
        if parent_scores and model_id in parent_scores:
            return parent_scores[model_id]
        
        # Check cache
        if model_id in self._score_cache:
            return self._score_cache[model_id]
        
        # Use score fetcher if provided
        if self.score_fetcher:
            try:
                score = self.score_fetcher(model_id)
                if score is not None:
                    self._score_cache[model_id] = score
                    return score
            except Exception as e:
                logger.debug(f"Score fetcher failed for {model_id}: {e}")
        
        # Could not get score
        logger.debug(f"No score available for parent model: {model_id}")
        return None
    
    def set_parent_score(self, model_id: str, score: float):
        """Manually set a parent's score in the cache."""
        self._score_cache[model_id] = score
    
    def clear_cache(self):
        """Clear the score cache."""
        self._score_cache.clear()


def calculate_treescore(
    context: ModelContext,
    hf_api: Optional[HuggingFaceAPI] = None,
    parent_scores: Optional[Dict[str, float]] = None,
    db_session = None,
) -> MetricResult:
    """
    Convenience function to calculate TreeScore.
    
    Args:
        context: Model context
        hf_api: HuggingFace API instance
        parent_scores: Optional pre-fetched parent scores
        db_session: Optional database session for looking up parent scores
        
    Returns:
        MetricResult with treescore
    """
    # Create score fetcher from database if session provided
    score_fetcher = None
    if db_session:
        def db_score_fetcher(model_id: str) -> Optional[float]:
            """Fetch score from database by model name."""
            try:
                # Import here to avoid circular imports
                from ..database import crud
                
                # Extract model name from model_id (owner/repo -> repo)
                if "/" in model_id:
                    model_name = model_id.split("/")[-1]
                else:
                    model_name = model_id
                
                logger.debug(f"Looking up parent score for: {model_name}")
                
                # Use the new get_packages_by_name function
                packages = crud.get_packages_by_name(db_session, model_name)
                
                if packages:
                    logger.debug(f"Found {len(packages)} packages for {model_name}")
                    # Get the first match with a valid rating
                    for pkg in packages:
                        rating = crud.get_package_rating(db_session, pkg.id)
                        if rating and rating.net_score >= 0:
                            logger.info(
                                f"Found parent score: {model_name} = "
                                f"{rating.net_score:.4f}"
                            )
                            return rating.net_score
                else:
                    logger.debug(f"No packages found for {model_name}")
                
                return None
            except Exception as e:
                logger.debug(f"DB score lookup failed for {model_id}: {e}")
                return None
        
        score_fetcher = db_score_fetcher
    
    metric = TreeScoreMetric(hf_api=hf_api, score_fetcher=score_fetcher)
    return metric.calculate(context, parent_scores=parent_scores)
