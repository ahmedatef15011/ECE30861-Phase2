"""
Test TreeScore calculation for a list of HuggingFace models.

This script:
1. Fetches model config and README from HuggingFace
2. Extracts lineage (parent models) using LLM analyzer
3. Calculates TreeScore using the TreeScoreMetric
4. Shows fallback behavior when no parents exist
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set UTF-8 encoding for console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from src.hf_api import HuggingFaceAPI  # noqa: E402
from src.models import (  # noqa: E402
    ParsedURL, URLCategory, ModelContext
)
from src.metrics.treescore import TreeScoreMetric  # noqa: E402
from src.llm.analyzer import (  # noqa: E402
    analyze_artifact_dependencies
)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Test models
MODEL_URLS = [
    "https://huggingface.co/google-bert/bert-base-uncased",
    "https://huggingface.co/parvk11/audience_classifier_model",
    "https://huggingface.co/distilbert-base-uncased-distilled-squad",
    "https://huggingface.co/caidas/swin2SR-lightweight-x2-64",
    "https://huggingface.co/vikhyatk/moondream2",
    "https://huggingface.co/microsoft/git-base",
    "https://huggingface.co/WinKawaks/vit-tiny-patch16-224",
    "https://huggingface.co/patrickjohncyh/fashion-clip",
    "https://huggingface.co/lerobot/diffusion_pusht",
    "https://huggingface.co/parthvpatil18/"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
    "https://huggingface.co/microsoft/resnet-50",
    "https://huggingface.co/crangana/trained-gender",
    "https://huggingface.co/onnx-community/trained-gender-ONNX"
]


def parse_model_url(url: str) -> ParsedURL:
    """Parse HuggingFace model URL into ParsedURL object."""
    parts = url.replace("https://huggingface.co/", "").split("/")
    
    if len(parts) >= 2:
        owner = parts[0]
        repo = parts[1]
        name = f"{owner}/{repo}"
    else:
        owner = None
        repo = parts[0] if parts else "unknown"
        name = repo
    
    return ParsedURL(
        url=url,
        category=URLCategory.MODEL,
        name=name,
        platform="huggingface",
        owner=owner,
        repo=repo
    )


def test_treescore_for_model(
    model_url: str,
    hf_api: HuggingFaceAPI,
    metric: TreeScoreMetric
) -> Dict[str, Any]:
    """
    Test TreeScore calculation for a single model.
    
    Returns:
        Dictionary with test results
    """
    logger.info(f"\nTesting: {model_url}")
    logger.info("=" * 80)
    
    try:
        # Parse URL
        parsed_url = parse_model_url(model_url)
        logger.info(f"  Model ID: {parsed_url.name}")
        
        # Fetch model info
        logger.info("  Fetching model data...")
        config_data = hf_api.get_model_config(parsed_url)
        readme_content = hf_api.get_readme_content(parsed_url)
        
        logger.info(f"  Config fetched: {len(config_data)} keys")
        logger.info(
            f"  README fetched: {len(readme_content)} chars"
        )
        
        # Extract lineage using LLM analyzer
        logger.info("\n  Analyzing dependencies with LLM...")
        try:
            lineage_metadata = analyze_artifact_dependencies(
                config_data=config_data,
                readme_content=readme_content,
                model_url=model_url
            )
            
            parent_models = lineage_metadata.get("parent_models", [])
            datasets = lineage_metadata.get("datasets", [])
            
            logger.info(f"  Found {len(parent_models)} parent model(s)")
            for parent in parent_models:
                logger.info(
                    f"    - {parent.get('id')} "
                    f"({parent.get('relationship', 'unknown')})"
                )
            
            if datasets:
                logger.info(f"  Found {len(datasets)} dataset(s)")
                for ds in datasets[:3]:  # Show first 3
                    logger.info(f"    - {ds}")
        
        except Exception as e:
            logger.warning(f"  LLM analysis failed: {e}")
            lineage_metadata = {
                "parent_models": [],
                "datasets": [],
                "code_repositories": []
            }
        
        # Create ModelContext
        context = ModelContext(
            model_url=parsed_url,
            config_data=config_data,
            readme_content=readme_content
        )
        
        # Calculate TreeScore
        logger.info("\n  Calculating TreeScore...")
        
        # Simulate net_score (in real scenario, this would come from
        # other metrics)
        simulated_net_score = 0.75
        
        result = metric.calculate(
            context=context,
            current_net_score=simulated_net_score
        )
        
        tree_score = result.score
        latency_ms = result.latency
        
        # Determine if fallback was used
        has_parents = len(parent_models) > 0
        used_fallback = not has_parents
        
        logger.info(
            f"\n  RESULT: TreeScore = {tree_score:.4f} "
            f"(latency: {latency_ms}ms)"
        )
        
        if used_fallback:
            logger.info(
                f"  Note: No parents found - used fallback to "
                f"net_score ({simulated_net_score})"
            )
        else:
            logger.info(
                f"  Note: Calculated from {len(parent_models)} "
                f"parent model(s)"
            )
        
        return {
            "url": model_url,
            "model_id": parsed_url.name,
            "success": True,
            "tree_score": tree_score,
            "latency_ms": latency_ms,
            "parent_count": len(parent_models),
            "used_fallback": used_fallback,
            "lineage_metadata": lineage_metadata
        }
    
    except Exception as e:
        logger.error(f"  ERROR: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        
        return {
            "url": model_url,
            "model_id": model_url.split("/")[-1],
            "success": False,
            "error": str(e)
        }


def main():
    """Run TreeScore tests on all models."""
    logger.info("=" * 80)
    logger.info("TREESCORE CALCULATION TEST")
    logger.info("=" * 80)
    logger.info(f"\nTesting {len(MODEL_URLS)} model(s)...\n")
    
    # Initialize API and metric
    hf_api = HuggingFaceAPI()
    metric = TreeScoreMetric(hf_api=hf_api)
    
    # Test each model
    results = []
    for i, url in enumerate(MODEL_URLS, 1):
        logger.info(f"\n[{i}/{len(MODEL_URLS)}]")
        result = test_treescore_for_model(url, hf_api, metric)
        results.append(result)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    logger.info(f"\nTotal Models: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if successful:
        logger.info("\nSuccessful Tests:")
        logger.info("-" * 80)
        
        with_parents = [
            r for r in successful if not r["used_fallback"]
        ]
        without_parents = [
            r for r in successful if r["used_fallback"]
        ]
        
        if with_parents:
            logger.info(
                f"\nModels with parent dependencies "
                f"({len(with_parents)}):"
            )
            for r in with_parents:
                logger.info(
                    f"  {r['model_id']}: "
                    f"TreeScore={r['tree_score']:.4f}, "
                    f"Parents={r['parent_count']}"
                )
        
        if without_parents:
            logger.info(
                f"\nModels without parents (fallback to net_score) "
                f"({len(without_parents)}):"
            )
            for r in without_parents:
                logger.info(
                    f"  {r['model_id']}: "
                    f"TreeScore={r['tree_score']:.4f} (fallback)"
                )
    
    if failed:
        logger.info("\nFailed Tests:")
        logger.info("-" * 80)
        for r in failed:
            logger.info(f"  {r['model_id']}: {r['error']}")
    
    # Performance stats
    if successful:
        avg_latency = sum(
            r['latency_ms'] for r in successful
        ) / len(successful)
        logger.info(
            f"\nAverage Latency: {avg_latency:.2f}ms per model"
        )
    
    logger.info("\n" + "=" * 80)
    
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
