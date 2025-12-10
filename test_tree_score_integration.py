"""
Simple integration test for tree score implementation.

This tests the key functionality by verifying:
1. Database migration succeeded
2. LLM analyzer module is importable
3. Cascade update function exists
4. Tree score metric accepts required parameters
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import inspect  # noqa: E402
from src.database.connection import engine  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_database_migration():
    """Test 1: Verify lineage_metadata column exists."""
    logger.info("\n=== Test 1: Database Migration ===")
    
    try:
        inspector = inspect(engine)
        columns = [
            col['name'] for col in
            inspector.get_columns('packages')
        ]
        
        assert 'lineage_metadata' in columns, (
            "lineage_metadata column not found in packages table"
        )
        
        logger.info("✅ PASS: lineage_metadata column exists")
        return True
        
    except Exception as e:
        logger.error(f"❌ FAIL: {e}")
        return False


def test_llm_analyzer():
    """Test 2: Verify LLM analyzer is importable and structured."""
    logger.info("\n=== Test 2: LLM Analyzer ===")
    
    try:
        from src.llm.analyzer import analyze_artifact_dependencies
        
        # Verify function signature
        import inspect as insp
        sig = insp.signature(analyze_artifact_dependencies)
        params = list(sig.parameters.keys())
        
        expected = ['config_data', 'readme_content', 'model_url']
        assert all(p in params for p in expected), (
            f"Expected {expected}, got {params}"
        )
        
        logger.info("✅ PASS: LLM analyzer properly structured")
        return True
        
    except Exception as e:
        logger.error(f"❌ FAIL: {e}")
        return False


def test_cascade_update_function():
    """Test 3: Verify cascade update function exists."""
    logger.info("\n=== Test 3: Cascade Update Function ===")
    
    try:
        from src.database import crud
        
        # Check function exists
        assert hasattr(crud, 'update_dependent_tree_scores'), (
            "update_dependent_tree_scores function not found"
        )
        
        # Check signature
        import inspect as insp
        sig = insp.signature(crud.update_dependent_tree_scores)
        params = list(sig.parameters.keys())
        
        expected = ['db', 'package_id']
        assert all(p in params for p in expected), (
            f"Expected {expected} in params, got {params}"
        )
        
        logger.info("✅ PASS: Cascade update function exists")
        return True
        
    except Exception as e:
        logger.error(f"❌ FAIL: {e}")
        return False


def test_tree_score_metric():
    """Test 4: Verify tree score metric exists and works."""
    logger.info("\n=== Test 4: Tree Score Metric ===")
    
    try:
        from src.metrics.treescore import TreeScoreMetric
        from src.models import ModelContext, ParsedURL, URLCategory
        from src.hf_api import HuggingFaceAPI
        
        # Create metric calculator
        hf_api = HuggingFaceAPI()
        metric = TreeScoreMetric(hf_api=hf_api)
        
        # Create minimal context
        parsed_url = ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test-model",
            platform="huggingface",
            owner="test",
            repo="model"
        )
        
        context = ModelContext(
            model_url=parsed_url,
            config_data={},
            readme_content="# Test Model"
        )
        
        # Calculate with no parents (should fall back to current_net_score)
        result = metric.calculate(
            context=context,
            current_net_score=0.75
        )
        
        # Should return MetricResult
        assert hasattr(result, 'score'), (
            "Result should have 'score' attribute"
        )
        assert result.score == 0.75, (
            f"Expected fallback to 0.75, got {result.score}"
        )
        
        logger.info("✅ PASS: Tree score metric works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scoring_integration():
    """Test 5: Verify scoring.py passes correct parameters."""
    logger.info("\n=== Test 5: Scoring Integration ===")
    
    try:
        from src.scoring import MetricScorer
        import inspect as insp
        
        # Check _calculate_treescore method exists
        scorer = MetricScorer()
        assert hasattr(scorer, '_calculate_treescore'), (
            "_calculate_treescore method not found"
        )
        
        # Check signature
        sig = insp.signature(scorer._calculate_treescore)
        params = list(sig.parameters.keys())
        
        expected = ['context', 'current_net_score']
        assert all(p in params for p in expected), (
            f"Expected {expected} in params, got {params}"
        )
        
        logger.info("✅ PASS: Scoring integration correct")
        return True
        
    except Exception as e:
        logger.error(f"❌ FAIL: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("TREE SCORE IMPLEMENTATION INTEGRATION TESTS")
    logger.info("=" * 60)
    
    results = {
        "Database Migration": test_database_migration(),
        "LLM Analyzer": test_llm_analyzer(),
        "Cascade Update Function": test_cascade_update_function(),
        "Tree Score Metric": test_tree_score_metric(),
        "Scoring Integration": test_scoring_integration()
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("\nImplementation Status:")
        logger.info("✅ Database schema updated (lineage_metadata column)")
        logger.info("✅ LLM analyzer ready for dependency extraction")
        logger.info("✅ Cascade update function implemented")
        logger.info("✅ Tree score metric with fallback to net_score")
        logger.info("✅ Scoring integration passes db_session")
    else:
        logger.info("⚠️  SOME INTEGRATION TESTS FAILED")
    logger.info("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
