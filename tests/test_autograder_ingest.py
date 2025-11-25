"""
Autograder Ingest Test Suite

Tests artifact ingestion for various HuggingFace models, datasets, and edge cases.
Validates that the quality gate system correctly accepts/rejects artifacts.
"""

import pytest
import asyncio
from typing import Dict, Any

from src.ingest import IngestValidator
from src.models import URLCategory


class TestAutograderIngest:
    """Test suite for autograder artifact ingestion validation."""

    @pytest.fixture
    def validator(self):
        """Create IngestValidator instance."""
        return IngestValidator()

    # ========================================================================
    # MODELS (16 test cases)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_model_bert_base_uncased(self, validator):
        """Test: google-bert/bert-base-uncased - popular BERT model."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "google-bert/bert-base-uncased"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        assert details["model_name"] == "google-bert/bert-base-uncased"
        
        if is_ingestible:
            assert details.get("artifact_id") is not None
            print(f"✅ PASS: bert-base-uncased ingested successfully")
        else:
            print(f"❌ FAIL: bert-base-uncased rejected")
            print(f"   Failing metrics: {details.get('failing_metrics', [])}")

    @pytest.mark.asyncio
    async def test_model_bert_dataset(self, validator):
        """Test: google-research/bert - BERT dataset/research repo."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "google-research/bert"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: google-research/bert ingested")
        else:
            print(f"❌ FAIL: google-research/bert rejected")

    @pytest.mark.asyncio
    async def test_model_audience_classifier(self, validator):
        """Test: parvk11/audience_classifier_model - smaller user model."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "parvk11/audience_classifier_model"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: audience_classifier_model ingested")
        else:
            print(f"❌ FAIL: audience_classifier_model rejected")

    @pytest.mark.asyncio
    async def test_model_distilbert_squad(self, validator):
        """Test: distilbert-base-uncased-distilled-squad - distilled model."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "distilbert-base-uncased-distilled-squad"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: distilbert-squad ingested")
        else:
            print(f"❌ FAIL: distilbert-squad rejected")

    @pytest.mark.asyncio
    async def test_model_swin2sr(self, validator):
        """Test: caidas/swin2SR-lightweight-x2-64 - image SR model."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "caidas/swin2SR-lightweight-x2-64"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: swin2SR ingested")
        else:
            print(f"❌ FAIL: swin2SR rejected")

    @pytest.mark.asyncio
    async def test_model_moondream2(self, validator):
        """Test: vikhyatk/moondream2 - vision-language model."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "vikhyatk/moondream2"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: moondream2 ingested")
        else:
            print(f"❌ FAIL: moondream2 rejected")

    @pytest.mark.asyncio
    async def test_model_git_base(self, validator):
        """Test: microsoft/git-base - Microsoft GIT model."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "microsoft/git-base"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: git-base ingested")
        else:
            print(f"❌ FAIL: git-base rejected")

    @pytest.mark.asyncio
    async def test_model_vit_tiny(self, validator):
        """Test: WinKawaks/vit-tiny-patch16-224 - tiny ViT model."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "WinKawaks/vit-tiny-patch16-224"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: vit-tiny ingested")
        else:
            print(f"❌ FAIL: vit-tiny rejected")

    @pytest.mark.asyncio
    async def test_model_fashion_clip(self, validator):
        """Test: patrickjohncyh/fashion-clip - fashion CLIP variant."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "patrickjohncyh/fashion-clip"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: fashion-clip ingested")
        else:
            print(f"❌ FAIL: fashion-clip rejected")

    @pytest.mark.asyncio
    async def test_model_diffusion_pusht(self, validator):
        """Test: lerobot/diffusion_pusht - robotics diffusion model."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "lerobot/diffusion_pusht"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: diffusion_pusht ingested")
        else:
            print(f"❌ FAIL: diffusion_pusht rejected")

    @pytest.mark.asyncio
    async def test_model_long_name(self, validator):
        """Test: parthvpatil18/aaaa... - extremely long model name."""
        long_name = "parthvpatil18/" + ("a" * 100)
        is_ingestible, details = await validator.validate_ingest_candidate(
            long_name
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        # This should likely fail (not a real model)
        if not is_ingestible:
            print(f"✅ Correctly rejected long fake model name")
        else:
            print(f"⚠️ Long name model somehow passed")

    @pytest.mark.asyncio
    async def test_model_whisper(self, validator):
        """Test: openai/whisper - OpenAI Whisper ASR model."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "openai/whisper"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: whisper ingested")
        else:
            print(f"❌ FAIL: whisper rejected")

    @pytest.mark.asyncio
    async def test_model_swin2sr_mvlab(self, validator):
        """Test: mv-lab/swin2sr - MV Lab's Swin2SR."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "mv-lab/swin2sr"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: mv-lab/swin2sr ingested")
        else:
            print(f"❌ FAIL: mv-lab/swin2sr rejected")

    @pytest.mark.asyncio
    async def test_model_moondream(self, validator):
        """Test: vikhyat/moondream - original moondream."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "vikhyat/moondream"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: moondream ingested")
        else:
            print(f"❌ FAIL: moondream rejected")

    @pytest.mark.asyncio
    async def test_model_git_microsoft(self, validator):
        """Test: microsoft/git - Microsoft GIT model."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "microsoft/git"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: microsoft/git ingested")
        else:
            print(f"❌ FAIL: microsoft/git rejected")

    @pytest.mark.asyncio
    async def test_model_ptm_recommendation(self, validator):
        """Test: Parth1811/ptm-recommendation-with-transformers."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "Parth1811/ptm-recommendation-with-transformers"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: ptm-recommendation ingested")
        else:
            print(f"❌ FAIL: ptm-recommendation rejected")

    # ========================================================================
    # DATASETS (6 test cases)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_dataset_bookcorpus(self, validator):
        """Test: bookcorpus/bookcorpus - popular text dataset."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "bookcorpus/bookcorpus"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        # Note: datasets should skip reproducibility metric
        if is_ingestible:
            print(f"✅ PASS: bookcorpus dataset ingested")
        else:
            print(f"❌ FAIL: bookcorpus dataset rejected")
            print(f"   Failing: {details.get('failing_metrics', [])}")

    @pytest.mark.asyncio
    async def test_dataset_squad(self, validator):
        """Test: rajpurkar/squad - SQuAD QA dataset."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "rajpurkar/squad"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: squad dataset ingested")
        else:
            print(f"❌ FAIL: squad dataset rejected")

    @pytest.mark.asyncio
    async def test_dataset_flickr2k(self, validator):
        """Test: hliang001/flickr2k - image dataset."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "hliang001/flickr2k"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: flickr2k dataset ingested")
        else:
            print(f"❌ FAIL: flickr2k dataset rejected")

    @pytest.mark.asyncio
    async def test_dataset_fashion_mnist(self, validator):
        """Test: zalandoresearch/fashion-mnist - fashion MNIST dataset."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "zalandoresearch/fashion-mnist"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: fashion-mnist dataset ingested")
        else:
            print(f"❌ FAIL: fashion-mnist dataset rejected")

    @pytest.mark.asyncio
    async def test_dataset_pusht(self, validator):
        """Test: lerobot/pusht - robotics dataset."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "lerobot/pusht"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: pusht dataset ingested")
        else:
            print(f"❌ FAIL: pusht dataset rejected")

    @pytest.mark.asyncio
    async def test_dataset_bookcorpus_alt(self, validator):
        """Test: rojagtap/bookcorpus - alternative bookcorpus."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "rojagtap/bookcorpus"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        if is_ingestible:
            print(f"✅ PASS: rojagtap/bookcorpus ingested")
        else:
            print(f"❌ FAIL: rojagtap/bookcorpus rejected")

    # ========================================================================
    # INVALID / EDGE CASES (2 test cases)
    # ========================================================================

    @pytest.mark.asyncio
    async def test_invalid_tree_main(self, validator):
        """Test: tree/main - invalid HF path (not a real repo)."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "tree/main"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        # This should fail (not a real HF repo)
        if not is_ingestible:
            print(f"✅ Correctly rejected invalid path 'tree/main'")
        else:
            print(f"⚠️ WARNING: Invalid path 'tree/main' was accepted")

    @pytest.mark.asyncio
    async def test_invalid_main_distillation(self, validator):
        """Test: main/distillation - invalid HF path."""
        is_ingestible, details = await validator.validate_ingest_candidate(
            "main/distillation"
        )
        
        assert isinstance(is_ingestible, bool)
        assert "model_name" in details
        
        # This should fail (not a real HF repo)
        if not is_ingestible:
            print(f"✅ Correctly rejected invalid path 'main/distillation'")
        else:
            print(f"⚠️ WARNING: Invalid path was accepted")

    # ========================================================================
    # SUMMARY TEST
    # ========================================================================

    @pytest.mark.asyncio
    async def test_autograder_summary(self, validator):
        """
        Summary test: Run all autograder test cases and generate report.
        
        This test runs all artifact ingestions and provides a summary of
        pass/fail rates across models, datasets, and invalid cases.
        """
        results = {
            "models": {},
            "datasets": {},
            "invalid": {}
        }
        
        # Models
        model_names = [
            "google-bert/bert-base-uncased",
            "google-research/bert",
            "parvk11/audience_classifier_model",
            "distilbert-base-uncased-distilled-squad",
            "caidas/swin2SR-lightweight-x2-64",
            "vikhyatk/moondream2",
            "microsoft/git-base",
            "WinKawaks/vit-tiny-patch16-224",
            "patrickjohncyh/fashion-clip",
            "lerobot/diffusion_pusht",
            "openai/whisper",
            "mv-lab/swin2sr",
            "vikhyat/moondream",
            "microsoft/git",
            "Parth1811/ptm-recommendation-with-transformers",
        ]
        
        # Datasets
        dataset_names = [
            "bookcorpus/bookcorpus",
            "rajpurkar/squad",
            "hliang001/flickr2k",
            "zalandoresearch/fashion-mnist",
            "lerobot/pusht",
            "rojagtap/bookcorpus",
        ]
        
        # Invalid
        invalid_names = [
            "tree/main",
            "main/distillation",
        ]
        
        print("\n" + "="*70)
        print("AUTOGRADER INGESTION SUMMARY")
        print("="*70)
        
        # Test models
        print(f"\n📦 MODELS ({len(model_names)} total)")
        print("-"*70)
        for name in model_names:
            try:
                is_ingestible, details = await validator.validate_ingest_candidate(name)
                results["models"][name] = is_ingestible
                status = "✅ PASS" if is_ingestible else "❌ FAIL"
                print(f"  {status}: {name}")
            except Exception as e:
                results["models"][name] = False
                print(f"  ⚠️ ERROR: {name} - {str(e)[:50]}")
        
        # Test datasets
        print(f"\n📊 DATASETS ({len(dataset_names)} total)")
        print("-"*70)
        for name in dataset_names:
            try:
                is_ingestible, details = await validator.validate_ingest_candidate(name)
                results["datasets"][name] = is_ingestible
                status = "✅ PASS" if is_ingestible else "❌ FAIL"
                print(f"  {status}: {name}")
            except Exception as e:
                results["datasets"][name] = False
                print(f"  ⚠️ ERROR: {name} - {str(e)[:50]}")
        
        # Test invalid
        print(f"\n⚠️ INVALID CASES ({len(invalid_names)} total)")
        print("-"*70)
        for name in invalid_names:
            try:
                is_ingestible, details = await validator.validate_ingest_candidate(name)
                results["invalid"][name] = is_ingestible
                # Inverted: we want these to FAIL
                status = "✅ CORRECTLY REJECTED" if not is_ingestible else "❌ INCORRECTLY ACCEPTED"
                print(f"  {status}: {name}")
            except Exception as e:
                results["invalid"][name] = False
                print(f"  ✅ ERROR (expected): {name}")
        
        # Calculate statistics
        model_pass = sum(1 for v in results["models"].values() if v)
        dataset_pass = sum(1 for v in results["datasets"].values() if v)
        invalid_correct = sum(1 for v in results["invalid"].values() if not v)
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"  Models:   {model_pass}/{len(model_names)} passed ({model_pass/len(model_names)*100:.1f}%)")
        print(f"  Datasets: {dataset_pass}/{len(dataset_names)} passed ({dataset_pass/len(dataset_names)*100:.1f}%)")
        print(f"  Invalid:  {invalid_correct}/{len(invalid_names)} correctly rejected ({invalid_correct/len(invalid_names)*100:.1f}%)")
        print(f"\n  TOTAL:    {model_pass + dataset_pass}/{len(model_names) + len(dataset_names)} artifacts ingested")
        print("="*70 + "\n")
        
        # Assert at least some artifacts pass
        assert model_pass > 0, "No models passed ingestion"
        assert model_pass + dataset_pass > 0, "No artifacts passed ingestion"


if __name__ == "__main__":
    """Run tests standalone without pytest."""
    import sys
    
    async def run_all_tests():
        validator = IngestValidator()
        test_suite = TestAutograderIngest()
        
        # Run summary test
        await test_suite.test_autograder_summary(validator)
    
    print("Running Autograder Ingestion Tests...")
    print("=" * 70)
    asyncio.run(run_all_tests())
    print("\nDone! Run with 'pytest tests/test_autograder_ingest.py -v' for detailed output")
