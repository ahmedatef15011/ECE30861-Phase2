"""Tests for model ingest functionality."""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from src.ingest import (
    IngestValidator,
    INGEST_QUALITY_GATE_METRICS,
)
from src.models import (
    AuditResult,
    SizeScore,
)
from src.storage import LocalStorageBackend


class TestIngestValidator:
    """Test IngestValidator quality gate logic."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return IngestValidator()

    def test_quality_gate_all_pass(self, validator):
        """Test quality gate passes when all metrics >= 0.5."""
        # Create mock audit result with all metrics passing
        audit_result = AuditResult(
            name="test/model",
            category="MODEL",
            net_score=0.75,
            net_score_latency=100,
            reproducibility=0.7,
            reproducibility_latency=50,
            code_quality=0.8,
            code_quality_latency=50,
            license=0.9,
            license_latency=10,
            dataset_quality=0.6,
            dataset_quality_latency=20,
            ramp_up_time=0.75,
            ramp_up_time_latency=30,
            bus_factor=0.65,
            bus_factor_latency=40,
            performance_claims=0.7,
            performance_claims_latency=15,
            dataset_and_code_score=0.8,
            dataset_and_code_score_latency=35,
            size_score=SizeScore(
                raspberry_pi=0.6,
                jetson_nano=0.7,
                desktop_pc=0.8,
                aws_server=0.9,
            ),
            size_score_latency=0,
            reviewedness=-1.0,  # Not applicable
            reviewedness_latency=0,
            treescore=0.0,
            treescore_latency=0,
        )

        passes, failing = validator._apply_quality_gate(audit_result)

        assert passes is True
        assert len(failing) == 0

    def test_quality_gate_some_fail(self, validator):
        """Test quality gate fails when any metric < 0.5."""
        audit_result = AuditResult(
            name="test/model",
            category="MODEL",
            net_score=0.4,
            net_score_latency=100,
            reproducibility=0.3,  # FAIL
            reproducibility_latency=50,
            code_quality=0.8,
            code_quality_latency=50,
            license=0.4,  # FAIL
            license_latency=10,
            dataset_quality=0.6,
            dataset_quality_latency=20,
            ramp_up_time=0.75,
            ramp_up_time_latency=30,
            bus_factor=0.65,
            bus_factor_latency=40,
            performance_claims=0.7,
            performance_claims_latency=15,
            dataset_and_code_score=0.8,
            dataset_and_code_score_latency=35,
            size_score=SizeScore(
                raspberry_pi=0.6,
                jetson_nano=0.7,
                desktop_pc=0.8,
                aws_server=0.9,
            ),
            size_score_latency=0,
            reviewedness=-1.0,  # Not applicable
            reviewedness_latency=0,
            treescore=0.0,
            treescore_latency=0,
        )

        passes, failing = validator._apply_quality_gate(audit_result)

        assert passes is False
        assert len(failing) == 2
        assert any(m["metric"] == "reproducibility" for m in failing)
        assert any(m["metric"] == "license" for m in failing)

    def test_quality_gate_edge_case_exactly_half(self, validator):
        """Test quality gate passes when metric is exactly 0.5."""
        audit_result = AuditResult(
            name="test/model",
            category="MODEL",
            net_score=0.5,
            net_score_latency=100,
            reproducibility=0.5,  # Exactly threshold
            reproducibility_latency=50,
            code_quality=0.5,
            code_quality_latency=50,
            license=0.5,
            license_latency=10,
            dataset_quality=0.5,
            dataset_quality_latency=20,
            ramp_up_time=0.5,
            ramp_up_time_latency=30,
            bus_factor=0.5,
            bus_factor_latency=40,
            performance_claims=0.5,
            performance_claims_latency=15,
            dataset_and_code_score=0.5,
            dataset_and_code_score_latency=35,
            size_score=SizeScore(
                raspberry_pi=0.5,
                jetson_nano=0.5,
                desktop_pc=0.5,
                aws_server=0.5,
            ),
            size_score_latency=0,
            reviewedness=-1.0,  # Not applicable
            reviewedness_latency=0,
            treescore=0.0,
            treescore_latency=0,
        )

        passes, failing = validator._apply_quality_gate(audit_result)

        assert passes is True
        assert len(failing) == 0

    def test_extract_scores(self, validator):
        """Test score extraction from audit result."""
        audit_result = AuditResult(
            name="test/model",
            category="MODEL",
            net_score=0.75,
            net_score_latency=100,
            reproducibility=0.7,
            reproducibility_latency=50,
            code_quality=0.8,
            code_quality_latency=50,
            license=0.9,
            license_latency=10,
            dataset_quality=0.6,
            dataset_quality_latency=20,
            ramp_up_time=0.75,
            ramp_up_time_latency=30,
            bus_factor=0.65,
            bus_factor_latency=40,
            performance_claims=0.7,
            performance_claims_latency=15,
            dataset_and_code_score=0.8,
            dataset_and_code_score_latency=35,
            size_score=SizeScore(
                raspberry_pi=0.6,
                jetson_nano=0.7,
                desktop_pc=0.8,
                aws_server=0.9,
            ),
            size_score_latency=0,
            reviewedness=0.0,
            reviewedness_latency=0,
            treescore=0.0,
            treescore_latency=0,
        )

        scores = validator._extract_scores(audit_result)

        assert "net_score" in scores
        assert "reproducibility" in scores
        assert "code_quality" in scores
        assert "license" in scores
        assert scores["reproducibility"] == 0.7
        assert scores["code_quality"] == 0.8

    def test_validate_ingest_candidate_passes(self, validator):
        """Test full ingest validation when model passes."""
        # Mock the scorer
        with patch.object(validator.scorer, "score_model") as mock_score:
            mock_result = AuditResult(
                name="bert-base-uncased",
                category="MODEL",
                net_score=0.75,
                net_score_latency=100,
                reproducibility=0.7,
                reproducibility_latency=50,
                code_quality=0.8,
                code_quality_latency=50,
                license=0.9,
                license_latency=10,
                dataset_quality=0.6,
                dataset_quality_latency=20,
                ramp_up_time=0.75,
                ramp_up_time_latency=30,
                bus_factor=0.65,
                bus_factor_latency=40,
                performance_claims=0.7,
                performance_claims_latency=15,
                dataset_and_code_score=0.8,
                dataset_and_code_score_latency=35,
                size_score=SizeScore(
                    raspberry_pi=0.6,
                    jetson_nano=0.7,
                    desktop_pc=0.8,
                    aws_server=0.9,
                ),
                size_score_latency=0,
                reviewedness=-1.0,  # Not applicable
                reviewedness_latency=0,
                treescore=0.0,
                treescore_latency=0,
            )
            mock_score.return_value = mock_result

            passes, result = validator.validate_ingest_candidate(
                "google/bert-base-uncased"
            )

            assert passes is True
            assert result["is_ingestible"] is True
            assert result["artifact_id"] is not None
            assert result["all_scores"]["reproducibility"] == 0.7

    def test_validate_ingest_candidate_fails(self, validator):
        """Test full ingest validation when model fails."""
        # Mock the scorer
        with patch.object(validator.scorer, "score_model") as mock_score:
            mock_result = AuditResult(
                name="unknown/model",
                category="MODEL",
                net_score=0.3,
                net_score_latency=100,
                reproducibility=0.1,  # FAIL
                reproducibility_latency=50,
                code_quality=0.8,
                code_quality_latency=50,
                license=0.2,  # FAIL
                license_latency=10,
                dataset_quality=0.6,
                dataset_quality_latency=20,
                ramp_up_time=0.75,
                ramp_up_time_latency=30,
                bus_factor=0.65,
                bus_factor_latency=40,
                performance_claims=0.7,
                performance_claims_latency=15,
                dataset_and_code_score=0.8,
                dataset_and_code_score_latency=35,
                size_score=SizeScore(
                    raspberry_pi=0.6,
                    jetson_nano=0.7,
                    desktop_pc=0.8,
                    aws_server=0.9,
                ),
                size_score_latency=0,
                reviewedness=-1.0,  # Not applicable
                reviewedness_latency=0,
                treescore=0.0,
                treescore_latency=0,
            )
            mock_score.return_value = mock_result

            passes, result = validator.validate_ingest_candidate(
                "unknown/model"
            )

            assert passes is False
            assert result["is_ingestible"] is False
            assert result["artifact_id"] is None
            assert len(result["failing_metrics"]) == 2


class TestLocalStorage:
    """Test local filesystem storage backend."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage backend."""
        storage_path = tmp_path / "storage"
        return LocalStorageBackend(str(storage_path))

    def test_store_artifact(self, temp_storage, tmp_path):
        """Test storing an artifact."""
        # Create a dummy artifact file
        artifact_file = tmp_path / "model.zip"
        artifact_file.write_text("dummy model content")

        metadata = {
            "model_name": "test/model",
            "scores": {"reproducibility": 0.7},
        }

        result = temp_storage.store_artifact(
            "artifact-123",
            str(artifact_file),
            metadata,
        )

        assert result["artifact_id"] == "artifact-123"
        assert result["filename"] == "artifact-123.zip"
        assert result["metadata"]["model_name"] == "test/model"

    def test_get_artifact(self, temp_storage, tmp_path):
        """Test retrieving artifact metadata."""
        # Store an artifact first
        artifact_file = tmp_path / "model.zip"
        artifact_file.write_text("dummy model content")

        metadata = {"model_name": "test/model"}
        temp_storage.store_artifact("artifact-123", str(artifact_file), metadata)

        # Retrieve it
        result = temp_storage.get_artifact("artifact-123")

        assert result is not None
        assert result["artifact_id"] == "artifact-123"
        assert result["metadata"]["model_name"] == "test/model"

    def test_list_artifacts(self, temp_storage, tmp_path):
        """Test listing artifacts."""
        # Store multiple artifacts
        for i in range(3):
            artifact_file = tmp_path / f"model{i}.zip"
            artifact_file.write_text(f"content {i}")
            metadata = {"model_name": f"test/model{i}"}
            temp_storage.store_artifact(
                f"artifact-{i}",
                str(artifact_file),
                metadata,
            )

        result = temp_storage.list_artifacts(limit=10)

        assert result["total"] == 3
        assert len(result["artifacts"]) == 3
        assert result["has_more"] is False

    def test_delete_artifact(self, temp_storage, tmp_path):
        """Test deleting an artifact."""
        # Store an artifact
        artifact_file = tmp_path / "model.zip"
        artifact_file.write_text("content")
        metadata = {"model_name": "test/model"}
        temp_storage.store_artifact("artifact-123", str(artifact_file), metadata)

        # Delete it
        deleted = temp_storage.delete_artifact("artifact-123")

        assert deleted is True
        assert temp_storage.get_artifact("artifact-123") is None

    def test_search_artifacts(self, temp_storage, tmp_path):
        """Test searching artifacts."""
        # Store artifacts with different model names
        for name in ["bert-base", "gpt2", "bert-large"]:
            artifact_file = tmp_path / f"{name}.zip"
            artifact_file.write_text("content")
            metadata = {"model_name": f"google/{name}"}
            temp_storage.store_artifact(f"{name}-id", str(artifact_file), metadata)

        result = temp_storage.search_artifacts("bert")

        assert len(result) == 2
        assert all("bert" in a["metadata"]["model_name"] for a in result)


class TestIngestQualityGateMetrics:
    """Test quality gate metric definitions."""

    def test_quality_gate_metrics_defined(self):
        """Test that quality gate metrics are properly defined."""
        # Updated for reviewedness metric
        assert len(INGEST_QUALITY_GATE_METRICS) == 9
        assert "reproducibility" in INGEST_QUALITY_GATE_METRICS
        assert "code_quality" in INGEST_QUALITY_GATE_METRICS
        assert "license" in INGEST_QUALITY_GATE_METRICS
        assert "reviewedness" in INGEST_QUALITY_GATE_METRICS

    def test_quality_gate_thresholds(self):
        """Test that all thresholds are 0.5."""
        for metric, threshold in INGEST_QUALITY_GATE_METRICS.items():
            assert threshold == 0.5, f"{metric} has incorrect threshold"
