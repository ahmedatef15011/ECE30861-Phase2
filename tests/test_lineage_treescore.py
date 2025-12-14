"""
Tests for lineage extraction and TreeScore metric.
"""

import pytest
from unittest.mock import Mock

from src.lineage import LineageExtractor, LineageGraph, LineageNode, LineageEdge
from src.metrics.treescore import TreeScoreMetric, calculate_treescore
from src.models import ParsedURL, URLCategory, ModelContext, MetricResult


class TestLineageExtractor:
    """Tests for LineageExtractor."""

    def test_extract_from_config_name_or_path(self):
        """Test extraction from _name_or_path field."""
        extractor = LineageExtractor()
        
        config_data = {
            "config.json": {
                "_name_or_path": "meta-llama/Llama-2-7b-hf"
            }
        }
        
        parents = extractor._extract_from_config(config_data)
        
        assert len(parents) == 1
        assert parents[0][0] == "meta-llama/Llama-2-7b-hf"
        assert parents[0][1] == "fine_tuned_from"

    def test_extract_from_config_base_model(self):
        """Test extraction from base_model field."""
        extractor = LineageExtractor()
        
        config_data = {
            "config.json": {
                "base_model": "openai/whisper-small"
            }
        }
        
        parents = extractor._extract_from_config(config_data)
        
        assert len(parents) == 1
        assert parents[0][0] == "openai/whisper-small"
        # base_model is normalized to "derived_from" per OpenAPI spec
        assert parents[0][1] == "derived_from"

    def test_extract_from_config_merged_models(self):
        """Test extraction from merged_models list."""
        extractor = LineageExtractor()
        
        config_data = {
            "config.json": {
                "merged_models": [
                    "meta-llama/Llama-2-7b",
                    "mistralai/Mistral-7B-v0.1"
                ]
            }
        }
        
        parents = extractor._extract_from_config(config_data)
        
        assert len(parents) == 2
        assert any(p[0] == "meta-llama/Llama-2-7b" for p in parents)
        assert any(p[0] == "mistralai/Mistral-7B-v0.1" for p in parents)
        # merged_from is normalized to "derived_from" per OpenAPI spec
        assert all(p[1] == "derived_from" for p in parents)

    def test_extract_from_readme_fine_tuned(self):
        """Test extraction from README fine-tuned pattern."""
        extractor = LineageExtractor()
        
        readme = """
        # My Model
        
        This model is fine-tuned from [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
        on a custom dataset.
        """
        
        parents = extractor._extract_from_readme(readme)
        
        assert len(parents) >= 1
        # Should find meta-llama/Llama-2-7b
        found_llama = any("meta-llama/Llama-2-7b" in p[0] for p in parents)
        assert found_llama

    def test_extract_from_readme_yaml_base_model(self):
        """Test extraction from YAML front matter base_model."""
        extractor = LineageExtractor()
        
        readme = """---
base_model: mistralai/Mistral-7B-v0.1
license: apache-2.0
---

# My Fine-tuned Model
        """
        
        parents = extractor._extract_from_readme(readme)
        
        assert len(parents) >= 1
        assert any(p[0] == "mistralai/Mistral-7B-v0.1" for p in parents)

    def test_is_valid_model_ref(self):
        """Test model reference validation."""
        extractor = LineageExtractor()
        
        # Valid references
        assert extractor._is_valid_model_ref("meta-llama/Llama-2-7b")
        assert extractor._is_valid_model_ref("openai/whisper-small")
        assert extractor._is_valid_model_ref("https://huggingface.co/meta/model")
        
        # Invalid references
        assert not extractor._is_valid_model_ref("./local/path")
        assert not extractor._is_valid_model_ref("model.bin")
        assert not extractor._is_valid_model_ref("/absolute/path")
        assert not extractor._is_valid_model_ref("C:\\windows\\path")
        assert not extractor._is_valid_model_ref("ab")  # Too short

    def test_normalize_model_id(self):
        """Test model ID normalization."""
        extractor = LineageExtractor()
        
        # From URL
        result = extractor._normalize_model_id(
            "https://huggingface.co/meta-llama/Llama-2-7b"
        )
        assert result == "meta-llama/Llama-2-7b"
        
        # Already normalized
        result = extractor._normalize_model_id("openai/whisper-small")
        assert result == "openai/whisper-small"

    def test_extract_lineage_full(self):
        """Test full lineage extraction."""
        mock_hf_api = Mock()
        mock_hf_api.get_model_config.return_value = None
        mock_hf_api.get_readme_content.return_value = None
        
        extractor = LineageExtractor(hf_api=mock_hf_api)
        
        model_url = ParsedURL(
            url="https://huggingface.co/test/my-model",
            category=URLCategory.MODEL,
            name="my-model",
            platform="huggingface",
            owner="test",
            repo="my-model",
        )
        
        config_data = {
            "config.json": {
                "_name_or_path": "meta-llama/Llama-2-7b-hf"
            }
        }
        
        graph = extractor.extract_lineage(
            model_url=model_url,
            config_data=config_data,
            readme_content=None,
            max_depth=1,
            recursive=False,
        )
        
        assert len(graph.nodes) == 2  # Root + parent
        assert len(graph.edges) == 1
        
        # Check parent IDs
        parent_ids = graph.get_parent_ids()
        assert "meta-llama/Llama-2-7b-hf" in parent_ids

    def test_lineage_graph_operations(self):
        """Test LineageGraph add and get operations."""
        graph = LineageGraph()
        
        # Add nodes
        node1 = LineageNode(
            artifact_id="model1",
            name="Model 1",
            source="config"
        )
        node2 = LineageNode(
            artifact_id="model2",
            name="Model 2",
            source="readme"
        )
        
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node1)  # Duplicate - should not be added
        
        assert len(graph.nodes) == 2
        
        # Add edges
        edge1 = LineageEdge(
            from_node_id="model1",
            to_node_id="model2",
            relationship="based_on"
        )
        
        graph.add_edge(edge1)
        graph.add_edge(edge1)  # Duplicate - should not be added
        
        assert len(graph.edges) == 1
        
        # Get parent IDs
        parent_ids = graph.get_parent_ids()
        assert "model1" in parent_ids


class TestTreeScoreMetric:
    """Tests for TreeScoreMetric."""

    def test_treescore_no_parents(self):
        """Test TreeScore when no parents found."""
        mock_hf_api = Mock()
        mock_hf_api.get_model_config.return_value = None
        mock_hf_api.get_readme_content.return_value = None
        
        metric = TreeScoreMetric(hf_api=mock_hf_api)
        
        context = ModelContext(
            model_url=ParsedURL(
                url="https://huggingface.co/test/model",
                category=URLCategory.MODEL,
                name="model",
                platform="huggingface",
                owner="test",
                repo="model",
            )
        )
        
        result = metric.calculate(context)
        
        assert result.score == 0.0
        assert result.latency >= 0

    def test_treescore_with_parent_scores(self):
        """Test TreeScore with provided parent scores."""
        mock_hf_api = Mock()
        mock_hf_api.get_model_config.return_value = None
        mock_hf_api.get_readme_content.return_value = None
        
        metric = TreeScoreMetric(hf_api=mock_hf_api)
        
        context = ModelContext(
            model_url=ParsedURL(
                url="https://huggingface.co/test/model",
                category=URLCategory.MODEL,
                name="model",
                platform="huggingface",
                owner="test",
                repo="model",
            ),
            config_data={
                "config.json": {
                    "_name_or_path": "meta-llama/Llama-2-7b"
                }
            }
        )
        
        parent_scores = {
            "meta-llama/Llama-2-7b": 0.85
        }
        
        result = metric.calculate(context, parent_scores=parent_scores)
        
        assert result.score == 0.85  # Average of single parent
        assert result.latency >= 0

    def test_treescore_multiple_parents(self):
        """Test TreeScore with multiple parents."""
        mock_hf_api = Mock()
        mock_hf_api.get_model_config.return_value = None
        mock_hf_api.get_readme_content.return_value = None
        
        metric = TreeScoreMetric(hf_api=mock_hf_api)
        
        # Create lineage graph with multiple parents
        lineage_graph = LineageGraph()
        lineage_graph.add_node(LineageNode("test/model", "model", "root"))
        lineage_graph.add_node(LineageNode("parent1/model", "model1", "config"))
        lineage_graph.add_node(LineageNode("parent2/model", "model2", "config"))
        lineage_graph.add_edge(
            LineageEdge("parent1/model", "test/model", "merged_from")
        )
        lineage_graph.add_edge(
            LineageEdge("parent2/model", "test/model", "merged_from")
        )
        
        context = ModelContext(
            model_url=ParsedURL(
                url="https://huggingface.co/test/model",
                category=URLCategory.MODEL,
                name="model",
                platform="huggingface",
                owner="test",
                repo="model",
            )
        )
        
        parent_scores = {
            "parent1/model": 0.8,
            "parent2/model": 0.6,
        }
        
        result = metric.calculate(
            context,
            lineage_graph=lineage_graph,
            parent_scores=parent_scores
        )
        
        # Average of 0.8 and 0.6 = 0.7
        assert result.score == pytest.approx(0.7)

    def test_treescore_with_score_fetcher(self):
        """Test TreeScore with custom score fetcher."""
        mock_hf_api = Mock()
        
        def mock_score_fetcher(model_id):
            scores = {
                "meta-llama/Llama-2-7b": 0.9,
                "openai/whisper": 0.75,
            }
            return scores.get(model_id)
        
        metric = TreeScoreMetric(
            hf_api=mock_hf_api,
            score_fetcher=mock_score_fetcher
        )
        
        # Set up a lineage graph
        lineage_graph = LineageGraph()
        lineage_graph.add_node(LineageNode("test/model", "model", "root"))
        lineage_graph.add_node(LineageNode("meta-llama/Llama-2-7b", "Llama", "config"))
        lineage_graph.add_edge(
            LineageEdge("meta-llama/Llama-2-7b", "test/model", "fine_tuned_from")
        )
        
        context = ModelContext(
            model_url=ParsedURL(
                url="https://huggingface.co/test/model",
                category=URLCategory.MODEL,
                name="model",
                platform="huggingface",
                owner="test",
                repo="model",
            )
        )
        
        result = metric.calculate(context, lineage_graph=lineage_graph)
        
        assert result.score == 0.9

    def test_treescore_cache(self):
        """Test that TreeScoreMetric caches parent scores."""
        mock_hf_api = Mock()
        call_count = [0]
        
        def mock_score_fetcher(model_id):
            call_count[0] += 1
            return 0.8
        
        metric = TreeScoreMetric(
            hf_api=mock_hf_api,
            score_fetcher=mock_score_fetcher
        )
        
        # Manually set a cached score
        metric.set_parent_score("cached/model", 0.7)
        
        # Fetch the cached score
        score = metric._get_parent_score("cached/model")
        assert score == 0.7
        assert call_count[0] == 0  # Fetcher should not be called
        
        # Clear cache and fetch again
        metric.clear_cache()
        score = metric._get_parent_score("cached/model")
        assert score == 0.8  # Should now use fetcher
        assert call_count[0] == 1


class TestCalculateTreescore:
    """Tests for calculate_treescore convenience function."""

    def test_calculate_treescore_basic(self):
        """Test basic calculate_treescore call."""
        context = ModelContext(
            model_url=ParsedURL(
                url="https://huggingface.co/test/model",
                category=URLCategory.MODEL,
                name="model",
                platform="huggingface",
                owner="test",
                repo="model",
            )
        )
        
        result = calculate_treescore(context)
        
        assert isinstance(result, MetricResult)
        assert 0.0 <= result.score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
