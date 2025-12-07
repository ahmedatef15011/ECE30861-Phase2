"""
Lineage extraction module for HuggingFace models.

Extracts parent model references from config.json and model cards to build
a lineage graph showing model derivation relationships (fine-tuned from,
based on, etc.)
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from .logging_utils import get_logger
from .hf_api import HuggingFaceAPI
from .models import ParsedURL, URLCategory

logger = get_logger()


@dataclass
class LineageNode:
    """Represents a node in the lineage graph."""
    artifact_id: str  # HuggingFace repo ID (e.g., "meta-llama/Llama-2-7b")
    name: str
    source: str  # "huggingface", "config.json", "readme", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageEdge:
    """Represents an edge in the lineage graph."""
    from_node_id: str  # Parent model ID
    to_node_id: str    # Child model ID (the model being analyzed)
    relationship: str  # "fine_tuned_from", "based_on", "merged_from", etc.


@dataclass
class LineageGraph:
    """Complete lineage graph for a model."""
    nodes: List[LineageNode] = field(default_factory=list)
    edges: List[LineageEdge] = field(default_factory=list)
    
    def add_node(self, node: LineageNode):
        """Add a node if it doesn't already exist."""
        if not any(n.artifact_id == node.artifact_id for n in self.nodes):
            self.nodes.append(node)
    
    def add_edge(self, edge: LineageEdge):
        """Add an edge if it doesn't already exist."""
        existing = any(
            e.from_node_id == edge.from_node_id and 
            e.to_node_id == edge.to_node_id and
            e.relationship == edge.relationship
            for e in self.edges
        )
        if not existing:
            self.edges.append(edge)
    
    def get_parent_ids(self) -> List[str]:
        """Get all parent model IDs (nodes that are sources of edges)."""
        # The "from_node_id" in edges represents parents
        parent_ids = set()
        for edge in self.edges:
            parent_ids.add(edge.from_node_id)
        return list(parent_ids)


class LineageExtractor:
    """
    Extracts lineage information from HuggingFace model metadata.
    
    Analyzes config.json, model cards, and other structured metadata
    to identify parent models and build a lineage graph.
    """
    
    # Fields in config.json that may contain parent model references
    CONFIG_PARENT_FIELDS = [
        "_name_or_path",      # Most common - original model path
        "base_model",         # Explicit base model reference
        "parent_model",       # Some models use this
        "pretrained_model_name_or_path",
        "model_name_or_path",
        "finetuned_from",
        "base_model_name_or_path",
    ]
    
    # Fields for merged models (multiple parents)
    CONFIG_MERGE_FIELDS = [
        "merged_models",
        "merge_sources",
        "models",  # For model merge configs
    ]
    
    # Patterns for extracting HuggingFace model IDs from text
    HF_MODEL_PATTERNS = [
        # Full URL: https://huggingface.co/org/model
        r'https?://huggingface\.co/([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+)',
        # Direct reference: org/model-name
        r'([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+)',
    ]
    
    # README patterns indicating parent model references
    README_PARENT_PATTERNS = [
        (r'fine[- ]?tuned?\s+(?:from|on)\s+\[?([^\]\n]+)\]?', 'fine_tuned_from'),
        (r'based\s+on\s+\[?([^\]\n]+)\]?', 'based_on'),
        (r'derived\s+from\s+\[?([^\]\n]+)\]?', 'derived_from'),
        (r'starting\s+from\s+\[?([^\]\n]+)\]?', 'based_on'),
        (r'trained\s+from\s+\[?([^\]\n]+)\]?', 'trained_from'),
        (r'initialized\s+from\s+\[?([^\]\n]+)\]?', 'initialized_from'),
        (r'(?:is\s+)?(?:a\s+)?(?:PEFT|LoRA|QLoRA)\s+(?:fine[- ]?tuned?\s+)?(?:version\s+)?(?:of|from)\s+\[?([^\]\n]+)\]?', 'peft_of'),
    ]
    
    def __init__(self, hf_api: Optional[HuggingFaceAPI] = None):
        self.hf_api = hf_api or HuggingFaceAPI()
        self._visited: Set[str] = set()  # Track visited models to avoid cycles
    
    def extract_lineage(
        self,
        model_url: ParsedURL,
        config_data: Optional[Dict[str, Any]] = None,
        readme_content: Optional[str] = None,
        max_depth: int = 3,
        recursive: bool = True,
    ) -> LineageGraph:
        """
        Extract lineage graph for a model.
        
        Args:
            model_url: Parsed URL of the model
            config_data: Pre-fetched config data (optional)
            readme_content: Pre-fetched README content (optional)
            max_depth: Maximum depth for recursive parent lookup
            recursive: Whether to recursively fetch parent info
            
        Returns:
            LineageGraph with nodes and edges
        """
        self._visited.clear()
        graph = LineageGraph()
        
        model_id = self._get_model_id(model_url)
        if not model_id:
            logger.warning("Could not determine model ID for lineage extraction")
            return graph
        
        # Add the root node (the model being analyzed)
        root_node = LineageNode(
            artifact_id=model_id,
            name=model_id.split("/")[-1] if "/" in model_id else model_id,
            source="huggingface",
            metadata={"is_root": True}
        )
        graph.add_node(root_node)
        
        # Extract parents recursively
        self._extract_parents(
            model_id=model_id,
            config_data=config_data,
            readme_content=readme_content,
            graph=graph,
            depth=0,
            max_depth=max_depth,
            recursive=recursive,
        )
        
        logger.info(f"Extracted lineage: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        return graph
    
    def _extract_parents(
        self,
        model_id: str,
        config_data: Optional[Dict[str, Any]],
        readme_content: Optional[str],
        graph: LineageGraph,
        depth: int,
        max_depth: int,
        recursive: bool,
    ):
        """Recursively extract parent models."""
        if depth >= max_depth:
            logger.debug(f"Max depth {max_depth} reached for {model_id}")
            return
        
        if model_id in self._visited:
            logger.debug(f"Already visited {model_id}, skipping to avoid cycle")
            return
        
        self._visited.add(model_id)
        
        # Extract parents from config.json
        parent_refs = self._extract_from_config(config_data)
        
        # Extract parents from README
        readme_refs = self._extract_from_readme(readme_content)
        parent_refs.extend(readme_refs)
        
        # Process each parent reference
        for parent_id, relationship in parent_refs:
            # Validate and normalize the parent ID
            normalized_id = self._normalize_model_id(parent_id)
            if not normalized_id:
                continue
            
            # Skip self-references
            if normalized_id == model_id:
                continue
            
            # Add parent node
            parent_node = LineageNode(
                artifact_id=normalized_id,
                name=normalized_id.split("/")[-1] if "/" in normalized_id else normalized_id,
                source="config.json" if "config" in relationship else "readme",
                metadata={"relationship": relationship}
            )
            graph.add_node(parent_node)
            
            # Add edge from parent to child
            edge = LineageEdge(
                from_node_id=normalized_id,
                to_node_id=model_id,
                relationship=relationship
            )
            graph.add_edge(edge)
            
            # Recursively get parent's lineage
            if recursive and depth + 1 < max_depth:
                try:
                    parent_parsed = self._create_parsed_url(normalized_id)
                    if parent_parsed:
                        parent_config = self.hf_api.get_model_config(parent_parsed)
                        parent_readme = self.hf_api.get_readme_content(parent_parsed)
                        self._extract_parents(
                            model_id=normalized_id,
                            config_data=parent_config,
                            readme_content=parent_readme,
                            graph=graph,
                            depth=depth + 1,
                            max_depth=max_depth,
                            recursive=recursive,
                        )
                except Exception as e:
                    logger.debug(f"Could not fetch parent {normalized_id}: {e}")
    
    def _extract_from_config(
        self, config_data: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, str]]:
        """
        Extract parent model references from config.json.
        
        Returns list of (parent_id, relationship) tuples.
        """
        parents = []
        
        if not config_data:
            return parents
        
        # Check config.json if present
        config = config_data.get("config.json", {})
        if isinstance(config, dict):
            # Check single parent fields
            for field in self.CONFIG_PARENT_FIELDS:
                value = config.get(field)
                if value and isinstance(value, str):
                    # Skip if it's just a local path or filename
                    if self._is_valid_model_ref(value):
                        relationship = self._field_to_relationship(field)
                        parents.append((value, relationship))
            
            # Check merge fields (multiple parents)
            for field in self.CONFIG_MERGE_FIELDS:
                value = config.get(field)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and self._is_valid_model_ref(item):
                            parents.append((item, "merged_from"))
                        elif isinstance(item, dict):
                            # Some merge configs have structured entries
                            model_name = item.get("model") or item.get("name")
                            if model_name and self._is_valid_model_ref(model_name):
                                parents.append((model_name, "merged_from"))
        
        # Also check model_index.json for pipeline components
        model_index = config_data.get("model_index.json", {})
        if isinstance(model_index, list):
            for entry in model_index:
                if isinstance(entry, dict):
                    base = entry.get("base_model")
                    if base and self._is_valid_model_ref(base):
                        parents.append((base, "based_on"))
        
        return parents
    
    def _extract_from_readme(
        self, readme_content: Optional[str]
    ) -> List[Tuple[str, str]]:
        """
        Extract parent model references from README content.
        
        Returns list of (parent_id, relationship) tuples.
        """
        parents = []
        
        if not readme_content:
            return parents
        
        # Look for common patterns indicating parent models
        for pattern, relationship in self.README_PARENT_PATTERNS:
            matches = re.findall(pattern, readme_content, re.IGNORECASE)
            for match in matches:
                # Clean up the match
                model_ref = match.strip()
                
                # Try to extract a valid model ID from the match
                extracted = self._extract_model_id_from_text(model_ref)
                if extracted and self._is_valid_model_ref(extracted):
                    parents.append((extracted, relationship))
        
        # Also look for YAML front matter with base_model field
        yaml_match = re.search(r'^---\s*\n(.*?)\n---', readme_content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
            # Look for base_model field
            base_model_match = re.search(
                r'base_model:\s*["\']?([^"\'\n]+)["\']?', 
                yaml_content
            )
            if base_model_match:
                base_model = base_model_match.group(1).strip()
                if self._is_valid_model_ref(base_model):
                    parents.append((base_model, "base_model"))
        
        return parents
    
    def _is_valid_model_ref(self, value: str) -> bool:
        """Check if a string looks like a valid HuggingFace model reference."""
        if not value or len(value) < 3:
            return False
        
        # Skip obvious non-model references
        skip_patterns = [
            r'^\./',           # Local paths
            r'^\.\./',         # Relative paths
            r'^/',             # Absolute paths
            r'^[a-z]:\\',      # Windows paths
            r'\.bin$',         # Binary files
            r'\.safetensors$', # Safetensor files
            r'\.pt$',          # PyTorch files
            r'^https?://(?!huggingface\.co)',  # Non-HF URLs
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, value, re.IGNORECASE):
                return False
        
        # Should look like "org/model" or a URL
        if "/" in value:
            # HuggingFace URL
            if "huggingface.co" in value:
                return True
            # Standard org/model format
            parts = value.split("/")
            if len(parts) == 2:
                org, model = parts
                # Basic validation
                if org and model and len(org) >= 1 and len(model) >= 1:
                    return True
        
        return False
    
    def _normalize_model_id(self, value: str) -> Optional[str]:
        """Normalize a model reference to org/model format."""
        if not value:
            return None
        
        # Extract from HuggingFace URL
        hf_match = re.search(
            r'huggingface\.co/([a-zA-Z0-9_-]+/[a-zA-Z0-9._-]+)', 
            value
        )
        if hf_match:
            return hf_match.group(1)
        
        # Already in org/model format
        if "/" in value:
            parts = value.strip().split("/")
            if len(parts) >= 2:
                # Take last two parts (org/model)
                org = parts[-2]
                model = parts[-1]
                # Clean up model name
                model = re.sub(r'[^\w._-]', '', model)
                if org and model:
                    return f"{org}/{model}"
        
        return None
    
    def _extract_model_id_from_text(self, text: str) -> Optional[str]:
        """Try to extract a model ID from free-form text."""
        for pattern in self.HF_MODEL_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def _field_to_relationship(self, field: str) -> str:
        """Convert a config field name to a relationship type."""
        field_mapping = {
            "_name_or_path": "fine_tuned_from",
            "base_model": "based_on",
            "parent_model": "derived_from",
            "pretrained_model_name_or_path": "pretrained_from",
            "model_name_or_path": "based_on",
            "finetuned_from": "fine_tuned_from",
            "base_model_name_or_path": "based_on",
        }
        return field_mapping.get(field, "related_to")
    
    def _get_model_id(self, model_url: ParsedURL) -> Optional[str]:
        """Get the model ID from a parsed URL."""
        if model_url.owner and model_url.repo:
            return f"{model_url.owner}/{model_url.repo}"
        return model_url.repo
    
    def _create_parsed_url(self, model_id: str) -> Optional[ParsedURL]:
        """Create a ParsedURL from a model ID."""
        if "/" not in model_id:
            return None
        
        parts = model_id.split("/")
        if len(parts) < 2:
            return None
        
        owner = parts[0]
        repo = parts[1]
        
        return ParsedURL(
            url=f"https://huggingface.co/{model_id}",
            category=URLCategory.MODEL,
            name=repo,
            platform="huggingface",
            owner=owner,
            repo=repo,
        )
