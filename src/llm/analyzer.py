"""LLM-based artifact analysis for dependency extraction.

This module uses AWS Bedrock (Claude) to analyze config.json and
extract artifact dependencies beyond what traditional parsing can find.
"""

import json
from typing import Dict, Optional, Any

from .bedrock_client import BedrockClient
from ..logging_utils import get_logger

logger = get_logger()


def analyze_artifact_dependencies(
    config_data: Optional[Dict[str, Any]] = None,
    readme_content: Optional[str] = None,
    model_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use LLM to analyze artifact dependencies from config and README.
    
    This function uses Claude to extract parent model relationships,
    dataset dependencies, and code linkages that may not be captured
    by traditional regex-based parsing.
    
    Args:
        config_data: Parsed config.json data
        readme_content: README.md content
        model_url: URL of the artifact
        
    Returns:
        Dictionary with structure:
        {
            "parent_models": [
                {"id": "owner/model", "relationship": "fine_tuned_from"},
                ...
            ],
            "datasets": [
                {"id": "owner/dataset", "relationship": "trained_on"},
                ...
            ],
            "code_repositories": [
                {"id": "owner/repo", "relationship": "implemented_in"},
                ...
            ],
            "analysis_metadata": {
                "analyzed_at": "ISO timestamp",
                "confidence": "high|medium|low",
                "llm_model": "claude-3-haiku"
            }
        }
    """
    try:
        # Build analysis prompt
        prompt = _build_dependency_analysis_prompt(
            config_data, readme_content, model_url
        )
        
        # Use Claude 3 Haiku for cost-effective analysis
        client = BedrockClient(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            max_tokens=2048,
            temperature=0.1  # Low temperature for factual extraction
        )
        
        system_prompt = """You are an expert at analyzing machine learning \
model metadata and documentation to extract dependency relationships.

Your task is to identify:
1. Parent models (models this was fine-tuned from or based on)
2. Datasets used for training/fine-tuning
3. Code repositories linked to this model

Return ONLY a valid JSON object with this exact structure:
{
  "parent_models": [{"id": "owner/model", "relationship": "type"}],
  "datasets": [{"id": "owner/dataset", "relationship": "type"}],
  "code_repositories": [{"id": "owner/repo", "relationship": "type"}],
  "confidence": "high|medium|low"
}

Relationship types:
- fine_tuned_from, based_on, merged_from (for parent models)
- trained_on, evaluated_on (for datasets)
- implemented_in, uses_code_from (for code repos)

Be conservative - only include dependencies you're confident about.
If uncertain, use "medium" or "low" confidence."""
        
        logger.info("Analyzing artifact dependencies with LLM...")
        response = client.generate(
            prompt=prompt,
            system=system_prompt
        )
        
        # Parse LLM response
        dependencies = _parse_llm_response(response)
        
        # Add metadata
        from datetime import datetime
        dependencies["analysis_metadata"] = {
            "analyzed_at": datetime.utcnow().isoformat(),
            "confidence": dependencies.pop("confidence", "medium"),
            "llm_model": "claude-3-haiku"
        }
        
        logger.info(
            f"LLM dependency analysis complete: "
            f"{len(dependencies.get('parent_models', []))} parents, "
            f"{len(dependencies.get('datasets', []))} datasets, "
            f"{len(dependencies.get('code_repositories', []))} code repos"
        )
        
        return dependencies
        
    except Exception as e:
        logger.warning(f"LLM dependency analysis failed: {e}")
        # Return empty dependencies structure
        from datetime import datetime
        return {
            "parent_models": [],
            "datasets": [],
            "code_repositories": [],
            "analysis_metadata": {
                "analyzed_at": datetime.utcnow().isoformat(),
                "confidence": "none",
                "llm_model": "none",
                "error": str(e)
            }
        }


def _build_dependency_analysis_prompt(
    config_data: Optional[Dict[str, Any]],
    readme_content: Optional[str],
    model_url: Optional[str],
) -> str:
    """Build prompt for LLM dependency analysis."""
    parts = [
        "Analyze the following model metadata to extract dependencies:\n"
    ]
    
    if model_url:
        parts.append(f"**Model URL**: {model_url}\n")
    
    if config_data:
        config_str = json.dumps(config_data, indent=2)
        # Truncate if too long
        if len(config_str) > 8000:
            config_str = config_str[:8000] + "\n... (truncated)"
        parts.append(f"**Config.json**:\n```json\n{config_str}\n```\n")
    
    if readme_content:
        # Truncate README if too long
        readme_truncated = readme_content[:4000]
        if len(readme_content) > 4000:
            readme_truncated += "\n... (truncated)"
        parts.append(f"**README.md**:\n{readme_truncated}\n")
    
    parts.append(
        "\nExtract all dependency relationships and return as JSON."
    )
    
    return "\n".join(parts)


def _parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response to extract dependency JSON."""
    try:
        # Try to find JSON in response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line (```)
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)
        
        # Parse JSON
        data = json.loads(response)
        
        # Validate structure
        if not isinstance(data, dict):
            raise ValueError("Response is not a JSON object")
        
        # Ensure required fields exist
        data.setdefault("parent_models", [])
        data.setdefault("datasets", [])
        data.setdefault("code_repositories", [])
        data.setdefault("confidence", "medium")
        
        return data
        
    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {e}")
        logger.debug(f"Raw response: {response}")
        return {
            "parent_models": [],
            "datasets": [],
            "code_repositories": [],
            "confidence": "none"
        }
