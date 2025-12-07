"""LLM-based metric helpers using AWS Bedrock.

Provides LLM-enhanced analysis for quality metrics:
- Code reproducibility analysis
- README quality assessment
- Dataset quality inference from descriptions
- Code quality evaluation

These functions gracefully fall back to deterministic scoring
if Bedrock is unavailable or disabled.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Check if LLM is enabled (default to True on AWS, False locally)
LLM_ENABLED = os.getenv("BEDROCK_ENABLED", "true").lower() == "true"


def get_llm_client():
    """Get Bedrock client with error handling.
    
    Returns:
        BedrockClient or None if unavailable
    """
    if not LLM_ENABLED:
        logger.debug("LLM disabled via BEDROCK_ENABLED=false")
        return None
    
    try:
        from src.llm import get_bedrock_client
        return get_bedrock_client()
    except ImportError as e:
        logger.warning(f"LLM module not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize LLM client: {e}")
        return None


def analyze_code_reproducibility(
    code: str,
    model_name: str = "unknown"
) -> Tuple[float, Dict[str, Any]]:
    """
    Use LLM to analyze code reproducibility.
    
    Evaluates:
    - Will the code run without modifications?
    - Are all dependencies clearly specified?
    - Is the code complete (no missing pieces)?
    - Are there obvious bugs or issues?
    
    Args:
        code: Python code to analyze
        model_name: Model name for logging
        
    Returns:
        Tuple of (score 0.0-1.0, details dict)
    """
    client = get_llm_client()
    if not client:
        return -1.0, {"method": "llm_unavailable", "fallback": True}
    
    try:
        prompt = f"""Analyze this Python code from an ML model card for reproducibility.

Code:
```python
{code[:4000]}  # Truncated if too long
```

Evaluate:
1. Will this code run as-is without modifications? (0-10)
2. Are all required imports present? (0-10)
3. Is the code complete (no placeholder comments like "...")?  (0-10)
4. Are there obvious bugs or syntax errors? (0-10, where 10=no bugs)
5. Does it show a complete usage example? (0-10)

Return a JSON object:
{{
    "runs_without_changes": <0-10>,
    "imports_complete": <0-10>,
    "code_complete": <0-10>,
    "no_bugs": <0-10>,
    "complete_example": <0-10>,
    "overall_score": <0.0-1.0>,
    "issues": ["list of issues found"],
    "verdict": "runs_perfectly|needs_minor_fixes|needs_major_fixes|broken"
}}"""

        system = """You are an expert Python developer evaluating ML model code examples.
Be strict but fair. Common issues: missing imports, incomplete examples, placeholder code.
Return ONLY valid JSON, no other text."""

        response = client.generate(
            prompt=prompt,
            system=system,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Parse JSON response
        result = _parse_json_response(response)
        
        if result and "overall_score" in result:
            score = float(result["overall_score"])
            # Map verdict to score adjustment
            verdict = result.get("verdict", "needs_minor_fixes")
            if verdict == "runs_perfectly":
                score = max(score, 0.9)
            elif verdict == "needs_minor_fixes":
                score = min(max(score, 0.5), 0.8)
            elif verdict == "needs_major_fixes":
                score = min(score, 0.5)
            elif verdict == "broken":
                score = min(score, 0.2)
            
            return score, {
                "method": "llm_analysis",
                "model": client.model_id,
                "details": result
            }
        else:
            return -1.0, {"method": "llm_parse_error", "raw": response[:500]}
            
    except Exception as e:
        logger.error(f"LLM code analysis failed for {model_name}: {e}")
        return -1.0, {"method": "llm_error", "error": str(e)}


def analyze_readme_quality(
    readme_content: str,
    model_name: str = "unknown"
) -> Tuple[float, Dict[str, Any]]:
    """
    Use LLM to analyze README/model card quality.
    
    Evaluates:
    - Completeness of documentation
    - Clarity of usage instructions
    - Technical depth
    - Presence of important sections
    
    Args:
        readme_content: README markdown content
        model_name: Model name for logging
        
    Returns:
        Tuple of (score 0.0-1.0, details dict)
    """
    client = get_llm_client()
    if not client:
        return -1.0, {"method": "llm_unavailable", "fallback": True}
    
    try:
        # Truncate to fit context
        content = readme_content[:6000]
        if len(readme_content) > 6000:
            content += "\n...[truncated]"
        
        prompt = f"""Analyze this ML model README/model card for quality and completeness.

README:
{content}

Evaluate these aspects (0-10 each):
1. Usage instructions clarity
2. Technical depth (architecture, parameters, etc.)
3. Training information (data, process, hyperparameters)
4. Evaluation results (metrics, benchmarks)
5. Limitations and bias disclosure
6. Code examples quality
7. Overall documentation completeness

Return a JSON object:
{{
    "usage_clarity": <0-10>,
    "technical_depth": <0-10>,
    "training_info": <0-10>,
    "evaluation_results": <0-10>,
    "limitations_disclosed": <0-10>,
    "code_examples": <0-10>,
    "completeness": <0-10>,
    "overall_score": <0.0-1.0>,
    "missing_sections": ["list of recommended sections to add"],
    "strengths": ["list of documentation strengths"]
}}"""

        system = """You are an ML documentation expert evaluating model cards.
Good model cards have: clear usage, technical details, training info, 
benchmarks, and honest limitation disclosure.
Return ONLY valid JSON, no other text."""

        response = client.generate(
            prompt=prompt,
            system=system,
            temperature=0.1,
            max_tokens=1000
        )
        
        result = _parse_json_response(response)
        
        if result and "overall_score" in result:
            score = float(result["overall_score"])
            return score, {
                "method": "llm_analysis",
                "model": client.model_id,
                "details": result
            }
        else:
            return -1.0, {"method": "llm_parse_error", "raw": response[:500]}
            
    except Exception as e:
        logger.error(f"LLM README analysis failed for {model_name}: {e}")
        return -1.0, {"method": "llm_error", "error": str(e)}


def analyze_dataset_from_readme(
    readme_content: str,
    model_name: str = "unknown"
) -> Tuple[float, Dict[str, Any]]:
    """
    Use LLM to infer dataset quality from README when no dataset is linked.
    
    Looks for and evaluates:
    - Dataset descriptions
    - Training data mentions
    - Data statistics
    - Data quality indicators
    
    Args:
        readme_content: README markdown content
        model_name: Model name for logging
        
    Returns:
        Tuple of (score 0.0-0.65, details dict)
        Score capped at 0.65 since actual dataset not verified
    """
    client = get_llm_client()
    if not client:
        return -1.0, {"method": "llm_unavailable", "fallback": True}
    
    try:
        content = readme_content[:5000]
        
        prompt = f"""Analyze this ML model README to evaluate the quality of dataset information provided.

README:
{content}

Look for:
1. What training data was used?
2. Dataset size and statistics?
3. Data sources mentioned?
4. Data preprocessing described?
5. Data quality considerations?

Return a JSON object:
{{
    "datasets_mentioned": ["list of dataset names found"],
    "data_size_mentioned": true/false,
    "data_sources_clear": true/false,
    "preprocessing_described": true/false,
    "data_quality_discussed": true/false,
    "training_data_score": <0-10>,
    "documentation_score": <0-10>,
    "overall_score": <0.0-1.0>,
    "summary": "Brief summary of training data described"
}}

If no dataset information found, overall_score should be 0.0-0.2."""

        system = """You are an ML data expert evaluating training data documentation.
Well-documented training data includes: dataset names, sizes, sources, 
preprocessing steps, and quality considerations.
Return ONLY valid JSON, no other text."""

        response = client.generate(
            prompt=prompt,
            system=system,
            temperature=0.1,
            max_tokens=800
        )
        
        result = _parse_json_response(response)
        
        if result and "overall_score" in result:
            # Cap at 0.65 since no actual dataset verified
            score = min(0.65, float(result["overall_score"]))
            return score, {
                "method": "llm_fallback_analysis",
                "model": client.model_id,
                "capped_at": 0.65,
                "details": result
            }
        else:
            return -1.0, {"method": "llm_parse_error", "raw": response[:500]}
            
    except Exception as e:
        logger.error(f"LLM dataset analysis failed for {model_name}: {e}")
        return -1.0, {"method": "llm_error", "error": str(e)}


def analyze_code_from_readme(
    readme_content: str,
    model_name: str = "unknown"
) -> Tuple[float, Dict[str, Any]]:
    """
    Use LLM to evaluate code quality from README when no repo is linked.
    
    Evaluates:
    - Code examples present and quality
    - Installation instructions
    - API documentation
    - Architecture descriptions
    
    Args:
        readme_content: README markdown content
        model_name: Model name for logging
        
    Returns:
        Tuple of (score 0.0-0.65, details dict)
        Score capped at 0.65 since actual repo not verified
    """
    client = get_llm_client()
    if not client:
        return -1.0, {"method": "llm_unavailable", "fallback": True}
    
    try:
        content = readme_content[:5000]
        
        prompt = f"""Analyze this ML model README to evaluate code-related documentation quality.

README:
{content}

Evaluate:
1. Are there code examples? How complete are they?
2. Installation instructions present and clear?
3. API/usage documentation quality?
4. Architecture/technical description quality?
5. Would a developer be able to use this model easily?

Return a JSON object:
{{
    "code_examples_count": <number>,
    "code_examples_quality": <0-10>,
    "installation_instructions": true/false,
    "installation_clarity": <0-10>,
    "api_documented": true/false,
    "architecture_described": true/false,
    "developer_friendly": <0-10>,
    "overall_score": <0.0-1.0>,
    "missing": ["list of missing code-related documentation"]
}}"""

        system = """You are a software engineer evaluating ML model documentation.
Good documentation has: working code examples, clear install instructions,
API docs, and architecture explanations.
Return ONLY valid JSON, no other text."""

        response = client.generate(
            prompt=prompt,
            system=system,
            temperature=0.1,
            max_tokens=800
        )
        
        result = _parse_json_response(response)
        
        if result and "overall_score" in result:
            # Cap at 0.65 since no actual repo verified
            score = min(0.65, float(result["overall_score"]))
            return score, {
                "method": "llm_fallback_analysis",
                "model": client.model_id,
                "capped_at": 0.65,
                "details": result
            }
        else:
            return -1.0, {"method": "llm_parse_error", "raw": response[:500]}
            
    except Exception as e:
        logger.error(f"LLM code analysis failed for {model_name}: {e}")
        return -1.0, {"method": "llm_error", "error": str(e)}


def analyze_reviewedness_from_readme(
    readme_content: str,
    model_name: str = "unknown"
) -> Tuple[float, Dict[str, Any]]:
    """
    Use LLM to infer review quality when no GitHub repo is available.
    
    Looks for indicators of review/collaboration:
    - Multiple contributors
    - Institutional backing
    - Peer review (papers)
    - Version history
    - Acknowledgments
    
    Args:
        readme_content: README markdown content
        model_name: Model name for logging
        
    Returns:
        Tuple of (score 0.0-0.60 or -1.0, details dict)
        Returns -1.0 if metric is truly not applicable
    """
    client = get_llm_client()
    if not client:
        return -1.0, {"method": "llm_unavailable", "fallback": True}
    
    try:
        content = readme_content[:4000]
        
        prompt = f"""Analyze this ML model README for indicators of review and collaboration.

README:
{content}

Look for evidence of:
1. Multiple contributors/authors
2. Institutional backing (companies, universities)
3. Peer review (associated papers, citations)
4. Version history (changelogs, updates)
5. Community involvement (acknowledgments, thanks)

Return a JSON object:
{{
    "contributors_found": ["list of contributor/author names if any"],
    "multiple_contributors": true/false,
    "institutions_mentioned": ["list of institutions"],
    "has_institutional_backing": true/false,
    "paper_citations": true/false,
    "peer_reviewed": true/false,
    "version_history_present": true/false,
    "acknowledgments_present": true/false,
    "collaboration_score": <0-10>,
    "overall_score": <0.0-1.0>,
    "evidence_summary": "Brief summary of review/collaboration evidence"
}}

If no evidence of review/collaboration found at all, set overall_score to -1."""

        system = """You are evaluating ML model documentation for signs of review and collaboration.
Well-reviewed models typically have: multiple authors, institutional backing,
associated papers, version history, and acknowledgments.
Return ONLY valid JSON, no other text."""

        response = client.generate(
            prompt=prompt,
            system=system,
            temperature=0.1,
            max_tokens=800
        )
        
        result = _parse_json_response(response)
        
        if result and "overall_score" in result:
            raw_score = float(result["overall_score"])
            
            # Return -1 if truly not applicable
            if raw_score < 0:
                return -1.0, {
                    "method": "llm_analysis",
                    "not_applicable": True,
                    "details": result
                }
            
            # Cap at 0.60 since no actual PR history verified
            score = min(0.60, raw_score)
            return score, {
                "method": "llm_fallback_analysis",
                "model": client.model_id,
                "capped_at": 0.60,
                "details": result
            }
        else:
            return -1.0, {"method": "llm_parse_error", "raw": response[:500]}
            
    except Exception as e:
        logger.error(f"LLM reviewedness analysis failed for {model_name}: {e}")
        return -1.0, {"method": "llm_error", "error": str(e)}


def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown formatting."""
    try:
        # Try direct parse first
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code blocks
    try:
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0]
            return json.loads(json_str.strip())
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0]
            return json.loads(json_str.strip())
    except (json.JSONDecodeError, IndexError):
        pass
    
    # Try finding JSON object pattern
    try:
        import re
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, AttributeError):
        pass
    
    logger.warning(f"Failed to parse JSON from LLM response: {response[:200]}")
    return None
