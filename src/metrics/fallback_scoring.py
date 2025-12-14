"""Fallback scoring strategies when primary resources are missing.

Provides partial credit for models that don't have linked datasets,
code repositories, or other external resources by analyzing README content.

Philosophy:
- Models shouldn't be penalized to 0 for missing external links
- README content often contains valuable information
- Partial credit encourages incremental improvement
- Fallback scores capped below full resource scores (0.65 max vs 1.0)

LLM Enhancement:
- Uses AWS Bedrock for semantic analysis when available
- Provides deeper understanding of README content
- Falls back gracefully to deterministic scoring if LLM unavailable
"""

import re
import logging
import os
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)

# Import LLM scoring helpers (graceful fallback if unavailable)
try:
    from .llm_scoring import (
        analyze_dataset_from_readme,
        analyze_code_from_readme,
        analyze_reviewedness_from_readme,
        LLM_ENABLED,
    )
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    LLM_ENABLED = False
    logger.debug("LLM scoring module not available - using deterministic only")


class ResourceAvailability(Enum):
    """Resource availability levels."""
    FULL = "full"           # Linked resource found and accessible
    PARTIAL = "partial"     # Some info found in README
    MINIMAL = "minimal"     # Only hints/mentions found
    NONE = "none"           # Nothing found


class FallbackScorer:
    """
    Fallback scoring when primary resources are missing.
    
    Now enhanced with LLM analysis for deeper semantic understanding.
    
    Use this when:
    - No datasets are linked but README might describe training data
    - No GitHub repo linked but README has code examples
    - No review history available but README shows collaboration
    """

    # Maximum scores for fallback (lower than full resource scores)
    MAX_DATASET_FALLBACK = 0.80
    MAX_CODE_FALLBACK = 0.65
    # Reviewedness cap - allows well-documented models to reach high scores
    MAX_REVIEWEDNESS_FALLBACK = 0.90
    
    # LLM weight in blended scoring (40% LLM, 60% deterministic)
    LLM_WEIGHT = 0.4
    DETERMINISTIC_WEIGHT = 0.6

    def __init__(
        self,
        readme_content: Optional[str] = None,
        model_name: str = "unknown",
        use_llm: bool = True
    ):
        """
        Initialize fallback scorer.
        
        Args:
            readme_content: README/model card content
            model_name: Name of the model (for logging)
            use_llm: Whether to use LLM analysis (default: True)
        """
        self.readme_content = readme_content or ""
        self.readme_lower = self.readme_content.lower()
        self.model_name = model_name
        self.use_llm = use_llm and HAS_LLM and LLM_ENABLED
        
        # Store LLM analysis results for debugging
        self._llm_results: Dict[str, Any] = {}

    # =========================================================================
    # DATASET QUALITY FALLBACK
    # =========================================================================

    def get_dataset_fallback_score(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate dataset quality when no linked datasets found.
        
        Now enhanced with LLM analysis for deeper semantic understanding.
        
        Looks for dataset information in README:
        - Dataset names/references
        - Training data descriptions
        - Data statistics
        - Data sources
        
        Returns:
            Tuple of (score, details_dict)
        """
        if not self.readme_content:
            return 0.0, {"reason": "No README content"}

        # Try LLM analysis first
        llm_score = -1.0
        llm_details = {}
        if self.use_llm:
            llm_score, llm_details = self._get_llm_dataset_score()
            self._llm_results["dataset"] = llm_details
        
        # Calculate deterministic score
        det_score, det_details = self._get_deterministic_dataset_score()
        
        # Blend scores if LLM available
        if llm_score >= 0:
            # Take max between LLM and deterministic scores (more generous)
            final_score = max(llm_score, det_score)
            final_score = min(self.MAX_DATASET_FALLBACK, final_score)
            
            details = {
                "method": "llm_enhanced_fallback",
                "llm_score": llm_score,
                "deterministic_score": det_score,
                "final_score": final_score,
                "llm_details": llm_details,
                "deterministic_details": det_details,
            }
            logger.info(
                f"Dataset fallback (LLM-enhanced) for {self.model_name}: "
                f"LLM={llm_score:.2f}, Det={det_score:.2f}, "
                f"Max={final_score:.2f}"
            )
        else:
            final_score = det_score
            details = det_details
            logger.info(
                f"Dataset fallback (deterministic) for {self.model_name}: "
                f"{final_score:.2f}"
            )
        
        return final_score, details
    
    def _get_llm_dataset_score(self) -> Tuple[float, Dict[str, Any]]:
        """Get LLM-based dataset quality score from README analysis."""
        try:
            return analyze_dataset_from_readme(
                readme_content=self.readme_content,
                model_name=self.model_name
            )
        except Exception as e:
            logger.warning(f"LLM dataset analysis failed: {e}")
            return -1.0, {"error": str(e)}
    
    def _get_deterministic_dataset_score(self) -> Tuple[float, Dict[str, Any]]:
        """Calculate deterministic dataset fallback score."""
        score = 0.0
        details = {
            "method": "readme_fallback",
            "indicators_found": []
        }

        # Check for dataset mentions (0.0 - 0.20)
        dataset_names = self._extract_dataset_mentions()
        if dataset_names:
            score += min(0.20, 0.07 * len(dataset_names))
            details["indicators_found"].append(
                f"dataset_names: {dataset_names[:5]}"
            )

        # Check for training data description (0.0 - 0.20)
        if self._has_training_data_description():
            score += 0.20
            details["indicators_found"].append("training_data_description")

        # Check for data statistics (0.0 - 0.15)
        if self._has_data_statistics():
            score += 0.15
            details["indicators_found"].append("data_statistics")

        # Check for data source/origin (0.0 - 0.15)
        if self._has_data_source():
            score += 0.15
            details["indicators_found"].append("data_source")

        # Check for data preprocessing info (0.0 - 0.15)
        if self._has_preprocessing_info():
            score += 0.15
            details["indicators_found"].append("preprocessing_info")

        # Base credit for having a README with content (0.10)
        if len(self.readme_content) > 100:
            score += 0.10
            details["indicators_found"].append("readme_exists")

        # Baseline for HuggingFace models (implies some data quality)
        if score == 0.0 and len(self.readme_content) > 50:
            score = 0.20
            details["indicators_found"].append("baseline_credit")

        final_score = min(self.MAX_DATASET_FALLBACK, score)
        details["score"] = final_score
        details["max_possible"] = self.MAX_DATASET_FALLBACK
        details["note"] = "Partial credit - no linked datasets found"

        return final_score, details


    def _extract_dataset_mentions(self) -> List[str]:
        """Extract mentioned dataset names from README."""
        # Common dataset name patterns
        patterns = [
            r'\b(imagenet|cifar|mnist|coco|squad|glue|superglue)\b',
            r'\b(wikipedia|bookcorpus|common\s*crawl|openwebtext)\b',
            r'\b(wikitext|ptb|penn\s*treebank)\b',
            r'\b(laion|cc3m|cc12m|yfcc|flickr)\b',
            r'\b(webtext|pile|redpajama|refinedweb)\b',
            r'\bdataset[:\s]+["\']?([a-zA-Z0-9_-]+)["\']?',
            r'trained\s+on\s+(?:the\s+)?([a-zA-Z0-9_-]+)',
            r'fine-?tuned\s+on\s+(?:the\s+)?([a-zA-Z0-9_-]+)',
        ]
        
        mentions = set()
        for pattern in patterns:
            matches = re.findall(pattern, self.readme_lower, re.IGNORECASE)
            for m in matches:
                if isinstance(m, str) and m and len(m) > 2:
                    mentions.add(m.lower())
        
        return list(mentions)

    def _has_training_data_description(self) -> bool:
        """Check if README describes training data."""
        indicators = [
            'training data',
            'trained on',
            'training corpus',
            'training set',
            'fine-tuned on',
            'pretraining data',
            'pre-training data',
            'finetuning data',
        ]
        return any(ind in self.readme_lower for ind in indicators)

    def _has_data_statistics(self) -> bool:
        """Check if README contains data statistics."""
        # Look for numbers with data-related context
        patterns = [
            r'\d+\s*(million|billion|k|m|b)\s*(tokens?|samples?|examples?)',
            r'\d+\s*(million|billion|k|m|b)\s*(images?|documents?|sentences?)',
            r'\d+\s*(gb|mb|tb)\s*(of\s+)?(data|text|images?)',
            r'(samples?|examples?|instances?):\s*\d+',
            r'(images?|documents?):\s*\d+',
            r'\d+[kmb]\s+training',
        ]
        return any(re.search(p, self.readme_lower) for p in patterns)

    def _has_data_source(self) -> bool:
        """Check if README mentions data source/origin."""
        indicators = [
            'collected from',
            'sourced from',
            'scraped from',
            'downloaded from',
            'obtained from',
            'data source',
            'original data',
            'curated from',
            'aggregated from',
        ]
        return any(ind in self.readme_lower for ind in indicators)

    def _has_preprocessing_info(self) -> bool:
        """Check if README describes data preprocessing."""
        indicators = [
            'preprocessing',
            'pre-processing',
            'tokenization',
            'tokenized',
            'cleaned',
            'filtered',
            'normalized',
            'augmentation',
            'data processing',
            'data cleaning',
        ]
        return any(ind in self.readme_lower for ind in indicators)

    # =========================================================================
    # CODE QUALITY FALLBACK
    # =========================================================================
    def get_code_fallback_score(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate code quality when no linked GitHub repo found.
        
        Now enhanced with LLM analysis for deeper understanding.
        
        Looks for code-related information in README:
        - Code snippets
        - Installation instructions
        - Usage examples
        - Architecture descriptions
        
        Returns:
            Tuple of (score, details_dict)
        """
        if not self.readme_content:
            return 0.3, {"reason": "No README content", "default": True}

        # Try LLM analysis first
        llm_score = -1.0
        llm_details = {}
        if self.use_llm:
            llm_score, llm_details = self._get_llm_code_score()
            self._llm_results["code"] = llm_details
        
        # Calculate deterministic score
        det_score, det_details = self._get_deterministic_code_score()
        
        # Blend scores if LLM available
        if llm_score >= 0:
            final_score = (
                self.LLM_WEIGHT * llm_score +
                self.DETERMINISTIC_WEIGHT * det_score
            )
            final_score = min(self.MAX_CODE_FALLBACK, final_score)
            
            details = {
                "method": "llm_enhanced_fallback",
                "llm_score": llm_score,
                "deterministic_score": det_score,
                "blended_score": final_score,
                "llm_details": llm_details,
                "deterministic_details": det_details,
            }
            logger.info(
                f"Code fallback (LLM-enhanced) for {self.model_name}: "
                f"LLM={llm_score:.2f}, Det={det_score:.2f}, "
                f"Final={final_score:.2f}"
            )
        else:
            final_score = det_score
            details = det_details
            logger.info(
                f"Code fallback (deterministic) for {self.model_name}: "
                f"{final_score:.2f}"
            )
        
        return final_score, details
    
    def _get_llm_code_score(self) -> Tuple[float, Dict[str, Any]]:
        """Get LLM-based code quality score from README analysis."""
        try:
            return analyze_code_from_readme(
                readme_content=self.readme_content,
                model_name=self.model_name
            )
        except Exception as e:
            logger.warning(f"LLM code analysis failed: {e}")
            return -1.0, {"error": str(e)}
    
    def _get_deterministic_code_score(self) -> Tuple[float, Dict[str, Any]]:
        """Calculate deterministic code fallback score."""
        score = 0.0
        details = {
            "method": "readme_fallback",
            "indicators_found": []
        }

        # Check for code snippets (0.0 - 0.20)
        code_blocks = self._count_code_blocks()
        if code_blocks > 0:
            score += min(0.20, 0.05 * code_blocks)
            details["indicators_found"].append(f"code_blocks: {code_blocks}")

        # Check for installation instructions (0.0 - 0.10)
        if self._has_installation_instructions():
            score += 0.10
            details["indicators_found"].append("installation_instructions")

        # Check for usage examples (0.0 - 0.15)
        if self._has_usage_examples():
            score += 0.15
            details["indicators_found"].append("usage_examples")

        # Check for architecture description (0.0 - 0.10)
        if self._has_architecture_description():
            score += 0.10
            details["indicators_found"].append("architecture_description")

        # Check for API documentation (0.0 - 0.10)
        if self._has_api_documentation():
            score += 0.10
            details["indicators_found"].append("api_documentation")

        # If nothing found, return baseline
        if score == 0.0:
            return 0.3, {
                "reason": "No code indicators in README",
                "default": True
            }

        final_score = min(self.MAX_CODE_FALLBACK, score)
        details["score"] = final_score
        details["max_possible"] = self.MAX_CODE_FALLBACK
        details["note"] = "Partial credit - no linked code repository found"

        return final_score, details

    def _count_code_blocks(self) -> int:
        """Count code blocks in README."""
        pattern = r'```(?:python|py|bash|sh|javascript|js|java|cpp)?\s*\n'
        matches = re.findall(pattern, self.readme_content, re.IGNORECASE)
        return len(matches)

    def _has_installation_instructions(self) -> bool:
        """Check for installation instructions."""
        indicators = [
            'pip install',
            'installation',
            'requirements',
            'dependencies',
            'setup.py',
            'conda install',
            'npm install',
            'how to install',
        ]
        return any(ind in self.readme_lower for ind in indicators)

    def _has_usage_examples(self) -> bool:
        """Check for usage examples."""
        indicators = [
            'usage',
            'example',
            'how to use',
            'quick start',
            'getting started',
            'tutorial',
            'quickstart',
        ]
        return any(ind in self.readme_lower for ind in indicators)

    def _has_architecture_description(self) -> bool:
        """Check for architecture description."""
        indicators = [
            'architecture',
            'model structure',
            'layers',
            'transformer',
            'encoder',
            'decoder',
            'attention',
            'hidden size',
            'num_layers',
            'parameters',
        ]
        return any(ind in self.readme_lower for ind in indicators)

    def _has_api_documentation(self) -> bool:
        """Check for API documentation."""
        indicators = [
            'api',
            'parameters',
            'arguments',
            'returns',
            'inputs',
            'outputs',
            'config',
            'forward(',
            'generate(',
        ]
        return any(ind in self.readme_lower for ind in indicators)

    # =========================================================================
    # REVIEWEDNESS FALLBACK
    # =========================================================================
    def get_reviewedness_fallback_score(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate reviewedness when no linked GitHub repo found.
        
        Now enhanced with LLM analysis for deeper understanding.
        
        Looks for collaboration/review indicators:
        - Multiple contributors mentioned
        - Acknowledgments
        - Institutional backing
        - Version history
        - Paper citations (implies peer review)
        
        Returns:
            Tuple of (score, details_dict)
        """
        # Give baseline credit for HuggingFace-hosted models without README
        # HuggingFace has some level of content moderation
        if not self.readme_content:
            baseline = 0.25
            return baseline, {
                "method": "huggingface_baseline",
                "reason": "No README available",
                "note": "Baseline credit for HuggingFace-hosted model",
                "score": baseline
            }

        # Try LLM analysis first
        llm_score = -1.0
        llm_details = {}
        if self.use_llm:
            llm_score, llm_details = self._get_llm_reviewedness_score()
            self._llm_results["reviewedness"] = llm_details
        
        # Calculate deterministic score
        det_score, det_details = self._get_deterministic_reviewedness_score()
        
        # If deterministic returns -1 (N/A), check if LLM can provide score
        if det_score < 0:
            if llm_score >= 0:
                # LLM found something deterministic missed
                final_score = min(self.MAX_REVIEWEDNESS_FALLBACK, llm_score)
                details = {
                    "method": "llm_only_fallback",
                    "llm_score": llm_score,
                    "llm_details": llm_details,
                    "note": "LLM found indicators deterministic analysis missed"
                }
                logger.info(
                    f"Reviewedness fallback (LLM-only) for {self.model_name}: "
                    f"{final_score:.2f}"
                )
                return final_score, details
            else:
                return det_score, det_details
        
        # Blend scores if LLM available
        if llm_score >= 0:
            final_score = (
                self.LLM_WEIGHT * llm_score +
                self.DETERMINISTIC_WEIGHT * det_score
            )
            final_score = min(self.MAX_REVIEWEDNESS_FALLBACK, final_score)
            
            details = {
                "method": "llm_enhanced_fallback",
                "llm_score": llm_score,
                "deterministic_score": det_score,
                "blended_score": final_score,
                "llm_details": llm_details,
                "deterministic_details": det_details,
            }
            logger.info(
                f"Reviewedness fallback (LLM-enhanced) for {self.model_name}: "
                f"LLM={llm_score:.2f}, Det={det_score:.2f}, "
                f"Final={final_score:.2f}"
            )
        else:
            final_score = det_score
            details = det_details
            logger.info(
                f"Reviewedness fallback (deterministic) for {self.model_name}: "
                f"{final_score:.2f}"
            )
        
        return final_score, details
    
    def _get_llm_reviewedness_score(self) -> Tuple[float, Dict[str, Any]]:
        """Get LLM-based reviewedness score from README analysis."""
        try:
            return analyze_reviewedness_from_readme(
                readme_content=self.readme_content,
                model_name=self.model_name
            )
        except Exception as e:
            logger.warning(f"LLM reviewedness analysis failed: {e}")
            return -1.0, {"error": str(e)}
    
    def _get_deterministic_reviewedness_score(self) -> Tuple[float, Dict[str, Any]]:
        """Calculate deterministic reviewedness fallback score."""
        score = 0.0
        details = {
            "method": "readme_fallback",
            "indicators_found": []
        }

        # Check for multiple authors/contributors (0.0 - 0.25)
        if self._has_multiple_contributors():
            score += 0.25
            details["indicators_found"].append("multiple_contributors")

        # Check for institutional backing (0.0 - 0.25) - strong indicator
        if self._has_institutional_backing():
            score += 0.25
            details["indicators_found"].append("institutional_backing")

        # Check for acknowledgments (0.0 - 0.15)
        if self._has_acknowledgments():
            score += 0.15
            details["indicators_found"].append("acknowledgments")

        # Check for version history (0.0 - 0.15)
        if self._has_version_history():
            score += 0.15
            details["indicators_found"].append("version_history")

        # Check for citation/paper (0.0 - 0.20) - implies peer review
        if self._has_paper_citation():
            score += 0.20
            details["indicators_found"].append("paper_citation")

        # Check for contribution guidelines (0.0 - 0.15)
        if self._has_contribution_guidelines():
            score += 0.15
            details["indicators_found"].append("contribution_guidelines")

        # Check for CI/CD badges (0.0 - 0.15)
        if self._has_ci_cd_badges():
            score += 0.15
            details["indicators_found"].append("ci_cd_badges")

        # Check for code review references (0.0 - 0.15)
        if self._has_code_review_references():
            score += 0.15
            details["indicators_found"].append("code_review_references")

        # Check for model card completeness (0.0 - 0.20)
        if self._has_complete_model_card():
            score += 0.20
            details["indicators_found"].append("complete_model_card")

        # Check for usage examples (0.0 - 0.15)
        if self._has_usage_examples():
            score += 0.15
            details["indicators_found"].append("usage_examples")

        # If no indicators found, give small baseline instead of -1.0
        # Models shouldn't be completely penalized for missing metadata
        if score == 0.0:
            baseline = 0.20
            return baseline, {
                "method": "baseline_fallback",
                "reason": "No review indicators found in README",
                "note": "Baseline credit for HuggingFace-hosted model",
                "score": baseline
            }

        final_score = min(self.MAX_REVIEWEDNESS_FALLBACK, score)
        details["score"] = final_score
        details["max_possible"] = self.MAX_REVIEWEDNESS_FALLBACK
        details["note"] = "Partial credit - inferred from README"

        return final_score, details

    def _has_multiple_contributors(self) -> bool:
        """Check for multiple contributors."""
        patterns = [
            r'contributors?',
            r'authors?:\s*\w+.*,.*\w+',  # Multiple names
            r'\bteam\b',
            r'developed by.*and',
            r'created by.*and',
            r'maintainers?',
        ]
        return any(re.search(p, self.readme_lower) for p in patterns)

    def _has_institutional_backing(self) -> bool:
        """Check for institutional backing (implies review process)."""
        institutions = [
            'google', 'microsoft', 'meta', 'facebook', 'openai',
            'anthropic', 'huggingface', 'nvidia', 'amazon', 'apple',
            'deepmind', 'stability', 'mistral', 'cohere', 'ai21',
            'university', 'research lab', 'institute', 'laboratory',
            'mit', 'stanford', 'berkeley', 'cmu', 'eth',
        ]
        return any(inst in self.readme_lower for inst in institutions)

    def _has_acknowledgments(self) -> bool:
        """Check for acknowledgments section."""
        indicators = [
            'acknowledgment',
            'acknowledgement',
            'thanks to',
            'we thank',
            'supported by',
            'funded by',
            'grant',
        ]
        return any(ind in self.readme_lower for ind in indicators)

    def _has_version_history(self) -> bool:
        """Check for version history (implies iterative review)."""
        indicators = [
            'changelog',
            'version history',
            'release notes',
            'what\'s new',
            'updates:',
        ]
        # Also check for version patterns
        version_pattern = r'\bv\d+\.\d+(\.\d+)?\b'
        has_versions = bool(re.search(version_pattern, self.readme_lower))
        
        return has_versions or any(ind in self.readme_lower for ind in indicators)

    def _has_paper_citation(self) -> bool:
        """Check for paper citation (implies peer review process)."""
        indicators = [
            'arxiv',
            'paper',
            'publication',
            'cite',
            'bibtex',
            '@article',
            '@inproceedings',
            '@misc',
            'doi:',
            'proceedings',
            'conference',
            'journal',
        ]
        return any(ind in self.readme_lower for ind in indicators)

    def _has_contribution_guidelines(self) -> bool:
        """Check for contribution guidelines (implies code review process)."""
        indicators = [
            'contributing',
            'contribution guideline',
            'pull request',
            'code of conduct',
            'codeowners',
            'how to contribute',
            'development guide',
            'pr template',
            'issue template',
        ]
        return any(ind in self.readme_lower for ind in indicators)

    def _has_ci_cd_badges(self) -> bool:
        """Check for CI/CD badges (implies automated review/testing)."""
        indicators = [
            'github.com/.*/(workflows|actions)',  # GitHub Actions
            'travis-ci',
            'circleci',
            'jenkins',
            'azure-pipelines',
            'codecov',
            'coveralls',
            'shields.io',
            'badge',
            'build passing',
            'build: passing',
            'tests passing',
            'ci:',
        ]
        # For regex patterns
        for ind in indicators:
            if '.*' in ind or '/' in ind:
                if re.search(ind, self.readme_lower):
                    return True
            elif ind in self.readme_lower:
                return True
        return False

    def _has_code_review_references(self) -> bool:
        """Check for explicit code review references."""
        indicators = [
            'code review',
            'reviewed by',
            'approved by',
            'review process',
            'peer review',
            'lgtm',
            'sign-off',
            'maintainer',
            'core team',
        ]
        return any(ind in self.readme_lower for ind in indicators)

    def _has_complete_model_card(self) -> bool:
        """Check for complete model card (implies professional process)."""
        # Count key sections that indicate a complete model card
        sections = [
            'model description',
            'intended use',
            'limitations',
            'training data',
            'evaluation',
            'how to use',
            'license',
            '## ',  # Markdown headers indicate structure
        ]
        found = sum(1 for s in sections if s in self.readme_lower)
        # Consider complete if 3+ sections found
        return found >= 3

    # =========================================================================
    # LLM RESULTS ACCESS
    # =========================================================================
    def get_llm_results(self) -> Dict[str, Any]:
        """
        Get all LLM analysis results collected during scoring.
        
        Returns:
            Dict with LLM results for each metric type (dataset, code, reviewedness)
        """
        return {
            "llm_enabled": self.use_llm,
            "model_name": self.model_name,
            "results": self._llm_results.copy()
        }
    
    def is_llm_available(self) -> bool:
        """Check if LLM is available for scoring."""
        return HAS_LLM and LLM_ENABLED
