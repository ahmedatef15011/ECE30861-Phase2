"""Fallback scoring strategies when primary resources are missing.

Provides partial credit for models that don't have linked datasets,
code repositories, or other external resources by analyzing README content.

Philosophy:
- Models shouldn't be penalized to 0 for missing external links
- README content often contains valuable information
- Partial credit encourages incremental improvement
- Fallback scores capped below full resource scores (0.65 max vs 1.0)
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


class ResourceAvailability(Enum):
    """Resource availability levels."""
    FULL = "full"           # Linked resource found and accessible
    PARTIAL = "partial"     # Some info found in README
    MINIMAL = "minimal"     # Only hints/mentions found
    NONE = "none"           # Nothing found


class FallbackScorer:
    """
    Fallback scoring when primary resources are missing.
    
    Use this when:
    - No datasets are linked but README might describe training data
    - No GitHub repo linked but README has code examples
    - No review history available but README shows collaboration
    """

    # Maximum scores for fallback (lower than full resource scores)
    MAX_DATASET_FALLBACK = 0.65
    MAX_CODE_FALLBACK = 0.65
    MAX_REVIEWEDNESS_FALLBACK = 0.60

    def __init__(self, readme_content: Optional[str] = None):
        self.readme_content = readme_content or ""
        self.readme_lower = self.readme_content.lower()

    # =========================================================================
    # DATASET QUALITY FALLBACK
    # =========================================================================
    def get_dataset_fallback_score(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate dataset quality when no linked datasets found.
        
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

        score = 0.0
        details = {
            "method": "readme_fallback",
            "indicators_found": []
        }

        # Check for dataset mentions (0.0 - 0.15)
        dataset_names = self._extract_dataset_mentions()
        if dataset_names:
            score += min(0.15, 0.05 * len(dataset_names))
            details["indicators_found"].append(
                f"dataset_names: {dataset_names[:5]}"
            )

        # Check for training data description (0.0 - 0.15)
        if self._has_training_data_description():
            score += 0.15
            details["indicators_found"].append("training_data_description")

        # Check for data statistics (0.0 - 0.10)
        if self._has_data_statistics():
            score += 0.10
            details["indicators_found"].append("data_statistics")

        # Check for data source/origin (0.0 - 0.10)
        if self._has_data_source():
            score += 0.10
            details["indicators_found"].append("data_source")

        # Check for data preprocessing info (0.0 - 0.10)
        if self._has_preprocessing_info():
            score += 0.10
            details["indicators_found"].append("preprocessing_info")

        # Base credit for having a README with content (0.05)
        if len(self.readme_content) > 100:
            score += 0.05
            details["indicators_found"].append("readme_exists")

        final_score = min(self.MAX_DATASET_FALLBACK, score)
        details["score"] = final_score
        details["max_possible"] = self.MAX_DATASET_FALLBACK
        details["note"] = "Partial credit - no linked datasets found"

        logger.info(
            f"Dataset fallback score: {final_score:.2f} - "
            f"{details['indicators_found']}"
        )
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

        logger.info(
            f"Code fallback score: {final_score:.2f} - "
            f"{details['indicators_found']}"
        )
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
        
        Looks for collaboration/review indicators:
        - Multiple contributors mentioned
        - Acknowledgments
        - Institutional backing
        - Version history
        - Paper citations (implies peer review)
        
        Returns:
            Tuple of (score, details_dict) or (-1.0, details) if truly N/A
        """
        if not self.readme_content:
            return -1.0, {"reason": "No README - metric not applicable"}

        score = 0.0
        details = {
            "method": "readme_fallback",
            "indicators_found": []
        }

        # Check for multiple authors/contributors (0.0 - 0.20)
        if self._has_multiple_contributors():
            score += 0.20
            details["indicators_found"].append("multiple_contributors")

        # Check for institutional backing (0.0 - 0.15)
        if self._has_institutional_backing():
            score += 0.15
            details["indicators_found"].append("institutional_backing")

        # Check for acknowledgments (0.0 - 0.10)
        if self._has_acknowledgments():
            score += 0.10
            details["indicators_found"].append("acknowledgments")

        # Check for version history (0.0 - 0.10)
        if self._has_version_history():
            score += 0.10
            details["indicators_found"].append("version_history")

        # Check for citation/paper (0.0 - 0.15) - implies peer review
        if self._has_paper_citation():
            score += 0.15
            details["indicators_found"].append("paper_citation")

        # If no indicators found at all, return -1 (truly not applicable)
        if score == 0.0:
            return -1.0, {
                "reason": "No review indicators found",
                "note": "Metric not applicable"
            }

        final_score = min(self.MAX_REVIEWEDNESS_FALLBACK, score)
        details["score"] = final_score
        details["max_possible"] = self.MAX_REVIEWEDNESS_FALLBACK
        details["note"] = "Partial credit - inferred from README"

        logger.info(
            f"Reviewedness fallback score: {final_score:.2f} - "
            f"{details['indicators_found']}"
        )
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
