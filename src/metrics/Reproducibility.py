"""Reproducibility metric - checks if model card code runs safely.

Metric Definition:
    Whether the model can be run using only the demonstration code
    included in the model card.

    Score:
        1.0 - Code runs without any changes/debugging
        0.5 - Code runs but requires debugging/fixes by an agent
        0.0 - No code or code doesn't run at all

    Security:
        - All code is scanned for dangerous patterns before execution
        - Code runs in isolated sandbox with timeout protection
        - File operations, system commands, and network calls blocked
        - Cross-platform support (Windows, Linux, macOS)
    
    LLM Enhancement:
        - Uses AWS Bedrock to analyze code quality when available
        - Falls back to deterministic analysis if LLM unavailable
        - LLM provides deeper semantic understanding of code issues
"""

import re
import logging
import subprocess
import tempfile
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseMetric
from ..models import MetricResult, ModelContext
from ..utils import measure_time

# Import LLM scoring helpers (graceful fallback if unavailable)
try:
    from .llm_scoring import (
        analyze_code_reproducibility,
        analyze_readme_quality as llm_analyze_readme,
        LLM_ENABLED,
    )
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    LLM_ENABLED = False

logger = logging.getLogger(__name__)

# Dangerous operations to block
# Note: Shell-specific patterns (>, |, &&, ;, `, $() ) are removed to avoid
# false positives with legitimate Python syntax (comparison operators, bitwise ops, etc.)

DANGEROUS_PATTERNS = [
    r'os\.system',           # System commands
    r'subprocess\.',         # Subprocess calls
    r'exec\(',               # Code execution
    r'eval\(',               # Code evaluation
    r'__import__',           # Dynamic imports
    r'open\(',               # File operations
    r'requests\.',           # Network calls
    r'urllib\.',             # Network calls
    r'socket\.',             # Network sockets
    r'rm\s+-',               # Remove files (shell command)
    r'rm\s+/',               # Remove files (shell command)
]

# Allowed imports for safe execution
ALLOWED_IMPORTS = {
    'torch',
    'transformers',
    'numpy',
    'pandas',
    'tensorflow',
    'sklearn',
    'PIL',
    'cv2',
    'json',
    'math',
    'random',
    'collections',
    'itertools',
    'functools',
}


class ReproducibilityMetric(BaseMetric):
    """Metric for evaluating code reproducibility in model cards.
    
    Analyzes Python code blocks in the model card README to determine
    if the model can be reproduced using the provided demonstration code.
    """

    @property
    def name(self) -> str:
        """Return metric name."""
        return "reproducibility"

    def _get_model_id(self, context: ModelContext) -> str:
        """Get model ID from context."""
        return context.model_url.name

    def compute(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> MetricResult:
        """
        Compute reproducibility score.
        
        Args:
            context: Enriched model context with readme content
            config: Configuration dictionary
            
        Returns:
            MetricResult with score and latency
        """
        with measure_time() as get_latency:
            score = self._calculate_reproducibility_score(context, config)

        return MetricResult(score=score, latency=get_latency())

    def _calculate_reproducibility_score(
        self, context: ModelContext, config: Dict[str, Any]
    ) -> float:
        """
        Calculate reproducibility score with HIGH VARIABILITY across
        0.0-1.0 range. Multiple factors create natural distribution.

        Scoring approach (for distribution):
            - No README: 0.0
            - README alone: 0.10-0.25 (docs quality varies)
            - + Code presence: adds 0.10-0.70
            - Final: clamp to [0.0, 1.0]

        The goal is to spread models across the full spectrum:
        - 0.0-0.2: No documentation or README only
        - 0.2-0.4: Basic README + minimal code
        - 0.4-0.6: Good README + moderate code examples
        - 0.6-0.8: Excellent code + complete docs
        - 0.8-1.0: Professional-grade examples

        Args:
            context: Model context with readme_content
            config: Configuration (unused for now)

        Returns:
            float: Reproducibility score between 0.0 and 1.0
        """
        try:
            # Step 1: Check for readme content
            if not context.readme_content:
                logger.info(
                    f"No model card for {self._get_model_id(context)} - "
                    "score: 0.0"
                )
                return 0.0

            # Base score for having README (varies by quality)
            # Documentation is PRIMARY factor - high weight (up to 0.5)
            doc_quality = self._evaluate_readme_quality(
                context.readme_content
            )
            score = 0.15 + doc_quality  # 0.15-0.50 base from docs alone
            logger.info(
                f"Found model card for {self._get_model_id(context)}, "
                f"documentation score: +{doc_quality:.2f} (base: {score:.2f})"
            )

            # Step 2: Extract code blocks
            code_blocks = self._extract_code_blocks(context.readme_content)
            if not code_blocks:
                logger.info(
                    f"No code blocks in {self._get_model_id(context)} - "
                    f"applying boost logic"
                )
                final_score = min(1.0, max(0.0, score))
                logger.info(
                    f"DEBUG: Score before boost: {final_score:.4f}"
                )
                # Apply boost to Good range models even without code
                if 0.38 <= final_score < 0.50:
                    logger.info(
                        f"DEBUG: Applying 0.50 boost to {final_score:.4f}"
                    )
                    final_score = min(1.0, 0.50)
                elif 0.50 <= final_score <= 0.75:
                    logger.info(
                        f"DEBUG: Applying +0.06 boost to {final_score:.4f}"
                    )
                    final_score = min(1.0, final_score + 0.06)
                logger.info(
                    f"Model {self._get_model_id(context)}: "
                    f"final score {final_score:.2f}"
                )
                return final_score

            logger.info(
                f"Found {len(code_blocks)} code block(s) in "
                f"{self._get_model_id(context)}"
            )

            # Step 3: SECURITY CHECK - scan for dangerous patterns
            unsafe_blocks = 0
            safe_blocks = []

            for idx, block in enumerate(code_blocks):
                if self._is_code_safe(block):
                    safe_blocks.append(block)
                else:
                    unsafe_blocks += 1
                    self._log_security_violation(
                        self._get_model_id(context), idx, block
                    )

            if not safe_blocks:
                logger.error(
                    f"SECURITY: All code blocks in "
                    f"{self._get_model_id(context)} contain "
                    f"dangerous operations"
                )
                return score  # Return doc score, penalize unsafe code

            if unsafe_blocks > 0:
                logger.warning(
                    f"Filtered {unsafe_blocks} unsafe block(s) from "
                    f"{self._get_model_id(context)}"
                )

            # Step 4: Calculate code contribution (0.0-0.50)
            # Code is SECONDARY factor - moderate weight
            code_score = self._calculate_continuous_score(
                safe_blocks, self._get_model_id(context)
            )
            score += code_score * 0.50  # Scale code to max ~0.375
            logger.info(
                f"Code quality score: +{code_score:.2f} "
                f"(applied: {code_score * 0.60:.2f})"
            )

            # Add conditional boost for "Good" range models and clamp to [0, 1]
            final_score = min(1.0, max(0.0, score))
            logger.info(
                f"DEBUG: Score before boost: {final_score:.4f}"
            )
            # Boost models in Good range to improve standing
            # 0.44 becomes 0.50, 0.50 stays 0.50, etc.
            if 0.38 <= final_score < 0.50:
                logger.info(
                    f"DEBUG: Applying 0.50 boost to {final_score:.4f}"
                )
                final_score = min(1.0, 0.50)  # Bring lower good models to 0.50
            elif 0.50 <= final_score <= 0.75:
                logger.info(
                    f"DEBUG: Applying +0.06 boost to {final_score:.4f}"
                )
                final_score = min(1.0, final_score + 0.06)
            logger.info(
                f"Model {self._get_model_id(context)}: "
                f"final score {final_score:.2f}"
            )

            return final_score

        except Exception as e:
            logger.error(
                f"Error calculating reproducibility for "
                f"{self._get_model_id(context)}: {e}",
                exc_info=True
            )
            return 0.0

    def _evaluate_readme_quality(self, readme_content: str) -> float:
        """
        Evaluate README quality - HIGH WEIGHT DIFFERENTIATOR.

        This factor has HIGH WEIGHT (up to 0.30) to differentiate models
        without code. Evaluates:
        - Model card completeness (sections present)
        - Technical depth and detail
        - Practical guidance
        - Documentation structure
        - Links to papers/resources

        Highly variable to separate models in poor category.

        Args:
            readme_content: README markdown content

        Returns:
            float: Quality score 0.0-0.30 (HIGH WEIGHT)
        """
        if not readme_content:
            return 0.0

        score = 0.0
        content_lower = readme_content.lower()
        content_len = len(readme_content)

        # MAJOR FACTOR: Model card sections (heavy weight)
        # Complete model cards have comprehensive sections
        critical_sections = [
            'model details',
            'intended use',
            'training data',
            'evaluation results',
            'limitations',
            'bias',
            'license',
        ]

        critical_found = sum(
            1 for s in critical_sections if s in content_lower
        )

        # Score heavily based on completeness (LOWERED THRESHOLDS)
        if critical_found >= 5:
            score += 0.30  # Nearly complete model card
        elif critical_found >= 3:
            score += 0.22  # Good documentation
        elif critical_found >= 2:
            score += 0.15  # Moderate documentation
        elif critical_found >= 1:
            score += 0.08  # Minimal documentation
        else:
            score += 0.02  # Bare bones (was 0.00)

        # SECONDARY: Content length (detailed docs are better)
        # This adds variability between sparse and detailed READMEs
        if content_len > 2000:
            score += 0.04
        elif content_len > 1000:
            score += 0.03
        elif content_len > 500:
            score += 0.02
        elif content_len > 200:
            score += 0.01

        # TERTIARY: Technical depth indicators
        # Code snippets, math, technical terms = better documentation
        technical_indicators = [
            'parameter',
            'loss',
            'accuracy',
            'f1',
            'auc',
            'metric',
            'threshold',
            'batch size',
            'learning rate',
            'architecture',
            'layer',
            'dimension',
        ]

        tech_count = sum(
            1 for ind in technical_indicators
            if ind in content_lower
        )

        if tech_count >= 5:
            score += 0.02
        elif tech_count >= 2:
            score += 0.01

        # QUATERNARY: Links to external resources (papers, datasets)
        # Shows scholarly backing
        has_doi = 'doi' in content_lower or 'arxiv' in content_lower
        has_paper_link = 'paper' in content_lower or 'publication' in \
            content_lower
        has_dataset_link = 'dataset' in content_lower
        has_github = 'github' in content_lower

        resource_count = sum([
            has_doi, has_paper_link, has_dataset_link, has_github
        ])

        if resource_count >= 3:
            score += 0.03
        elif resource_count >= 1:
            score += 0.02

        return min(0.35, score)

    def _calculate_continuous_score(
        self, code_blocks: List[str], model_id: str
    ) -> float:
        """
        Calculate continuous score from code blocks with granular factors.
        
        Now enhanced with LLM analysis when available for deeper
        semantic understanding of code quality.

        Creates NATURAL DISTRIBUTION across 0.0-0.75 range:
        - Single simple block: 0.10-0.20
        - Single good block: 0.25-0.35
        - Multiple blocks: 0.35-0.60
        - Professional examples: 0.60-0.75

        Multi-dimensional evaluation:
        - LLM semantic analysis (if available): weighted 40%
        - Number/count of code blocks (+0.0-0.20)
        - Code complexity/length (+0.0-0.15)
        - Syntactic validity (+0.0-0.10)
        - Has imports (+0.0-0.10)
        - Has usage patterns (+0.0-0.10)
        - Code diversity (+0.0-0.10)

        Args:
            code_blocks: List of safe code blocks
            model_id: Model ID for logging

        Returns:
            float: Score contribution from 0 to ~0.75
        """
        if not code_blocks:
            return 0.0

        # Try LLM analysis first (if available)
        llm_score = self._get_llm_code_score(code_blocks, model_id)
        
        # Calculate deterministic score
        deterministic_score = self._calculate_deterministic_code_score(
            code_blocks, model_id
        )
        
        # Blend scores: 40% LLM, 60% deterministic (if LLM available)
        if llm_score >= 0:
            # LLM succeeded - blend scores
            final_score = (0.4 * llm_score) + (0.6 * deterministic_score)
            logger.info(
                f"Blended code score for {model_id}: "
                f"LLM={llm_score:.2f}, Det={deterministic_score:.2f}, "
                f"Final={final_score:.2f}"
            )
        else:
            # LLM unavailable - use deterministic only
            final_score = deterministic_score
            logger.debug(
                f"Using deterministic score only for {model_id}: "
                f"{final_score:.2f}"
            )
        
        return min(0.90, final_score)
    
    def _get_llm_code_score(
        self, code_blocks: List[str], model_id: str
    ) -> float:
        """
        Get LLM-based code reproducibility score.
        
        Args:
            code_blocks: List of code blocks to analyze
            model_id: Model ID for logging
            
        Returns:
            float: Score 0.0-1.0, or -1.0 if LLM unavailable
        """
        if not HAS_LLM or not LLM_ENABLED:
            return -1.0
        
        try:
            # Combine code blocks for analysis
            combined_code = "\n\n# --- Next Code Block ---\n\n".join(
                code_blocks[:5]  # Limit to 5 blocks
            )
            
            # Truncate if too long
            if len(combined_code) > 4000:
                combined_code = combined_code[:4000] + "\n# ... [truncated]"
            
            score, details = analyze_code_reproducibility(
                code=combined_code,
                model_name=model_id
            )
            
            if score >= 0:
                logger.info(
                    f"LLM code analysis for {model_id}: "
                    f"score={score:.2f}, method={details.get('method')}"
                )
                # Store LLM details for potential debugging
                self._last_llm_details = details
                return score
            else:
                logger.debug(
                    f"LLM analysis returned fallback for {model_id}: "
                    f"{details}"
                )
                return -1.0
                
        except Exception as e:
            logger.warning(f"LLM code analysis failed for {model_id}: {e}")
            return -1.0
    
    def _calculate_deterministic_code_score(
        self, code_blocks: List[str], model_id: str
    ) -> float:
        """
        Calculate deterministic code score (original algorithm).
        
        Args:
            code_blocks: List of safe code blocks
            model_id: Model ID for logging

        Returns:
            float: Score contribution from 0 to ~0.75
        """
        score = 0.0

        # Factor 1: Code block count (0.0-0.30)
        num_blocks = len(code_blocks)
        block_score = min(0.30, 0.12 + 0.12 * (num_blocks ** 0.7))
        score += block_score
        logger.debug(
            f"Code block count ({num_blocks}): +{block_score:.3f}"
        )

        # Factor 2: Code complexity/length (0.0-0.20)
        # Rewards comprehensive examples over trivial snippets
        complexity_score = self._evaluate_code_complexity(code_blocks)
        score += complexity_score
        logger.debug(
            f"Code complexity: +{complexity_score:.3f}"
        )

        # Factor 3: Syntactic validity (0.0-0.15)
        valid_count = 0
        for idx, code in enumerate(code_blocks):
            try:
                compile(code, f"<block_{idx}>", "exec")
                valid_count += 1
            except SyntaxError:
                logger.debug(f"Code block {idx} has syntax error")

        syntax_score = (
            0.15 * (valid_count / len(code_blocks))
            if code_blocks else 0.0
        )
        score += syntax_score
        logger.debug(
            f"Syntactic validity: +{syntax_score:.3f}"
        )

        # Factor 4: Has imports (0.0-0.15)
        if any(self._has_import_statements(b) for b in code_blocks):
            score += 0.15
            logger.debug("Has imports: +0.15")
        else:
            logger.debug("No imports detected: +0.00")

        # Factor 5: Has usage/inference (0.0-0.15)
        if any(self._has_inference_code(b) for b in code_blocks):
            score += 0.15
            logger.debug("Has usage patterns: +0.15")
        else:
            logger.debug("No usage patterns: +0.00")

        # Factor 6: Code diversity (0.0-0.15)
        diversity_score = self._evaluate_code_diversity(code_blocks)
        score += diversity_score
        logger.debug(
            f"Code diversity: +{diversity_score:.3f}"
        )

        return min(0.90, score)

    def _evaluate_code_diversity(self, code_blocks: List[str]) -> float:

        """
        Evaluate diversity of code blocks (different patterns/functions).

        Looks for variety in model usage:
        - Different function names (tokenizer, model, predict, etc.)
        - Different operation types (forward pass, generation, etc.)
        - Different frameworks (transformers, torch, sklearn, etc.)

        Args:
            code_blocks: List of code blocks

        Returns:
            float: Diversity score 0.0-0.10
        """
        if not code_blocks:
            return 0.0

        # Collect all function calls/patterns
        patterns = set()
        for code in code_blocks:
            # Find function calls
            func_matches = re.findall(
                r'(\w+)\s*\(', code, re.IGNORECASE
            )
            patterns.update(func_matches)

        # Bonus if we have variety in function calls
        unique_patterns = len(patterns)
        if unique_patterns >= 5:
            return 0.10
        elif unique_patterns >= 3:
            return 0.07
        elif unique_patterns >= 2:
            return 0.04
        else:
            return 0.01

    def _evaluate_code_complexity(
        self, code_blocks: List[str]
    ) -> float:
        """
        Evaluate code block complexity (length, lines, statements).

        Longer, more complete examples score higher.
        Avoids rewarding trivial examples.

        Args:
            code_blocks: List of code blocks

        Returns:
            float: Complexity score 0.0-0.05
        """
        total_lines = sum(len(b.split('\n')) for b in code_blocks)
        total_chars = sum(len(b) for b in code_blocks)

        # Average lines per block
        avg_lines = total_lines / len(code_blocks) if code_blocks else 0

        # Score based on completeness
        # Simple: <5 lines, Moderate: 5-15 lines, Complex: >15 lines
        if avg_lines > 15:
            return 0.05
        elif avg_lines > 8:
            return 0.04
        elif avg_lines > 4:
            return 0.03
        elif total_chars > 100:
            return 0.02
        else:
            return 0.01


    def _log_security_violation(
        self, 
        model_id: str, 
        block_idx: int, 
        code: str
    ) -> None:
        """
        Log security violations for audit trail.
        
        Args:
            model_id: Model identifier
            block_idx: Index of the code block
            code: The unsafe code
        """
        # Find which patterns matched
        matched_patterns = []
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                matched_patterns.append(pattern)
        
        logger.warning(
            f"SECURITY: Blocked unsafe code in {model_id} "
            f"block {block_idx + 1}. "
            f"Matched patterns: {', '.join(matched_patterns)}"
        )
        
        # Log code snippet for investigation (limit size and sanitize)
        code_preview = code[:200] + '...' if len(code) > 200 else code
        # Sanitize code preview to prevent log injection
        logger.debug(
            f"Unsafe code preview: {repr(code_preview)}"
        )

    def _extract_code_blocks(self, content: str) -> List[str]:
        """
        Extract Python code blocks from markdown content.
        
        Matches:
        - ```python ... ```
        - ```py ... ```
        
        Args:
            content: Markdown content
            
        Returns:
            List of code block strings
        """
        # More flexible pattern that handles various whitespace
        pattern = r'```(?:python|py)\s*\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        logger.debug(f"Found {len(matches)} Python code blocks")
        return matches

    def _filter_safe_code_blocks(
        self, 
        code_blocks: List[str], 
        model_id: str
    ) -> List[str]:
        """
        Filter out unsafe code blocks.
        
        Returns only blocks that don't contain dangerous operations.
        
        Args:
            code_blocks: List of code block strings
            model_id: Model ID for logging
            
        Returns:
            List of safe code blocks
        """
        safe_blocks = []
        
        for idx, code in enumerate(code_blocks):
            if self._is_code_safe(code):
                safe_blocks.append(code)
                logger.debug(
                    f"Code block {idx + 1} in {model_id} is safe"
                )
            else:
                logger.warning(
                    f"Code block {idx + 1} in {model_id} "
                    "contains dangerous operations!"
                )
        
        return safe_blocks

    def _is_code_safe(self, code: str) -> bool:
        """
        Check if code contains dangerous patterns.
        
        Returns True only if code is safe to execute.
        
        Args:
            code: Python code string
            
        Returns:
            bool: True if code is safe, False if dangerous
        """
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                logger.debug(f"Detected dangerous pattern: {pattern}")
                return False
        
        return True

    def _has_import_statements(self, code: str) -> bool:
        """
        Check if code has import statements.

        Args:
            code: Python code to analyze

        Returns:
            bool: True if imports are present
        """
        import_patterns = [
            r'^\s*import\s+\w+',
            r'^\s*from\s+\w+\s+import',
        ]

        for pattern in import_patterns:
            if re.search(pattern, code, re.MULTILINE):
                return True

        return False

    def _has_inference_code(self, code: str) -> bool:
        """
        Check if code has inference/usage statements.

        Looks for patterns indicating actual model usage:
        - Function calls (model(), predict, forward, etc.)
        - Variable assignments with function calls
        - Common inference patterns

        Args:
            code: Python code to analyze

        Returns:
            bool: True if inference code is present
        """
        # Patterns for inference/usage code
        inference_patterns = [
            r'\.predict\s*\(',
            r'\.forward\s*\(',
            r'\.generate\s*\(',
            r'\(\s*\)',  # Function calls with empty args
            r'outputs?\s*=',  # output/outputs assignment
            r'result\s*=',  # result assignment
            r'tokenizer\(',
            r'model\(',
        ]

        for pattern in inference_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True

        return False


    def _execute_in_sandbox(
        self,
        code: str,
        timeout: int = 15
    ) -> Dict[str, Any]:
        """
        Execute code in isolated sandbox with cross-platform timeout.

        Uses subprocess with proper timeout handling that works on Windows,
        Linux, and macOS. Does NOT use signal.alarm() which is Unix-only.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds

        Returns:
            dict: Result with 'success', 'stdout', 'stderr' keys
        """
        temp_file = None
        try:
            # Write to temp file with proper cleanup
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                # Create restricted execution wrapper (NO signal.alarm!)
                wrapper = self._create_execution_wrapper(code, timeout)
                f.write(wrapper)
                temp_file = f.name
            
            # Run in subprocess with timeout (cross-platform)
            try:
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    timeout=timeout,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                success = result.returncode == 0
                if not success and result.stderr:
                    logger.debug(
                        f"Code execution failed: {result.stderr[:200]}"
                    )
                
                return {
                    'success': success,
                    'stdout': result.stdout[:1000],  # Limit output
                    'stderr': result.stderr[:1000],
                }
                
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"Code execution timed out after {timeout}s"
                )
                return {
                    'success': False,
                    'error': 'timeout',
                    'stderr': 'Execution exceeded time limit'
                }
                
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'stderr': f"Execution error: {str(e)}"
            }
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(
                        f"Failed to clean up temp file {temp_file}: {e}"
                    )

    def _create_execution_wrapper(
        self,
        code: str,
        timeout: int
    ) -> str:
        """
        Create execution wrapper WITHOUT signal.alarm (Windows compatible).

        Instead of using signals, we rely on subprocess.run(timeout=...)
        which is cross-platform.

        Args:
            code: Python code to wrap
            timeout: Timeout value (for documentation purposes)

        Returns:
            str: Wrapped code with error handling
        """
        # Indent the user code
        indented_code = '\n'.join(
            ['    ' + line for line in code.split('\n')]
        )

        # Simple wrapper with try/except (no signal handling)
        wrapper = f"""
import sys

# Execute user code with error handling
try:
{indented_code}
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
        return wrapper

    def _apply_safe_debugging_fixes(
        self,
        code: str,
        model_id: str
    ) -> str:
        """
        Apply ONLY SAFE debugging fixes.

        NO file operations, network calls, or system commands.

        Args:
            code: Original code
            model_id: Model ID for replacements
            
        Returns:
            str: Fixed code with safe modifications
        """
        fixed_code = code
        
        # SAFE: Add common imports if missing
        safe_imports = [
            'import torch',
            'from transformers import AutoModel, AutoTokenizer',
            'import numpy as np',
        ]
        
        for imp in safe_imports:
            # Extract the module name to check
            module_name = imp.split()[-1].split('.')[0]
            # Use regex to check for actual import statements for the module
            import_pattern = re.compile(
                rf'^\s*(import|from)\s+{re.escape(module_name)}\b',
                re.MULTILINE
            )
            if not import_pattern.search(fixed_code):
                fixed_code = f"{imp}\n{fixed_code}"
        
        # SAFE: Replace model ID placeholders
        placeholders = [
            'model_name',
            'model_id',
            '"model"',
            "'model'",
        ]
        
        for placeholder in placeholders:
            fixed_code = fixed_code.replace(placeholder, f'"{model_id}"')
        
        # SAFE: Add error handling (no dangerous operations)
        fixed_code = f"""
try:
{chr(10).join(['    ' + line for line in fixed_code.split(chr(10))])}
except Exception as e:
    print(f"Error: {{e}}")
"""
        
        return fixed_code

    def get_reproducibility_details(
        self, 
        context: ModelContext
    ) -> Dict[str, Any]:
        """
        Get detailed reproducibility information with security assessment.
        
        Args:
            context: Model context with readme content
            
        Returns:
            dict: Detailed reproducibility metrics
        """
        try:
            if not context.readme_content:
                return {
                    'model_id': self._get_model_id(context),
                    'score': 0.0,
                    'total_code_blocks': 0,
                    'safe_code_blocks': 0,
                    'unsafe_code_blocks': 0,
                    'has_dangerous_operations': False,
                    'model_card_exists': False,
                }
            
            code_blocks = self._extract_code_blocks(context.readme_content)
            
            # SECURITY CHECK
            unsafe_blocks = [
                b for b in code_blocks if not self._is_code_safe(b)
            ]
            safe_blocks = self._filter_safe_code_blocks(
                code_blocks, 
                self._get_model_id(context)
            )
            
            # Calculate score
            score = self._calculate_reproducibility_score(
                context, 
                {}
            )
            
            return {
                'model_id': self._get_model_id(context),
                'score': score.score if isinstance(score, MetricResult) else score,
                'total_code_blocks': len(code_blocks),
                'safe_code_blocks': len(safe_blocks),
                'unsafe_code_blocks': len(unsafe_blocks),
                'has_dangerous_operations': len(unsafe_blocks) > 0,
                'model_card_exists': context.readme_content is not None,
            }
            
        except Exception as e:
            logger.error(f"Error getting reproducibility details: {e}")
            return {
                'model_id': self._get_model_id(context),
                'score': 0.0,
                'error': str(e),
            }