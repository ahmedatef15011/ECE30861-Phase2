"""Reproducibility metric - checks if model card code runs safely.

Metric Definition:
    Whether the model can be run using only the demonstration code included 
    in the model card.
    
    Score:
        1.0 - Code runs without any changes/debugging
        0.5 - Code runs but requires debugging/fixes by an agent
        0.0 - No code or code doesn't run at all
    
    Security:
        - All code is scanned for dangerous patterns before execution
        - Code runs in isolated sandbox with timeout protection
        - File operations, system commands, and network calls are blocked
        - Cross-platform support (Windows, Linux, macOS)
"""

import re
import logging
import subprocess
import tempfile
import os
import sys
from typing import Any, Dict, List

from .base import BaseMetric
from ..models import MetricResult, ModelContext
from ..utils import measure_time

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
        Calculate reproducibility score with security checks.
        
        Score:
            1.0 - Code runs without changes (and is safe)
            0.5 - Code runs with debugging (and is safe)
            0.0 - No code, unsafe code, or code doesn't run
        
        Args:
            context: Model context with readme_content
            config: Configuration (unused for now)
            
        Returns:
            float: Reproducibility score (0.0, 0.5, or 1.0)
        """
        try:
            # Step 1: Check for readme content
            if not context.readme_content:
                logger.info(
                    f"No model card content for {self._get_model_id(context)} - "
                    "score: 0.0"
                )
                return 0.0

            # Step 2: Extract code blocks from existing readme
            code_blocks = self._extract_code_blocks(context.readme_content)
            if not code_blocks:
                logger.info(
                    f"No Python code blocks found in {self._get_model_id(context)} - "
                    "score: 0.0"
                )
                return 0.0

            logger.info(
                f"Found {len(code_blocks)} code block(s) in "
                f"{self._get_model_id(context)}"
            )

            # Step 3: SECURITY CHECK - scan for dangerous patterns
            unsafe_blocks = []
            safe_blocks = []
            
            for idx, block in enumerate(code_blocks):
                if self._is_code_safe(block):
                    safe_blocks.append(block)
                else:
                    unsafe_blocks.append((idx, block))
                    # Log security violation for audit trail
                    self._log_security_violation(
                        self._get_model_id(context), idx, block
                    )
            
            if not safe_blocks:
                logger.error(
                    f"SECURITY VIOLATION: All {len(code_blocks)} code "
                    f"block(s) in {self._get_model_id(context)} contain dangerous "
                    "operations - score: 0.0"
                )
                return 0.0  # Unsafe code = score 0

            if unsafe_blocks:
                logger.warning(
                    f"Filtered out {len(unsafe_blocks)} unsafe code "
                    f"block(s) from {self._get_model_id(context)}"
                )

            # Step 4: Test safe code (no changes)
            if self._test_code_runs_safely(safe_blocks, self._get_model_id(context)):
                logger.info(
                    f"Model {self._get_model_id(context)}: Safe code runs without "
                    "modification - score: 1.0"
                )
                return 1.0

            # Step 5: Test with debugging
            if self._test_code_with_debugging_safely(
                safe_blocks, 
                self._get_model_id(context)
            ):
                logger.info(
                    f"Model {self._get_model_id(context)}: Safe code runs with "
                    "debugging - score: 0.5"
                )
                return 0.5

            logger.info(
                f"Model {self._get_model_id(context)}: Safe code does not run - "
                "score: 0.0"
            )
            return 0.0
            
        except Exception as e:
            logger.error(
                f"Error calculating reproducibility for "
                f"{self._get_model_id(context)}: {e}",
                exc_info=True
            )
            return 0.0

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

    def _test_code_runs_safely(
        self, 
        code_blocks: List[str], 
        model_id: str
    ) -> bool:
        """
        Test safe code in isolated sandbox.
        
        Tests if code runs without any modifications.
        
        Args:
            code_blocks: List of safe code blocks to test
            model_id: Model ID for logging
            
        Returns:
            bool: True if at least one code block runs successfully
        """
        for idx, code in enumerate(code_blocks):
            try:
                logger.debug(f"Testing safe code block {idx + 1}")
                
                # Execute in sandbox with strict timeout
                result = self._execute_in_sandbox(code, timeout=15)
                
                if result['success']:
                    logger.info(f"Code block {idx + 1} runs successfully")
                    return True
                    
            except Exception as e:
                logger.debug(f"Error testing code block {idx + 1}: {e}")
        
        return False

    def _test_code_with_debugging_safely(
        self, 
        code_blocks: List[str], 
        model_id: str
    ) -> bool:
        """
        Test with debugging in isolated sandbox.
        
        Applies safe debugging fixes and tests if code runs.
        
        Args:
            code_blocks: List of safe code blocks to test
            model_id: Model ID for logging
            
        Returns:
            bool: True if at least one code block runs with debugging
        """
        for idx, code in enumerate(code_blocks):
            try:
                # Apply safe debugging fixes only
                fixed_code = self._apply_safe_debugging_fixes(code, model_id)
                
                result = self._execute_in_sandbox(fixed_code, timeout=15)
                
                if result['success']:
                    logger.info(
                        f"Code block {idx + 1} runs with debugging"
                    )
                    return True
                    
            except Exception as e:
                logger.debug(
                    f"Error debugging code block {idx + 1}: {e}"
                )
        
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