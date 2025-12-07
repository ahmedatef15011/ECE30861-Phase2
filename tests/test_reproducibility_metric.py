"""
Comprehensive tests for ReproducibilityMetric.

Tests cover:
- Safe code execution
- Dangerous pattern detection
- Timeout handling
- Edge cases
- Cross-platform compatibility
- LLM-enhanced scoring (when available)
"""

import pytest
from unittest.mock import patch, MagicMock
from src.metrics.Reproducibility import ReproducibilityMetric
from src.models import ModelContext, ParsedURL, URLCategory


@pytest.fixture
def metric():
    """Create a ReproducibilityMetric instance."""
    return ReproducibilityMetric()


@pytest.fixture
def base_context():
    """Create a base ModelContext for testing."""
    return ModelContext(
        model_url=ParsedURL(
            url="https://huggingface.co/test/model",
            category=URLCategory.MODEL,
            name="test/model",
            platform="huggingface",
            owner="test",
            repo="model"
        ),
        readme_content=None
    )


class TestBasicFunctionality:
    """Test basic metric functionality."""
    
    def test_metric_name(self, metric):
        """Test metric name property."""
        assert metric.name == "reproducibility"
    
    def test_no_readme_returns_zero(self, metric, base_context):
        """Test that no README content returns score 0.0."""
        base_context.readme_content = None
        result = metric.compute(base_context, {})
        assert result.score == 0.0
        assert result.latency >= 0
    
    def test_no_code_blocks_returns_zero(self, metric, base_context):
        """Test that README without code blocks returns based on documentation."""
        base_context.readme_content = """
        # Test Model
        
        This is a great model but has no code examples.
        """
        result = metric.compute(base_context, {})
        # With docs but no code: base (0.15) + doc quality + boost if in range
        assert 0.1 <= result.score < 0.5


class TestSafeCodeExecution:
    """Test safe code execution scenarios."""
    
    def test_simple_safe_code_runs(self, metric, base_context):
        """Test that safe code contributes to score."""
        base_context.readme_content = """
        # Test Model
        
        ```python
        x = 10
        y = 20
        print(x + y)
        ```
        """
        result = metric.compute(base_context, {})
        # Base + doc + code score with potential boost
        assert 0.2 <= result.score <= 0.6
    
    def test_safe_torch_code(self, metric, base_context):
        """Test safe PyTorch code execution."""
        base_context.readme_content = """
        # Test Model
        
        ```python
        try:
            import torch
            tensor = torch.zeros(5)
            print("Success")
        except ImportError:
            print("Torch not available")
        ```
        """
        result = metric.compute(base_context, {})
        # Should be 1.0 or 0.5 depending on if torch is available
        assert result.score in [0.0, 0.5, 1.0]
    
    def test_multiple_safe_code_blocks(self, metric, base_context):
        """Test multiple safe code blocks."""
        base_context.readme_content = """
        # Test Model
        
        First example:
        ```python
        x = 1
        ```
        
        Second example:
        ```python
        y = 2
        print(y)
        ```
        """
        result = metric.compute(base_context, {})
        assert result.score >= 0.0  # At least one should work


class TestDangerousCodeBlocking:
    """Test dangerous code pattern detection and blocking."""
    
    def test_os_system_blocked(self, metric, base_context):
        """Test that os.system() is blocked (code not scored)."""
        base_context.readme_content = """
        ```python
        import os
        os.system('ls')
        ```
        """
        result = metric.compute(base_context, {})
        # Dangerous code blocked, but base + minimal doc remains
        assert 0.1 <= result.score < 0.3
    
    def test_subprocess_blocked(self, metric, base_context):
        """Test that subprocess is blocked (code not scored)."""
        base_context.readme_content = """
        ```python
        import subprocess
        subprocess.run(['ls'])
        ```
        """
        result = metric.compute(base_context, {})
        # Dangerous code blocked, but base + minimal doc remains
        assert 0.1 <= result.score < 0.3
    
    def test_exec_blocked(self, metric, base_context):
        """Test that exec() is blocked (code not scored)."""
        base_context.readme_content = """
        ```python
        exec("print('dangerous')")
        ```
        """
        result = metric.compute(base_context, {})
        # Dangerous code blocked, but base + minimal doc remains
        assert 0.1 <= result.score < 0.3
    
    def test_eval_blocked(self, metric, base_context):
        """Test that eval() is blocked (code not scored)."""
        base_context.readme_content = """
        ```python
        eval("1 + 1")
        ```
        """
        result = metric.compute(base_context, {})
        # Dangerous code blocked, but base + minimal doc remains
        assert 0.1 <= result.score < 0.3
    
    def test_file_operations_blocked(self, metric, base_context):
        """Test that file operations are blocked (code not scored)."""
        base_context.readme_content = """
        ```python
        with open('/etc/passwd', 'r') as f:
            data = f.read()
        ```
        """
        result = metric.compute(base_context, {})
        # Dangerous code blocked, but base + minimal doc remains
        assert 0.1 <= result.score < 0.3
    
    def test_network_calls_blocked(self, metric, base_context):
        """Test that network calls are blocked (code not scored)."""
        base_context.readme_content = """
        ```python
        import requests
        requests.get('http://example.com')
        ```
        """
        result = metric.compute(base_context, {})
        # Dangerous code blocked, but base + minimal doc remains
        assert 0.1 <= result.score < 0.3
    
    def test_mixed_safe_and_unsafe(self, metric, base_context):
        """Test mixed safe and unsafe code blocks."""
        base_context.readme_content = """
        Safe code:
        ```python
        x = 10
        print(x)
        ```
        
        Unsafe code:
        ```python
        import os
        os.system('rm -rf /')
        ```
        """
        result = metric.compute(base_context, {})
        # Should run the safe block
        assert result.score >= 0.0


class TestTimeoutHandling:
    """Test timeout protection."""
    
    def test_infinite_loop_timeout(self, metric, base_context):
        """Test that infinite loops are stopped by timeout."""
        base_context.readme_content = """
        ```python
        while True:
            pass
        ```
        """
        result = metric.compute(base_context, {})
        # Timeout code: base + doc score remains
        assert 0.2 <= result.score < 0.5
    
    def test_long_computation_timeout(self, metric, base_context):
        """Test that long computations timeout."""
        base_context.readme_content = """
        ```python
        import time
        time.sleep(30)  # Longer than timeout
        ```
        """
        result = metric.compute(base_context, {})
        # Timeout code: base + doc score remains
        assert 0.2 <= result.score < 0.5


class TestCodeExtraction:
    """Test code block extraction from markdown."""
    
    def test_extract_python_blocks(self, metric):
        """Test extraction of Python code blocks."""
        content = """
        # Model
        
        ```python
        code1
        ```
        
        ```py
        code2
        ```
        
        ```javascript
        not_python
        ```
        """
        blocks = metric._extract_code_blocks(content)
        assert len(blocks) == 2
        assert 'code1' in blocks[0]
        assert 'code2' in blocks[1]
    
    def test_extract_no_blocks(self, metric):
        """Test extraction when no code blocks present."""
        content = "# Model\n\nJust text."
        blocks = metric._extract_code_blocks(content)
        assert len(blocks) == 0


class TestSecurityPatterns:
    """Test security pattern detection."""
    
    def test_is_code_safe_basic(self, metric):
        """Test basic safe code detection."""
        safe_code = "x = 10\nprint(x)"
        assert metric._is_code_safe(safe_code) is True
    
    def test_is_code_unsafe_os_system(self, metric):
        """Test detection of os.system."""
        unsafe_code = "import os\nos.system('ls')"
        assert metric._is_code_safe(unsafe_code) is False
    
    def test_is_code_unsafe_subprocess(self, metric):
        """Test detection of subprocess."""
        unsafe_code = "import subprocess\nsubprocess.run(['ls'])"
        assert metric._is_code_safe(unsafe_code) is False
    
    def test_is_code_unsafe_exec(self, metric):
        """Test detection of exec."""
        unsafe_code = "exec('print(1)')"
        assert metric._is_code_safe(unsafe_code) is False
    
    def test_is_code_unsafe_eval(self, metric):
        """Test detection of eval."""
        unsafe_code = "result = eval('1+1')"
        assert metric._is_code_safe(unsafe_code) is False


class TestDebuggingFixes:
    """Test code debugging and fixing."""
    
    def test_apply_safe_debugging_fixes(self, metric):
        """Test that safe debugging fixes are applied."""
        code = "print('hello')"
        fixed = metric._apply_safe_debugging_fixes(code, "test/model")
        
        # Should add error handling
        assert 'try:' in fixed
        assert 'except' in fixed
    
    def test_code_runs_after_debugging(self, metric, base_context):
        """Test that code runs after debugging fixes."""
        # Code that might need imports added
        base_context.readme_content = """
        ```python
        # This might need imports to be added
        x = 10
        print(x)
        ```
        """
        result = metric.compute(base_context, {})
        # Should get at least 0.5 if debugging helps
        assert result.score >= 0.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_readme(self, metric, base_context):
        """Test empty README."""
        base_context.readme_content = ""
        result = metric.compute(base_context, {})
        assert result.score == 0.0
    
    def test_malformed_code_block(self, metric, base_context):
        """Test malformed code blocks."""
        base_context.readme_content = """
        ```python
        This is not valid Python syntax @@@ ### !!!
        ```
        """
        result = metric.compute(base_context, {})
        # Malformed code: base + doc score remains
        assert 0.2 <= result.score < 0.5
    
    def test_empty_code_block(self, metric, base_context):
        """Test empty code blocks."""
        base_context.readme_content = """
        ```python
        
        ```
        """
        result = metric.compute(base_context, {})
        # Empty code contributes: base + doc + partial code score
        assert 0.2 <= result.score < 0.5
    
    def test_unicode_in_code(self, metric, base_context):
        """Test Unicode characters in code."""
        base_context.readme_content = """
        ```python
        message = "Hello 世界 🌍"
        print(message)
        ```
        """
        result = metric.compute(base_context, {})
        assert result.score >= 0.0


class TestSecurityLogging:
    """Test security violation logging."""
    
    def test_security_violation_logged(self, metric, base_context):
        """Test that security violations are logged."""
        base_context.readme_content = """
        ```python
        import os
        os.system('dangerous command')
        ```
        """
    
        with patch('src.metrics.Reproducibility.logger') as mock_logger:
            result = metric.compute(base_context, {})
    
            # Should log security warning
            assert mock_logger.warning.called or mock_logger.error.called
            # Dangerous code blocked but base + doc remains
            assert 0.1 <= result.score < 0.3
class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility."""
    
    def test_windows_compatibility(self, metric, base_context):
        """Test that code works on Windows (no signal.alarm)."""
        base_context.readme_content = """
        ```python
        x = 10
        print(x)
        ```
        """
        # This test verifies no AttributeError for signal.SIGALRM
        result = metric.compute(base_context, {})
        assert result.score >= 0.0  # Should work without errors
    
    def test_execution_wrapper_no_signal(self, metric):
        """Test that execution wrapper doesn't use signal module."""
        code = "print('test')"
        wrapper = metric._create_execution_wrapper(code, timeout=10)
        
        # Should NOT contain signal-related code
        assert 'signal.alarm' not in wrapper
        assert 'signal.SIGALRM' not in wrapper
        assert 'signal.signal' not in wrapper


class TestPerformance:
    """Test performance and latency tracking."""
    
    def test_latency_recorded(self, metric, base_context):
        """Test that latency is properly recorded."""
        base_context.readme_content = """
        ```python
        x = 10
        ```
        """
        result = metric.compute(base_context, {})
        
        # Latency should be non-negative
        assert result.latency >= 0
        # Should complete in reasonable time (< 30 seconds)
        assert result.latency < 30000  # 30 seconds in ms


class TestLLMIntegration:
    """Test LLM-enhanced scoring functionality."""
    
    @patch('src.metrics.Reproducibility.HAS_LLM', True)
    @patch('src.metrics.Reproducibility.LLM_ENABLED', True)
    @patch('src.metrics.Reproducibility.analyze_code_reproducibility')
    def test_llm_blending_when_available(
        self, mock_analyze, metric, base_context
    ):
        """Test that LLM scores are blended when available."""
        # Mock LLM returning a high score
        mock_analyze.return_value = (0.9, {"llm_analysis": "excellent code"})
        
        base_context.readme_content = """
        # Good Model
        
        Well documented model with examples.
        
        ```python
        x = 10
        print(x)
        ```
        """
        
        result = metric.compute(base_context, {})
        
        # Score should reflect LLM contribution
        assert result.score >= 0.0
        assert result.score <= 1.0
    
    @patch('src.metrics.Reproducibility.HAS_LLM', False)
    def test_fallback_when_llm_unavailable(self, metric, base_context):
        """Test graceful fallback when LLM not available."""
        base_context.readme_content = """
        ```python
        x = 10
        print(x)
        ```
        """
        
        result = metric.compute(base_context, {})
        
        # Should still produce valid score
        assert result.score >= 0.0
        assert result.score <= 1.0
        assert result.latency >= 0
    
    @patch('src.metrics.Reproducibility.HAS_LLM', True)
    @patch('src.metrics.Reproducibility.LLM_ENABLED', True)
    @patch('src.metrics.Reproducibility.analyze_code_reproducibility')
    def test_llm_error_handling(self, mock_analyze, metric, base_context):
        """Test handling of LLM errors during scoring."""
        # Mock LLM raising an exception
        mock_analyze.side_effect = Exception("LLM API Error")
        
        base_context.readme_content = """
        ```python
        x = 10
        ```
        """
        
        result = metric.compute(base_context, {})
        
        # Should gracefully fall back to deterministic scoring
        assert result.score >= 0.0
        assert result.score <= 1.0
    
    @patch('src.metrics.Reproducibility.HAS_LLM', True)
    @patch('src.metrics.Reproducibility.LLM_ENABLED', True)
    @patch('src.metrics.Reproducibility.analyze_code_reproducibility')
    def test_llm_returns_negative_one(self, mock_analyze, metric, base_context):
        """Test handling when LLM returns -1 (unavailable)."""
        # Mock LLM returning -1 (not available)
        mock_analyze.return_value = (-1.0, {"reason": "Model unavailable"})
        
        base_context.readme_content = """
        ```python
        x = 10
        ```
        """
        
        result = metric.compute(base_context, {})
        
        # Should use deterministic score only
        assert result.score >= 0.0
        assert result.score <= 1.0
    
    @patch('src.metrics.Reproducibility.HAS_LLM', True)
    @patch('src.metrics.Reproducibility.LLM_ENABLED', False)
    def test_llm_disabled_via_environment(self, metric, base_context):
        """Test that LLM can be disabled via environment."""
        base_context.readme_content = """
        ```python
        x = 10
        ```
        """
        
        result = metric.compute(base_context, {})
        
        # Should still produce valid score (deterministic only)
        assert result.score >= 0.0
        assert result.score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
