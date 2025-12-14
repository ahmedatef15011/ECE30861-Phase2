import re
import time
import subprocess
import json
import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@contextmanager
def measure_time():
    # context manager to measure execution time
    start_time = time.perf_counter()
    try:
        #return a lambda so callers can ask for the elapsed at the end
        yield lambda: int((time.perf_counter() - start_time) * 10000)

    finally:
        #nothing to clean up
        pass


def extract_model_size_from_text(text: str) -> Optional[float]:
    # extract model size in GB from text using various patterns.
    if not text:
        return None

    # patterns to match size indicators
    size_patterns = [
        r"(\d+(?:\.\d+)?)\s*([MGT]?B)\b",   # e.g., "7B", "13.5GB", "270M"
        r"(\d+(?:\.\d+)?)\s*billion",       # e.g., "7 billion parameters"
        r"(\d+(?:\.\d+)?)\s*million",       # e.g., "270 million parameters"
        r"(\d+(?:\.\d+)?)\s*([MGT])\b",     # e.g., "270M", "13B" without B suffix
    ]

    text_lower = text.lower()

    for pattern in size_patterns:
        #we pass IGNORECASE, but we already lowercased, so redundancy does not matter
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            try:
                size_str = match.group(1)
                size_value = float(size_str)

#figuring out the unit in billion, millions and bytes
                if len(match.groups()) > 1 and match.group(2):
                    unit = match.group(2).upper()
                else:
                    # check context for unit hints
                    context = text_lower[max(0, match.start() - 20):match.end() + 20]
                    if "billion" in context or "billion" in match.group(0).lower():
                        unit = "B"
                    elif "million" in context or "million" in match.group(0).lower():
                        unit = "M"
                    else:
                        unit = ""

                # convert to GB 
                if unit == "B":  # billion
                    return (
                        size_value * 2.0
                    )  # ~2GB per billion parameters
                elif unit == "M":  # million parameters
                    return size_value * 0.002  # ~2MB per million parameters
                elif unit == "GB":
                    return size_value
                elif unit == "MB":
                    return size_value / 1024.0
                elif unit == "TB":
                    return size_value * 1024.0

            except (ValueError, IndexError):
                #if the match is messy, skipping to the next one
                continue

    return None


def parse_license_from_readme(readme_content: str) -> Optional[str]:
    # extract license information from README content
    if not readme_content:
        return None

    # look for license section
    license_patterns = [
        r"##?\s*License\s*\n\s*(.+?)(?:\n##|\n\n|\Z)",
        r"License:\s*(.+?)(?:\n|\Z)",
        r"\*\*License\*\*:?\s*(.+?)(?:\n|\Z)",
    ]

    for pattern in license_patterns:
        match = re.search(pattern, readme_content, re.IGNORECASE | re.DOTALL)
        if match:
            license_text = match.group(1).strip()
            # clean up common license identifiers
            license_text = re.sub(
                r"\[([^\]]+)\]\([^\)]+\)", r"\1", license_text
            ) 
            return license_text[:200]  # limit length

    return None


def check_readme_sections(
    readme_content: str, required_sections: List[str]
) -> Dict[str, bool]:
    
    # check for presence of required sections in README content    
    if not readme_content:
        return {section: False for section in required_sections}

    readme_lower = readme_content.lower()
    results = {}

    for section in required_sections:
        section_lower = section.lower()
        # look for section headers
        patterns = [
            rf"##?\s*{re.escape(section_lower)}\s*\n",  # markdown header
            rf"\*\*{re.escape(section_lower)}\*\*",     # bold text
            rf"{re.escape(section_lower)}:",            # colon format
        ]

        found = any(re.search(pattern, readme_lower) for pattern in patterns)
        results[section] = found

    # returns - dict mapping section names to boolean presence
    return results

# extract performance claims and benchmark information from README.
def extract_performance_claims(
    readme_content: str, benchmark_keywords: List[str]
) -> Dict[str, Any]:
        
    if not readme_content:
        return {
            "benchmarks_mentioned": [],
            "numeric_results": False,
            "citations": False,
        }

    readme_lower = readme_content.lower()

    # check for benchmark mentions
    benchmarks_found = []
    for benchmark in benchmark_keywords:
        if benchmark.lower() in readme_lower:
            benchmarks_found.append(benchmark)

    # check for numeric results
    numeric_patterns = [
        r"\d+\.\d+%",           # percentage
        r"\d+%",                # percentage
        r"accuracy:\s*\d+",     # accuracy score
        r"f1:\s*\d+\.\d+",      # F1 score
        r"score:\s*\d+\.\d+",   # generic score
    ]

    has_numeric = any(re.search(pattern, readme_lower) for pattern in numeric_patterns)

    # check for citations or links
    citation_patterns = [
        r"\[([^\]]+)\]\([^\)]+\)",  # markdown links
        r"doi:\s*10\.\d+",          # DOI
        r"arxiv:\d+\.\d+",          # ArXiv
        r"https?://[^\s]+",         # general URLs
    ]

    has_citations = any(
        re.search(pattern, readme_lower) for pattern in citation_patterns
    )

    # returns - dict with 'benchmarks_mentioned', 'numeric_results', 'citations'
    return {
        "benchmarks_mentioned": benchmarks_found,
        "numeric_results": has_numeric,
        "citations": has_citations,
    }


# ============================================================================
# JavaScript Access Control Execution
# ============================================================================

def execute_access_control_program(
    javascript_code: str,
    timeout: int = 5
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Execute a JavaScript program for access control validation.
    
    The program must return an exit code of 0 to allow access.
    Any other exit code denies access.
    
    Args:
        javascript_code: The JavaScript program to execute
        timeout: Execution timeout in seconds (default: 5)
        
    Returns:
        Tuple of (access_granted, exit_code, error_message)
        - access_granted: True if exit code is 0, False otherwise
        - exit_code: The actual exit code from the program
        - error_message: Error message if execution failed, None if successful
    """
    try:
        # Run JavaScript using Node.js
        # The program should explicitly exit with a code via process.exit(code)
        result = subprocess.run(
            ["node", "--eval", javascript_code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        exit_code = result.returncode
        access_granted = exit_code == 0
        error_message = None
        
        if result.stderr and exit_code != 0:
            error_message = result.stderr
            logger.warning(f"⚠️  Access control program error: {result.stderr}")
        else:
            logger.info(
                f"🔍 Access control program executed: "
                f"exit_code={exit_code}, access_granted={access_granted}"
            )
        
        return access_granted, exit_code, error_message
        
    except subprocess.TimeoutExpired:
        error_msg = f"Access control program execution timeout ({timeout}s)"
        logger.error(f"❌ {error_msg}")
        return False, None, error_msg
        
    except FileNotFoundError:
        error_msg = "Node.js not installed or not in PATH"
        logger.error(f"❌ {error_msg}")
        return False, None, error_msg
        
    except Exception as e:
        error_msg = f"Access control program execution failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return False, None, error_msg


def validate_sensitive_model_access(
    javascript_code: Optional[str],
    model_name: str,
    user_id: int,
    timeout: int = 5
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Validate access to a sensitive model using its access control program.
    
    A sensitive model can only be accessed if:
    1. An access control program is defined
    2. The program executes successfully
    3. The program returns exit code 0
    
    Args:
        javascript_code: The access control JavaScript program (None if not set)
        model_name: Name of the model being accessed
        user_id: ID of the user requesting access
        timeout: Execution timeout in seconds
        
    Returns:
        Tuple of (access_granted, exit_code, error_message)
    """
    # If no program is defined, deny access (safety default)
    if not javascript_code or not javascript_code.strip():
        error_msg = "No access control program defined for sensitive model"
        logger.warning(f"⚠️  {error_msg}: {model_name}")
        return False, None, error_msg
    
    # Execute the program with context variables
    # Inject model_name and user_id as global variables for the script
    # Convert user_id to string if needed
    user_id_str = str(user_id) if isinstance(user_id, int) else user_id
    enhanced_code = f"""
    // Access control context
    const MODEL_NAME = "{model_name}";
    const USER_ID = "{user_id_str}";
    
    // User-defined access control logic
    {javascript_code}
    """
    
    return execute_access_control_program(enhanced_code, timeout)


def execute_with_context(
    javascript_code: str,
    context: dict,
    timeout: int = 5
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Execute JavaScript with provided context variables.
    
    Args:
        javascript_code: The JavaScript program
        context: Dictionary of variables to inject as globals
        timeout: Execution timeout in seconds
        
    Returns:
        Tuple of (success, exit_code, error_message)
    """
    # Build context initialization code
    context_code = ""
    for key, value in context.items():
        if isinstance(value, str):
            # Escape quotes in string values
            escaped = value.replace('"', '\\"')
            context_code += f'const {key} = "{escaped}";\n'
        elif isinstance(value, (int, float)):
            context_code += f'const {key} = {value};\n'
        elif isinstance(value, bool):
            context_code += f'const {key} = {str(value).lower()};\n'
        else:
            # For complex types, use JSON
            context_code += f'const {key} = {json.dumps(value)};\n'
    
    # Combine context and code
    enhanced_code = context_code + "\n" + javascript_code
    
    return execute_access_control_program(enhanced_code, timeout)
