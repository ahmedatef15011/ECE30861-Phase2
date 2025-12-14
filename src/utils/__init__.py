"""Utility modules for the ML Registry."""

# Import utility functions from sibling utils.py module
# Note: Using importlib to avoid package name collision with utils/ directory
import importlib.util
from pathlib import Path

# Load utils.py file directly
_utils_file = Path(__file__).parent.parent / 'utils.py'
_spec = importlib.util.spec_from_file_location("_src_utils", _utils_file)
_utils_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils_module)

# Re-export functions from utils.py
measure_time = _utils_module.measure_time
extract_model_size_from_text = _utils_module.extract_model_size_from_text
parse_license_from_readme = _utils_module.parse_license_from_readme
check_readme_sections = _utils_module.check_readme_sections
extract_performance_claims = _utils_module.extract_performance_claims
execute_access_control_program = _utils_module.execute_access_control_program
validate_sensitive_model_access = _utils_module.validate_sensitive_model_access
execute_with_context = _utils_module.execute_with_context

# Import exceptions from this package
from .exceptions import (
    UserNotFoundError,
    PackageNotFoundError,
    InvalidCredentialsError,
    UnauthorizedError,
)

__all__ = [
    # From utils.py module
    "measure_time",
    "extract_model_size_from_text",
    "parse_license_from_readme",
    "check_readme_sections",
    "extract_performance_claims",
    "execute_access_control_program",
    "validate_sensitive_model_access",
    "execute_with_context",
    # From exceptions.py in this package
    "UserNotFoundError",
    "PackageNotFoundError",
    "InvalidCredentialsError",
    "UnauthorizedError",
]
