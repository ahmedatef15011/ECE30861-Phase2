"""Dependency injection functions for FastAPI endpoints."""

from typing import Generator, Optional
import re
import signal
import threading
from contextlib import contextmanager
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from src.database.connection import SessionLocal
from src.database import crud
from src.database.models import User
from src.auth.jwt_handler import verify_token
from src.utils.exceptions import UnauthorizedError


# ============================================================================
# REGEX SAFETY UTILITIES
# ============================================================================

# Maximum length for regex patterns (prevents memory exhaustion)
MAX_REGEX_LENGTH = 256

# Maximum time for regex execution (seconds)
REGEX_TIMEOUT = 1.0

# Dangerous regex patterns that can cause ReDoS
REDOS_PATTERNS = [
    r'\(\.\*\)\+',           # (.*)+
    r'\(\.\+\)\+',           # (.+)+
    r'\([^)]*\+[^)]*\)\+',   # (a+)+ style patterns
    r'\([^)]*\*[^)]*\)\+',   # (a*)+
    r'\([^)]*\+[^)]*\)\*',   # (a+)*
    r'\([^)]*\|[^)]*\)\+',   # (a|a)+ style
    r'(\.\*){2,}',           # .*.* or more
    r'(\.\+){2,}',           # .+.+ or more
    r'\(\?[^)]*\)\+',        # non-capturing groups with +
]


class RegexTimeoutError(Exception):
    """Exception raised when regex execution times out."""
    pass


def _is_potentially_dangerous_regex(pattern: str) -> bool:
    """
    Check if a regex pattern is potentially dangerous (ReDoS vulnerable).
    
    Args:
        pattern: The regex pattern to check
        
    Returns:
        True if the pattern is potentially dangerous
    """
    # Check for common ReDoS patterns
    for redos_pattern in REDOS_PATTERNS:
        if re.search(redos_pattern, pattern):
            return True
    
    # Check for nested quantifiers like (a+)+, (a*)+, etc.
    # This is a simplified check for nested repetition
    nested_quantifier = r'\([^)]*[\+\*][^)]*\)[\+\*]'
    if re.search(nested_quantifier, pattern):
        return True
    
    # Check for overlapping alternations like (a|a|aa)+
    # Count characters that could cause exponential backtracking
    quantifier_count = len(re.findall(r'[\+\*\?]', pattern))
    group_count = len(re.findall(r'\(', pattern))
    
    # If there are many quantifiers in groups, it's suspicious
    if quantifier_count > 3 and group_count > 2:
        return True
    
    return False


def validate_regex_pattern(pattern: str) -> str:
    """
    Validate and sanitize a regex pattern for safe execution.
    
    This function checks for:
    1. Pattern length limits
    2. Valid regex syntax
    3. ReDoS vulnerability patterns
    
    Args:
        pattern: The regex pattern to validate
        
    Returns:
        The validated pattern (possibly sanitized)
        
    Raises:
        HTTPException: 400 if the pattern is invalid or dangerous
    """
    # Check length
    if len(pattern) > MAX_REGEX_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Regex pattern too long (max {MAX_REGEX_LENGTH} characters)"
        )
    
    # Handle special case of "*" (list all)
    if pattern == "*":
        return pattern
    
    # Check for empty pattern
    if not pattern or pattern.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty regex pattern"
        )
    
    # Try to compile the regex to check for syntax errors
    try:
        re.compile(pattern)
    except re.error as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid regex pattern: {str(e)}"
        )
    
    # Check for dangerous patterns that could cause ReDoS
    if _is_potentially_dangerous_regex(pattern):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Regex pattern contains potentially dangerous constructs"
        )
    
    return pattern


def safe_regex_match(pattern: str, text: str, timeout: float = REGEX_TIMEOUT) -> bool:
    """
    Safely execute a regex match with timeout protection.
    
    Args:
        pattern: The regex pattern
        text: The text to match against
        timeout: Maximum execution time in seconds
        
    Returns:
        True if pattern matches text, False otherwise
        
    Raises:
        RegexTimeoutError: If the regex execution times out
    """
    result = [False]
    exception = [None]
    
    def match_with_timeout():
        try:
            result[0] = bool(re.search(pattern, text, re.IGNORECASE))
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=match_with_timeout)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # Thread is still running - regex took too long
        raise RegexTimeoutError(f"Regex execution timed out after {timeout}s")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]


def sanitize_name_query(name: str) -> str:
    """
    Sanitize a name query for safe use in database operations.
    
    For exact matches, returns the name as-is.
    For patterns (containing regex metacharacters), validates as regex.
    
    Args:
        name: The name or pattern to query
        
    Returns:
        Sanitized name/pattern safe for database queries
        
    Raises:
        HTTPException: 400 if the pattern is invalid
    """
    # Special case for "list all"
    if name == "*":
        return name
    
    # Check if name contains regex metacharacters
    regex_metacharacters = set('[](){}|^$.*+?\\')
    has_regex = any(c in regex_metacharacters for c in name)
    
    if has_regex:
        # Validate as regex pattern
        return validate_regex_pattern(name)
    else:
        # Plain text name - just validate length
        if len(name) > MAX_REGEX_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Name too long (max {MAX_REGEX_LENGTH} characters)"
            )
        return name


def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency.
    
    Yields:
        SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# HTTP Bearer token security scheme
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Get the current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer credentials containing JWT token
        db: Database session
        
    Returns:
        Authenticated User object
        
    Raises:
        UnauthorizedError: If token is invalid or user not found
    """
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        raise UnauthorizedError("Invalid or expired token")
    
    username: Optional[str] = payload.get("sub")
    if not username:
        raise UnauthorizedError("Invalid token payload")
    
    user = crud.get_user_by_username(db, username)
    if not user:
        raise UnauthorizedError("User not found")
    
    return user


def get_current_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Verify that the current user is an admin.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User object if user is admin
        
    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


# Optional authentication (returns None if not authenticated)
def get_optional_user(
    db: Session = Depends(get_db),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.
    Useful for endpoints that have different behavior for authenticated users.
    
    Args:
        db: Database session
        credentials: Optional HTTP Bearer credentials
        
    Returns:
        User object if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        return None
    
    username: Optional[str] = payload.get("sub")
    if not username:
        return None
    
    return crud.get_user_by_username(db, username)


def validate_id(id_param: str) -> None:
    """
    Validate ID parameter format according to OpenAPI spec.
    
    IDs must match the pattern: ^[a-zA-Z0-9\-]+$
    (alphanumeric characters and hyphens only)
    
    Args:
        id_param: The ID parameter to validate
        
    Raises:
        HTTPException: 400 if id format is invalid (doesn't match pattern)
    """
    import re
    
    # OpenAPI spec pattern for ArtifactID: ^[a-zA-Z0-9\-]+$
    if not re.match(r'^[a-zA-Z0-9\-]+$', id_param):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid artifact ID format"
        )
