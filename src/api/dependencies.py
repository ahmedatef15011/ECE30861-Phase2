"""Dependency injection functions for FastAPI endpoints."""

from typing import Generator, Optional
import re
import signal
import threading
from contextlib import contextmanager
from fastapi import Depends, HTTPException, status, Header
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
REGEX_TIMEOUT = 0.5


class RegexTimeoutError(Exception):
    """Exception raised when regex execution times out."""
    pass


def _is_redos_pattern(pattern: str) -> bool:
    """
    Check if a regex pattern is a known ReDoS (Regular Expression Denial of Service) pattern.
    
    These specific patterns can cause exponential backtracking:
    - (a+)+$  - nested quantifiers
    - (a|aa)*$ - overlapping alternations with quantifier
    - (a{1,99999}){1,99999}$ - nested high repetition counts
    
    Args:
        pattern: The regex pattern to check
        
    Returns:
        True if the pattern is a known ReDoS pattern
    """
    # Pattern 1: Nested quantifiers like (a+)+, (a+)*, (a*)+, (a*)*
    # Matches: (x+)+, (x+)*, (x*)+, (x*)*, where x is any character/group
    nested_quantifier_pattern = r'\([^)]*[+*][^)]*\)[+*]'
    if re.search(nested_quantifier_pattern, pattern):
        return True
    
    # Pattern 2: Overlapping alternations with quantifier like (a|aa)+, (a|aa)*
    # This causes exponential backtracking on strings like "aaaaaaaaaaaa!"
    overlapping_alt_pattern = r'\([^)]*\|[^)]*\)[+*]'
    if re.search(overlapping_alt_pattern, pattern):
        return True
    
    # Pattern 3: Very high repetition counts like {1,99999} or {99999}
    # This can cause memory exhaustion
    high_rep_pattern = r'\{[\d,]*(?:9999|10000|99999|100000)[\d,]*\}'
    if re.search(high_rep_pattern, pattern):
        return True
    
    # Pattern 4: Nested repetition with braces like (a{n}){m} where n*m is large
    nested_brace_pattern = r'\([^)]*\{[^}]+\}[^)]*\)\{[^}]+\}'
    if re.search(nested_brace_pattern, pattern):
        return True
    
    return False


def validate_regex_pattern(pattern: str) -> str:
    """
    Validate a regex pattern for safe execution.
    
    Checks for:
    1. Pattern length limits
    2. Valid regex syntax  
    3. Known ReDoS vulnerability patterns
    
    Args:
        pattern: The regex pattern to validate
        
    Returns:
        The validated pattern
        
    Raises:
        HTTPException: 400 if the pattern is invalid or dangerous
    """
    # Handle special case of "*" (list all)
    if pattern == "*":
        return pattern
    
    # Check for empty pattern
    if not pattern or pattern.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="There is missing field(s) in the artifact_regex or it is formed improperly, or is invalid"
        )
    
    # Check length
    if len(pattern) > MAX_REGEX_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="There is missing field(s) in the artifact_regex or it is formed improperly, or is invalid"
        )
    
    # Try to compile the regex to check for syntax errors
    try:
        re.compile(pattern)
    except re.error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="There is missing field(s) in the artifact_regex or it is formed improperly, or is invalid"
        )
    
    # Check for known ReDoS patterns
    if _is_redos_pattern(pattern):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="There is missing field(s) in the artifact_regex or it is formed improperly, or is invalid"
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


# Custom header extraction for X-Authorization with strict "bearer " prefix
def get_x_authorization_token(
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
) -> str:
    """
    Extract JWT token from X-Authorization header.
    Strictly requires lowercase "bearer " prefix.
    
    Args:
        x_authorization: X-Authorization header value
        
    Returns:
        Extracted JWT token string
        
    Raises:
        UnauthorizedError: If header is missing or doesn't have "bearer " prefix
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not x_authorization:
        logger.warning(
            "🔒 AUTH FAILED: Missing X-Authorization header"
        )
        logger.info(
            "   Expected format: X-Authorization: bearer <token>"
        )
        raise UnauthorizedError("Missing X-Authorization header")
    
    logger.info(
        f"🔑 AUTH ATTEMPT: Received header value: '{x_authorization[:50]}...'"
        if len(x_authorization) > 50 else
        f"🔑 AUTH ATTEMPT: Received header value: '{x_authorization}'"
    )
    
    # Check for "bearer " prefix (case-insensitive)
    if not x_authorization.lower().startswith("bearer "):
        logger.warning(
            f"🔒 AUTH FAILED: Invalid format. "
            f"Received: '{x_authorization[:30]}...'"
        )
        logger.info(
            "   Expected format: bearer <token> (Bearer or bearer accepted)"
        )
        logger.info(
            f"   Got prefix: '{x_authorization[:10]}'"
        )
        raise UnauthorizedError(
            'Invalid X-Authorization format. Expected: "Bearer <token>" or "bearer <token>"'
        )
    
    # Strip "bearer " or "Bearer " prefix (7 characters) - case-insensitive
    token = x_authorization[7:]
    logger.info(
        f"✅ AUTH: Extracted JWT token (length: {len(token)})"
    )
    return token


def get_x_authorization_token_optional(
    x_authorization: Optional[str] = Header(None, alias="X-Authorization")
) -> Optional[str]:
    """
    Extract JWT token from X-Authorization header (optional).
    Returns None if header is missing or invalid.
    
    Args:
        x_authorization: X-Authorization header value
        
    Returns:
        Extracted JWT token string or None
    """
    if not x_authorization:
        return None
    
    # Check for "bearer " prefix (case-insensitive)
    if not x_authorization.lower().startswith("bearer "):
        return None
    
    # Strip "bearer " or "Bearer " prefix (7 characters) - case-insensitive
    return x_authorization[7:]


def get_current_user(
    token: str = Depends(get_x_authorization_token),
    db: Session = Depends(get_db)
) -> User:
    """
    Get the current authenticated user from JWT token.
    Validates token and enforces 1000 interaction limit.
    
    Args:
        token: JWT token from X-Authorization header
        db: Database session
        
    Returns:
        Authenticated User object
        
    Raises:
        UnauthorizedError: If token is invalid, expired, or limit exceeded
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔐 AUTH CHECK: Received token (length: {len(token)})")
    
    payload = verify_token(token)
    
    if not payload:
        raise UnauthorizedError("Invalid or expired token")
    
    username: Optional[str] = payload.get("sub")
    if not username:
        raise UnauthorizedError("Invalid token payload")
    
    # Validate token in database and check usage limit
    auth_token = crud.get_auth_token(db, token)
    if not auth_token:
        raise UnauthorizedError(
            "Token expired, revoked, or usage limit exceeded (1000 max)"
        )
    
    # Increment usage count
    if not crud.increment_token_usage(db, token):
        raise UnauthorizedError("Failed to track token usage")
    
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
    token: Optional[str] = Depends(get_x_authorization_token_optional)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.
    Useful for endpoints that have different behavior for authenticated users.
    
    Args:
        db: Database session
        token: Optional JWT token from X-Authorization header
        
    Returns:
        User object if authenticated, None otherwise
    """
    if not token:
        return None
    payload = verify_token(token)
    
    if not payload:
        return None
    
    username: Optional[str] = payload.get("sub")
    if not username:
        return None
    
    return crud.get_user_by_username(db, username)


def validate_id(id_param: str) -> None:
    r"""
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
