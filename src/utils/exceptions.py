"""Custom exception classes for the ML Registry API."""

from fastapi import HTTPException, status


class UserNotFoundError(HTTPException):
    """Exception raised when a user is not found."""
    
    def __init__(self, username: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not found"
        )


class PackageNotFoundError(HTTPException):
    """Exception raised when a package is not found."""
    
    def __init__(self, package_id: int):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Package {package_id} not found"
        )


class InvalidCredentialsError(HTTPException):
    """Exception raised when login credentials are invalid."""
    
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )


class UnauthorizedError(HTTPException):
    """Exception raised when a user is not authenticated."""
    
    def __init__(self, detail: str = "Not authenticated"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class ForbiddenError(HTTPException):
    """Exception raised when a user lacks permission."""
    
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )

class AccessControlDeniedError(HTTPException):
    """Exception raised when access control program denies access to sensitive model."""
    
    def __init__(self, detail: str = "Access denied by model access control policy"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )