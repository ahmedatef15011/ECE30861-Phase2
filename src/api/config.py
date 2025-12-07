"""Configuration settings for the FastAPI application."""

import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ml_registry.db")
    
    # JWT
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "600"))
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "ML Model Registry"
    PROJECT_VERSION: str = "1.0.0"
    
    # CORS
    CORS_ORIGINS: list = ["*"]  # TODO: Restrict in production
    
    # AWS Configuration
    AWS_REGION: Optional[str] = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET: Optional[str] = os.getenv("S3_BUCKET")
    
    # AWS Bedrock LLM Configuration
    BEDROCK_MODEL_ID: str = os.getenv(
        "BEDROCK_MODEL_ID", 
        "anthropic.claude-3-haiku-20240307-v1:0"
    )
    BEDROCK_MAX_TOKENS: int = int(os.getenv("BEDROCK_MAX_TOKENS", "4096"))
    BEDROCK_TEMPERATURE: float = float(os.getenv("BEDROCK_TEMPERATURE", "0.3"))
    BEDROCK_ENABLED: bool = os.getenv("BEDROCK_ENABLED", "true").lower() == "true"
    
    # Default admin user (as per spec)
    DEFAULT_ADMIN_USERNAME: str = "ece30861defaultadminuser"
    DEFAULT_ADMIN_PASSWORD: str = '''correcthorsebatterystaple123(!__+@**(A'"`;DROP TABLE packages;'''
    DEFAULT_ADMIN_EMAIL: str = "admin@mlregistry.local"


settings = Settings()
