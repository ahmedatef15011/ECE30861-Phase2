"""Database connection management."""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from .models import Base

# Load .env file if it exists (for local development and RDS connection)
# COMMENTED OUT TO USE LOCAL SQLITE FOR TESTING
# env_path = Path(__file__).parent.parent.parent / ".env"
# if env_path.exists():
#     load_dotenv(env_path)

# Database URL from environment variable or default to SQLite for local development
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./ml_registry.db"  # Local SQLite for development
)

# For PostgreSQL (production/AWS RDS), format should be:
# postgresql://username:password@host:port/database
# Example: postgresql://admin:password@db.example.com:5432/ml_registry

# Create engine
# For SQLite, we need check_same_thread=False for FastAPI
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

# Configure engine with production-ready settings
engine_kwargs = {
    "connect_args": connect_args,
    "echo": bool(os.getenv("SQL_ECHO", "False").lower() == "true"),  # Log SQL queries if SQL_ECHO=true
    "pool_pre_ping": True,  # Verify connections before using them
}

# Add PostgreSQL-specific connection pool settings for production
if DATABASE_URL.startswith("postgresql"):
    engine_kwargs.update({
        "pool_size": 5,  # Number of connections to keep in the pool
        "max_overflow": 10,  # Max extra connections beyond pool_size
        "pool_timeout": 10,  # Seconds to wait for a connection from the pool (reduced from 30)
        "pool_recycle": 3600,  # Recycle connections after 1 hour
        "connect_args": {
            "connect_timeout": 5,  # 5 second connection timeout (reduced from 10)
            "options": "-c statement_timeout=10000"  # 10 second query timeout
        }
    })

engine = create_engine(DATABASE_URL, **engine_kwargs)


# Register REGEXP function for SQLite with timeout protection
@event.listens_for(engine, "connect")
def enable_sqlite_regexp(dbapi_connection, connection_record):
    """
    Enable REGEXP operator for SQLite with timeout protection.
    
    SQLite doesn't have built-in regex support, so we need to provide
    a custom function using Python's re module.
    
    This implementation includes protection against ReDoS attacks by:
    1. Catching regex errors
    2. Using a timeout via threading (if pattern takes too long)
    """
    if DATABASE_URL.startswith("sqlite"):
        import threading
        
        REGEX_TIMEOUT = 0.5  # Max 500ms for regex execution
        
        def regexp(pattern, value):
            """Regex matching function for SQLite with timeout protection."""
            if value is None or pattern is None:
                return False
            
            result = [False]
            error = [None]
            
            def do_match():
                try:
                    result[0] = re.search(pattern, value, re.IGNORECASE) is not None
                except re.error:
                    result[0] = False
                except Exception as e:
                    error[0] = e
                    result[0] = False
            
            # Run regex in a thread with timeout
            thread = threading.Thread(target=do_match)
            thread.daemon = True
            thread.start()
            thread.join(REGEX_TIMEOUT)
            
            if thread.is_alive():
                # Regex took too long - likely ReDoS attack
                return False
            
            return result[0]
        
        dbapi_connection.create_function("regexp", 2, regexp)


# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def init_db() -> None:
    """Initialize database by creating all tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function for FastAPI to get database session.
    
    Usage in FastAPI:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            # Use db here
            pass
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reset_db() -> None:
    """
    Reset database by dropping and recreating all tables.
    
    WARNING: This will delete all data!
    Use only for testing or the /reset endpoint.
    """
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
