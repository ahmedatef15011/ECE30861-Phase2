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
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        echo=bool(os.getenv("SQL_ECHO", "False").lower() == "true"),
        pool_pre_ping=True,
    )
else:
    # PostgreSQL connection pool settings
    engine = create_engine(
        DATABASE_URL,
        echo=bool(os.getenv("SQL_ECHO", "False").lower() == "true"),
        pool_pre_ping=True,  # Verify connections before using them
        pool_size=10,  # Maximum number of permanent connections
        max_overflow=20,  # Maximum number of temporary connections
        pool_timeout=30,  # Timeout for getting connection from pool (seconds)
        pool_recycle=3600,  # Recycle connections after 1 hour
        connect_args={
            "connect_timeout": 10,  # Connection timeout (seconds)
            "options": "-c statement_timeout=30000"  # Query timeout 30 seconds
        }
    )


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
    import logging
    from sqlalchemy import text
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("   📊 Disposing existing connections...")
        # Close all connections in the pool
        engine.dispose()
        logger.info("   ✓ Connection pool disposed")
        
        # For PostgreSQL: Force terminate all other connections to prevent locks
        if DATABASE_URL.startswith("postgresql"):
            logger.info("   🔒 Terminating active database connections...")
            try:
                # Create temporary connection to terminate others
                with engine.connect() as conn:
                    # Get database name from URL
                    db_name = DATABASE_URL.split('/')[-1].split('?')[0]
                    
                    # Terminate all connections to this database except current one
                    conn.execute(text(
                        f"""
                        SELECT pg_terminate_backend(pid) 
                        FROM pg_stat_activity 
                        WHERE datname = :db_name 
                        AND pid <> pg_backend_pid()
                        """
                    ), {"db_name": db_name})
                    conn.commit()
                    logger.info("   ✓ Active connections terminated")
            except Exception as term_error:
                logger.warning(f"   ⚠️  Could not terminate connections: {term_error}")
                logger.info("   → Proceeding anyway...")
        
        logger.info("   🗑️  Dropping all tables...")
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        logger.info("   ✓ Tables dropped successfully")
        
        logger.info("   🏗️  Creating fresh tables...")
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("   ✓ Tables created successfully")
        
        logger.info("   🔄 Resetting connection pool...")
        # Reset connection pool after structure changes
        engine.dispose()
        logger.info("   ✓ Connection pool reset")
        
    except Exception as e:
        logger.error(f"   ❌ reset_db failed at some step: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        raise
