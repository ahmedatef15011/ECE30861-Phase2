"""Database connection management."""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, scoped_session
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
    connect_args = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        echo=bool(os.getenv("SQL_ECHO", "False").lower() == "true"),
        pool_pre_ping=True,
        poolclass=None,  # Use NullPool for SQLite to avoid connection issues
    )
else:
    # PostgreSQL connection pool settings
    engine = create_engine(
        DATABASE_URL,
        echo=bool(os.getenv("SQL_ECHO", "False").lower() == "true"),
        pool_pre_ping=True,  # Verify connections before using them
        pool_size=5,  # Reduced pool size to minimize connections during reset
        max_overflow=5,  # Reduced overflow
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
SessionLocal = scoped_session(sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
))


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
        # Ensure session is removed from registry to prevent holding connections
        SessionLocal.remove()


def reset_db(preserve_user_id: int = None) -> None:
    """
    Reset database by dropping and recreating all tables.
    
    Args:
        preserve_user_id: If provided, preserves the auth token for this user.
                         This allows the user who calls /reset to reuse their token.
    
    WARNING: This will delete all data!
    Use only for testing or the /reset endpoint.
    """
    import logging
    import time
    from sqlalchemy import text
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    # Step 0: Save the user's token if we're preserving it
    preserved_token = None
    preserved_expires = None
    if preserve_user_id:
        logger.info(f"   📊 Step 0: Saving token for user {preserve_user_id}...")
        try:
            with engine.connect() as conn:
                result = conn.execute(text(
                    """
                    SELECT token, expires_at
                    FROM auth_tokens
                    WHERE user_id = :user_id
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ), {"user_id": preserve_user_id})
                row = result.fetchone()
                if row:
                    preserved_token = row[0]
                    preserved_expires = row[1]
                    logger.info(f"   ✓ Token saved for preservation")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not save token: {e}")
            preserve_user_id = None  # Don't try to restore if we couldn't save
    
    try:
        logger.info("   📊 Step 1: Clearing session registry...")
        # Clear the session registry to ensure no sessions hold locks
        SessionLocal.remove()
        logger.info("   ✓ Session registry cleared")
        
        logger.info("   📊 Step 2: Disposing connection pool...")
        # Dispose all connections in the pool
        engine.dispose()
        logger.info("   ✓ Connection pool disposed")
        
        # For PostgreSQL: Force terminate all other connections to prevent locks
        if DATABASE_URL.startswith("postgresql"):
            logger.info("   🔒 Step 3: Terminating active database connections...")
            try:
                # Create temporary connection to terminate others
                with engine.connect() as conn:
                    # Get database name from URL
                    db_name = DATABASE_URL.split('/')[-1].split('?')[0]
                    
                    # First query: get count of connections to terminate
                    result = conn.execute(text(
                        """
                        SELECT COUNT(*) as cnt
                        FROM pg_stat_activity 
                        WHERE datname = :db_name 
                        AND pid <> pg_backend_pid()
                        """
                    ), {"db_name": db_name})
                    count = result.scalar()
                    logger.info(f"   📊 Found {count} other connections to terminate")
                    
                    # Second query: terminate them
                    if count > 0:
                        conn.execute(text(
                            """
                            SELECT pg_terminate_backend(pid) 
                            FROM pg_stat_activity 
                            WHERE datname = :db_name 
                            AND pid <> pg_backend_pid()
                            """
                        ), {"db_name": db_name})
                        conn.commit()
                        logger.info("   ✓ Active connections terminated")
                    
                    # Wait for connections to fully close
                    logger.info("   ⏳ Waiting for connections to close...")
                    time.sleep(0.5)  # Give connections time to fully disconnect
                    logger.info("   ✓ Wait complete")
                    
            except Exception as term_error:
                logger.warning(f"   ⚠️  Could not terminate connections: {term_error}")
                logger.info("   → Proceeding anyway...")
        
        logger.info("   🗑️  Step 4: Dropping all tables...")
        # Drop all tables with explicit timeout
        try:
            with engine.begin() as conn:
                # Set statement timeout to 60 seconds for DROP operations
                if DATABASE_URL.startswith("postgresql"):
                    conn.execute(text("SET statement_timeout = 60000"))
                
                # Drop all tables
                Base.metadata.drop_all(bind=conn)
            logger.info("   ✓ Tables dropped successfully")
        except Exception as drop_error:
            logger.error(f"   ❌ Failed to drop tables: {drop_error}")
            # Try to recover by disposing and trying again
            engine.dispose()
            time.sleep(1)
            Base.metadata.drop_all(bind=engine)
            logger.info("   ✓ Tables dropped (after retry)")
        
        logger.info("   🏗️  Step 5: Creating fresh tables...")
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("   ✓ Tables created successfully")
        
        logger.info("   🔄 Step 6: Resetting connection pool...")
        # Reset connection pool after structure changes
        engine.dispose()
        logger.info("   ✓ Connection pool reset")
        
        # Step 7: Recreate the default admin user (needed for token foreign key)
        if preserve_user_id and preserved_token:
            logger.info(f"   👤 Step 7a: Recreating default admin user...")
            try:
                from src.database.init_db import create_default_user
                create_default_user()
                logger.info(f"   ✓ Default admin user recreated")
            except Exception as e:
                logger.error(f"   ⚠️  Could not recreate default admin user: {e}")
                logger.info("   → Token restoration may fail")
        
        # Step 7b: Restore the preserved token if needed
        if preserve_user_id and preserved_token:
            logger.info(f"   📊 Step 7b: Restoring token for user {preserve_user_id}...")
            try:
                with engine.connect() as conn:
                    conn.execute(text(
                        """
                        INSERT INTO auth_tokens (user_id, token, expires_at, is_revoked, usage_count, created_at)
                        VALUES (:user_id, :token, :expires_at, false, 0, :created_at)
                        """
                    ), {
                        "user_id": preserve_user_id,
                        "token": preserved_token,
                        "expires_at": preserved_expires,
                        "created_at": datetime.utcnow()
                    })
                    conn.commit()
                    logger.info(f"   ✓ Token restored for user {preserve_user_id}")
            except Exception as e:
                logger.error(f"   ⚠️  Could not restore token: {e}")
                logger.info("   → Autograder will need to call /authenticate for a new token")
        
    except Exception as e:
        logger.error(f"   ❌ reset_db failed at some step: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise
