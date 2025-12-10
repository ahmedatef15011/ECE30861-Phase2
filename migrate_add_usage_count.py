"""Add usage_count column to auth_tokens table."""

import logging
from sqlalchemy import create_engine, text
from src.database.connection import DATABASE_URL

logger = logging.getLogger(__name__)


def migrate_usage_count():
    """Add usage_count column to auth_tokens table if it doesn't exist."""
    engine = create_engine(DATABASE_URL)

    logger.info("Checking auth_tokens.usage_count column...")

    with engine.connect() as conn:
        # First check if auth_tokens table exists
        try:
            conn.execute(text("SELECT 1 FROM auth_tokens LIMIT 0"))
        except Exception:
            # Table doesn't exist yet - init_db() will create it correctly
            logger.info(
                "Table auth_tokens doesn't exist yet. Will be created by "
                "init_db()."
            )
            return

        # Check if column exists (works for both SQLite and PostgreSQL)
        try:
            # Try to query the column
            conn.execute(text("""
                SELECT usage_count FROM auth_tokens LIMIT 1
            """))
            logger.info(
                "✅ Column 'usage_count' already exists. No migration needed."
            )
            return
        except Exception:
            # Column doesn't exist, continue with migration
            pass

        # Add the column
        logger.info("📝 Adding usage_count column to existing table...")
        
        # Use IF NOT EXISTS for PostgreSQL compatibility
        if DATABASE_URL.startswith("postgresql"):
            # PostgreSQL doesn't support IF NOT EXISTS in ALTER TABLE ADD
            # We already checked above, so just add it
            conn.execute(text("""
                ALTER TABLE auth_tokens
                ADD COLUMN usage_count INTEGER NOT NULL DEFAULT 0
            """))
        else:
            # SQLite
            conn.execute(text("""
                ALTER TABLE auth_tokens
                ADD COLUMN usage_count INTEGER NOT NULL DEFAULT 0
            """))
        
        conn.commit()

        logger.info(
            "✅ Migration complete! Column 'usage_count' added successfully."
        )


if __name__ == "__main__":
    migrate_usage_count()
