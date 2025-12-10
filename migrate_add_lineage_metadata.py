"""
Database migration: Add lineage_metadata column to Package table.

This migration adds a JSONB column to store LLM-analyzed dependency
information, avoiding repeated API calls for lineage extraction.

Run this script to update your database:
    python migrate_add_lineage_metadata.py
"""

import logging
import sys
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_lineage_metadata():
    """Add lineage_metadata column to Package table if it doesn't exist."""
    try:
        from src.database.connection import engine
        
        with engine.connect() as conn:
            # Check if column already exists
            logger.info("Checking if lineage_metadata column exists...")
            
            # Use SQLite-compatible PRAGMA for column check
            result = conn.execute(
                text("PRAGMA table_info(packages)")
            )
            columns = [row[1] for row in result.fetchall()]
            
            if 'lineage_metadata' in columns:
                logger.info(
                    "✅ Column 'lineage_metadata' already exists - "
                    "no migration needed"
                )
                return
            
            # Add lineage_metadata column (JSON for SQLite, JSONB for PG)
            logger.info("Adding lineage_metadata column to packages table...")
            conn.execute(text("""
                ALTER TABLE packages
                ADD COLUMN lineage_metadata JSON DEFAULT NULL
            """))
            conn.commit()
            
            logger.info("✅ Migration complete: lineage_metadata column added")
            logger.info(
                "   This column will store LLM-analyzed dependencies to "
                "avoid repeated API calls"
            )
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        logger.error(
            "Please ensure your database is running and accessible"
        )
        sys.exit(1)


if __name__ == "__main__":
    migrate_lineage_metadata()
