"""
Database migration script for adding sensitive module history and malicious model tracking tables.

Run this script to add the new security-related tables to an existing database.
"""

import os
import sys
import logging

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import inspect, text
from src.database.connection import engine, SessionLocal
from src.database.models import Base, SensitiveModuleHistory, MaliciousModelReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_security_tables():
    """Add sensitive module history and malicious model report tables if they don't exist."""
    
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()
    
    tables_to_create = []
    
    # Check if sensitive_module_history table exists
    if "sensitive_module_history" not in existing_tables:
        tables_to_create.append("sensitive_module_history")
        logger.info("Table 'sensitive_module_history' does not exist - will be created")
    else:
        logger.info("Table 'sensitive_module_history' already exists")
    
    # Check if malicious_model_reports table exists
    if "malicious_model_reports" not in existing_tables:
        tables_to_create.append("malicious_model_reports")
        logger.info("Table 'malicious_model_reports' does not exist - will be created")
    else:
        logger.info("Table 'malicious_model_reports' already exists")
    
    if tables_to_create:
        logger.info(f"Creating {len(tables_to_create)} new table(s)...")
        
        # Create only the tables that don't exist
        # Using checkfirst=True ensures we don't error on existing tables
        Base.metadata.create_all(engine, checkfirst=True)
        
        logger.info("✅ Security tables created successfully!")
        
        # Verify tables were created
        inspector = inspect(engine)
        new_tables = inspector.get_table_names()
        
        for table_name in tables_to_create:
            if table_name in new_tables:
                logger.info(f"  ✅ {table_name} created")
            else:
                logger.error(f"  ❌ {table_name} creation failed")
    else:
        logger.info("✅ All security tables already exist - no migration needed")
    
    return True


def verify_tables():
    """Verify that the security tables exist and have the correct structure."""
    inspector = inspect(engine)
    
    # Check sensitive_module_history
    if "sensitive_module_history" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("sensitive_module_history")]
        expected = ["id", "package_id", "user_id", "action", "field_changed", 
                   "old_value", "new_value", "change_summary", "changed_at",
                   "ip_address", "user_agent", "additional_context"]
        missing = [c for c in expected if c not in columns]
        if missing:
            logger.warning(f"sensitive_module_history missing columns: {missing}")
        else:
            logger.info("✅ sensitive_module_history table structure verified")
    else:
        logger.error("❌ sensitive_module_history table not found")
    
    # Check malicious_model_reports
    if "malicious_model_reports" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("malicious_model_reports")]
        expected = ["id", "package_id", "reported_by_user_id", "detection_method",
                   "severity", "status", "reason", "evidence", "reviewed_by_user_id",
                   "reviewed_at", "resolution_notes", "reported_at", "updated_at"]
        missing = [c for c in expected if c not in columns]
        if missing:
            logger.warning(f"malicious_model_reports missing columns: {missing}")
        else:
            logger.info("✅ malicious_model_reports table structure verified")
    else:
        logger.error("❌ malicious_model_reports table not found")


if __name__ == "__main__":
    print("=" * 60)
    print("Security Tables Migration")
    print("=" * 60)
    print()
    
    try:
        migrate_security_tables()
        print()
        verify_tables()
        print()
        print("=" * 60)
        print("Migration completed successfully!")
        print("=" * 60)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)
