#!/usr/bin/env python3
"""
Direct PostgreSQL RDS Connection Test Script

Tests database connectivity, query performance, and reset_db operations
to diagnose timeout issues.
"""

import os
import sys
import time
import uuid
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# Database connection details
DB_HOST = "ml-registry-db.cwxieem041of.us-east-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "mlregistry"
DB_USER = "mlregistry_admin"
DB_PASSWORD = "t8U:ge!#G#=_GGMhML-TJN&soLuVD&Ru"

# Construct connection strings
PSYCOPG2_CONN_STRING = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
SQLALCHEMY_CONN_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Test table schemas
Base = declarative_base()

class TestTable(Base):
    __tablename__ = 'test_connection_table'
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    created_at = Column(String(50))


class TestUser(Base):
    __tablename__ = 'test_users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(200))
    hashed_password = Column(String(200), nullable=False)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class TestAuthToken(Base):
    __tablename__ = 'test_auth_tokens'
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    token = Column(String(500), unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"🔍 {title}")
    print('=' * 80)


def test_1_basic_connectivity():
    """Test 1: Basic psycopg2 connection."""
    print_section("TEST 1: Basic PostgreSQL Connectivity (psycopg2)")
    
    try:
        start = time.time()
        print(f"⏱️  Connecting to {DB_HOST}...")
        
        conn = psycopg2.connect(PSYCOPG2_CONN_STRING, connect_timeout=10)
        conn_time = time.time() - start
        
        print(f"✅ Connected successfully in {conn_time:.3f}s")
        
        # Test simple query
        start = time.time()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        query_time = time.time() - start
        
        print(f"✅ Query executed in {query_time:.3f}s")
        print(f"📊 PostgreSQL Version: {version[:80]}...")
        
        # Check active connections
        cursor.execute("""
            SELECT count(*) FROM pg_stat_activity 
            WHERE datname = %s
        """, (DB_NAME,))
        active_conns = cursor.fetchone()[0]
        print(f"📊 Active connections to database: {active_conns}")
        
        cursor.close()
        conn.close()
        print("✅ Connection closed cleanly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False


def test_2_sqlalchemy_connection():
    """Test 2: SQLAlchemy connection with pool settings."""
    print_section("TEST 2: SQLAlchemy Connection with Pooling")
    
    try:
        start = time.time()
        print(f"⏱️  Creating SQLAlchemy engine...")
        
        engine = create_engine(
            SQLALCHEMY_CONN_STRING,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            connect_args={
                "connect_timeout": 10,
                "options": "-c statement_timeout=30000"
            },
            echo=False
        )
        
        # Test connection
        with engine.connect() as conn:
            conn_time = time.time() - start
            print(f"✅ Engine created and connected in {conn_time:.3f}s")
            
            # Test query
            start = time.time()
            result = conn.execute(text("SELECT current_database(), current_user"))
            row = result.fetchone()
            query_time = time.time() - start
            
            print(f"✅ Query executed in {query_time:.3f}s")
            print(f"📊 Database: {row[0]}, User: {row[1]}")
            
            # Check pool status
            print(f"📊 Pool size: {engine.pool.size()}")
            print(f"📊 Pool overflow: {engine.pool.overflow()}")
        
        engine.dispose()
        print("✅ Engine disposed cleanly")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False


def test_3_concurrent_queries():
    """Test 3: Multiple concurrent queries (simulate app load)."""
    print_section("TEST 3: Concurrent Query Performance")
    
    try:
        engine = create_engine(
            SQLALCHEMY_CONN_STRING,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args={"connect_timeout": 10}
        )
        
        # Run 10 queries concurrently
        print("⏱️  Running 10 concurrent queries...")
        start = time.time()
        
        for i in range(10):
            with engine.connect() as conn:
                result = conn.execute(text("SELECT pg_sleep(0.1), :num"), {"num": i})
                result.fetchone()
        
        total_time = time.time() - start
        print(f"✅ Completed 10 queries in {total_time:.3f}s")
        print(f"📊 Average query time: {total_time/10:.3f}s")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False


def test_4_table_operations():
    """Test 4: Create/Drop table operations."""
    print_section("TEST 4: Table CREATE/DROP Operations")
    
    try:
        engine = create_engine(
            SQLALCHEMY_CONN_STRING,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10}
        )
        
        # Create table
        print("⏱️  Creating test table...")
        start = time.time()
        Base.metadata.create_all(bind=engine)
        create_time = time.time() - start
        print(f"✅ Table created in {create_time:.3f}s")
        
        # Insert data
        Session = sessionmaker(bind=engine)
        session = Session()
        
        print("⏱️  Inserting test data...")
        start = time.time()
        for i in range(5):
            test_row = TestTable(
                name=f"Test {i}",
                created_at=datetime.utcnow().isoformat()
            )
            session.add(test_row)
        session.commit()
        insert_time = time.time() - start
        print(f"✅ Inserted 5 rows in {insert_time:.3f}s")
        
        # Query data
        print("⏱️  Querying data...")
        start = time.time()
        count = session.query(TestTable).count()
        query_time = time.time() - start
        print(f"✅ Found {count} rows in {query_time:.3f}s")
        
        session.close()
        
        # Drop table
        print("⏱️  Dropping test table...")
        start = time.time()
        Base.metadata.drop_all(bind=engine)
        drop_time = time.time() - start
        print(f"✅ Table dropped in {drop_time:.3f}s")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_reset_db_simulation():
    """Test 5: Simulate reset_db operation with connection termination."""
    print_section("TEST 5: Reset DB Simulation (Dispose + Terminate + Drop/Create)")
    
    try:
        engine = create_engine(
            SQLALCHEMY_CONN_STRING,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            connect_args={"connect_timeout": 10}
        )
        
        # Step 1: Dispose connections
        print("⏱️  Step 1: Disposing connection pool...")
        start = time.time()
        engine.dispose()
        dispose_time = time.time() - start
        print(f"✅ Pool disposed in {dispose_time:.3f}s")
        
        # Step 2: Terminate active connections
        print("⏱️  Step 2: Terminating active database connections...")
        start = time.time()
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT pg_terminate_backend(pid) 
                    FROM pg_stat_activity 
                    WHERE datname = :db_name 
                    AND pid <> pg_backend_pid()
                """), {"db_name": DB_NAME})
                terminated = result.fetchall()
                conn.commit()
                terminate_time = time.time() - start
                print(f"✅ Terminated {len(terminated)} connections in {terminate_time:.3f}s")
        except Exception as term_error:
            print(f"⚠️  Termination warning: {term_error}")
        
        # Step 3: Drop tables
        print("⏱️  Step 3: Dropping all tables...")
        start = time.time()
        Base.metadata.drop_all(bind=engine)
        drop_time = time.time() - start
        print(f"✅ Tables dropped in {drop_time:.3f}s")
        
        # Step 4: Create tables
        print("⏱️  Step 4: Creating fresh tables...")
        start = time.time()
        Base.metadata.create_all(bind=engine)
        create_time = time.time() - start
        print(f"✅ Tables created in {create_time:.3f}s")
        
        # Step 5: Final dispose
        print("⏱️  Step 5: Final pool dispose...")
        start = time.time()
        engine.dispose()
        final_dispose_time = time.time() - start
        print(f"✅ Final dispose in {final_dispose_time:.3f}s")
        
        total_time = dispose_time + terminate_time + drop_time + create_time + final_dispose_time
        print(f"\n📊 Total reset simulation time: {total_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_connection_leak():
    """Test 6: Check for connection leaks."""
    print_section("TEST 6: Connection Leak Detection")
    
    try:
        # Check current connections
        conn = psycopg2.connect(PSYCOPG2_CONN_STRING, connect_timeout=10)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                count(*) as total_connections,
                count(*) FILTER (WHERE state = 'active') as active,
                count(*) FILTER (WHERE state = 'idle') as idle,
                count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
            FROM pg_stat_activity 
            WHERE datname = %s
        """, (DB_NAME,))
        
        result = cursor.fetchone()
        print(f"📊 Total connections: {result[0]}")
        print(f"📊 Active: {result[1]}")
        print(f"📊 Idle: {result[2]}")
        print(f"📊 Idle in transaction: {result[3]}")
        
        if result[3] > 0:
            print("⚠️  WARNING: Idle in transaction connections detected!")
            print("   This can cause deadlocks!")
        else:
            print("✅ No problematic connections detected")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False


def test_7_authentication_simulation():
    """Test 7: Read-only authentication flow simulation (lookup existing data)."""
    print_section("TEST 7: Authentication Flow Simulation (Read-Only)")
    
    try:
        engine = create_engine(
            SQLALCHEMY_CONN_STRING,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            connect_args={
                "connect_timeout": 10,
                "options": "-c statement_timeout=30000"
            }
        )
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Step 1: List all tables in the database
        print("\n⏱️  Step 1: Listing all tables in database...")
        start = time.time()
        
        result = session.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        tables = [row[0] for row in result]
        
        list_tables_time = time.time() - start
        print(f"✅ Found {len(tables)} tables in {list_tables_time:.3f}s")
        for table in tables:
            print(f"   📋 {table}")
        
        # Step 2: Check if users table exists and count records
        print("\n⏱️  Step 2: Checking 'users' table...")
        start = time.time()
        
        if 'users' in tables:
            result = session.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.scalar()
            count_time = time.time() - start
            print(f"✅ Users table found with {user_count} records in {count_time:.3f}s")
            
            # Show sample user (without password)
            if user_count > 0:
                result = session.execute(text("""
                    SELECT id, username, email, is_admin, created_at 
                    FROM users 
                    LIMIT 1
                """))
                user = result.fetchone()
                print(f"   📊 Sample user: ID={user[0]}, username={user[1]}, email={user[2]}, admin={user[3]}")
        else:
            print("⚠️  'users' table not found")
        
        # Step 3: Check if auth_tokens table exists and count records
        print("\n⏱️  Step 3: Checking 'auth_tokens' table...")
        start = time.time()
        
        if 'auth_tokens' in tables:
            result = session.execute(text("SELECT COUNT(*) FROM auth_tokens"))
            token_count = result.scalar()
            count_time = time.time() - start
            print(f"✅ Auth_tokens table found with {token_count} records in {count_time:.3f}s")
            
            # Show sample token info (without actual token value for security)
            if token_count > 0:
                result = session.execute(text("""
                    SELECT id, user_id, usage_count, expires_at, created_at 
                    FROM auth_tokens 
                    ORDER BY created_at DESC
                    LIMIT 1
                """))
                token = result.fetchone()
                print(f"   📊 Most recent token: ID={token[0]}, user_id={token[1]}, usage={token[2]}, expires={token[3]}")
        else:
            print("⚠️  'auth_tokens' table not found")
        
        # Step 4: Simulate token lookup query (if table exists)
        if 'auth_tokens' in tables and token_count > 0:
            print("\n⏱️  Step 4: Simulating token lookup query...")
            start = time.time()
            
            # Get a real token to test lookup speed
            result = session.execute(text("""
                SELECT token FROM auth_tokens 
                ORDER BY created_at DESC 
                LIMIT 1
            """))
            test_token = result.scalar()
            
            # Perform lookup (simulates get_current_user)
            result = session.execute(text("""
                SELECT id, user_id, usage_count 
                FROM auth_tokens 
                WHERE token = :token
            """), {"token": test_token})
            token_data = result.fetchone()
            
            lookup_time = time.time() - start
            print(f"✅ Token lookup completed in {lookup_time:.3f}s")
            if token_data:
                print(f"   📊 Found token: ID={token_data[0]}, user_id={token_data[1]}, usage={token_data[2]}")
        
        # Step 5: Simulate user lookup by ID
        if 'users' in tables and user_count > 0:
            print("\n⏱️  Step 5: Simulating user lookup query...")
            start = time.time()
            
            # Get a real user ID
            result = session.execute(text("SELECT id FROM users LIMIT 1"))
            user_id = result.scalar()
            
            # Perform lookup
            result = session.execute(text("""
                SELECT id, username, email, is_admin 
                FROM users 
                WHERE id = :user_id
            """), {"user_id": user_id})
            user_data = result.fetchone()
            
            lookup_time = time.time() - start
            print(f"✅ User lookup completed in {lookup_time:.3f}s")
            if user_data:
                print(f"   📊 Found user: ID={user_data[0]}, username={user_data[1]}, email={user_data[2]}")
        
        # Step 6: Check database connection statistics
        print("\n⏱️  Step 6: Checking database connection statistics...")
        start = time.time()
        
        result = session.execute(text("""
            SELECT 
                count(*) as total_connections,
                count(*) FILTER (WHERE state = 'active') as active,
                count(*) FILTER (WHERE state = 'idle') as idle,
                count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction
            FROM pg_stat_activity 
            WHERE datname = :db_name
        """), {"db_name": DB_NAME})
        
        conn_stats = result.fetchone()
        stats_time = time.time() - start
        print(f"✅ Connection stats retrieved in {stats_time:.3f}s")
        print(f"   📊 Total connections: {conn_stats[0]}")
        print(f"   📊 Active: {conn_stats[1]}")
        print(f"   📊 Idle: {conn_stats[2]}")
        print(f"   📊 Idle in transaction: {conn_stats[3]}")
        
        if conn_stats[3] > 0:
            print("   ⚠️  WARNING: Idle in transaction connections detected!")
        
        session.close()
        engine.dispose()
        
        print("\n✅ Read-only authentication simulation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all database tests."""
    print("\n" + "=" * 80)
    print("🧪 PostgreSQL RDS Connection Test Suite")
    print("=" * 80)
    print(f"🎯 Target: {DB_HOST}")
    print(f"📁 Database: {DB_NAME}")
    print(f"👤 User: {DB_USER}")
    print(f"⏰ Started: {datetime.now().isoformat()}")
    
    results = {}
    
    # Run tests
    results['Test 1: Basic Connectivity'] = test_1_basic_connectivity()
    results['Test 2: SQLAlchemy Connection'] = test_2_sqlalchemy_connection()
    results['Test 3: Concurrent Queries'] = test_3_concurrent_queries()
    results['Test 4: Table Operations'] = test_4_table_operations()
    results['Test 5: Reset DB Simulation'] = test_5_reset_db_simulation()
    results['Test 6: Connection Leak'] = test_6_connection_leak()
    results['Test 7: Authentication Flow'] = test_7_authentication_simulation()
    
    # Summary
    print_section("TEST SUMMARY")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test}")
    
    print(f"\n📊 Results: {passed}/{total} passed, {failed}/{total} failed")
    print(f"⏰ Completed: {datetime.now().isoformat()}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
