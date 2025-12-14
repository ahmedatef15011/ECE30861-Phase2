#!/usr/bin/env python3
"""
Test script to diagnose and test the reset_db() operation.
Tests both the diagnosis of what might block reset, and the actual reset operation.
"""

import os
import sys
import time
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker

# Database connection details
DB_HOST = "ml-registry-db.cwxieem041of.us-east-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "mlregistry"
DB_USER = "mlregistry_admin"
DB_PASSWORD = "t8U:ge!#G#=_GGMhML-TJN&soLuVD&Ru"

SQLALCHEMY_CONN_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"🔍 {title}")
    print('=' * 80)


def diagnose_reset_blockers():
    """Diagnose what might block the reset operation."""
    print_section("DIAGNOSIS: What could block reset?")
    
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
        
        # Check for locks
        print("\n⏱️  Checking for table locks...")
        start = time.time()
        
        result = session.execute(text("""
            SELECT 
                pg_class.relname AS table_name,
                pg_locks.mode AS lock_mode,
                pg_locks.granted,
                pg_stat_activity.state,
                pg_stat_activity.pid
            FROM pg_locks
            JOIN pg_class ON pg_locks.relation = pg_class.oid
            LEFT JOIN pg_stat_activity ON pg_locks.pid = pg_stat_activity.pid
            WHERE pg_class.relkind = 'r'
            AND pg_class.relname NOT LIKE 'pg_%'
            ORDER BY pg_class.relname
        """))
        
        locks = result.fetchall()
        check_time = time.time() - start
        
        if locks:
            print(f"❌ Found {len(locks)} locks in {check_time:.3f}s:")
            for lock in locks[:10]:
                print(f"   📋 {lock[0]}: {lock[1]} (granted={lock[2]}, state={lock[3]}, PID={lock[4]})")
        else:
            print(f"✅ No locks found in {check_time:.3f}s")
        
        # Check for idle in transaction
        print("\n⏱️  Checking for 'idle in transaction' connections...")
        start = time.time()
        
        result = session.execute(text("""
            SELECT 
                pid,
                usename,
                NOW() - state_change AS idle_duration,
                query_start
            FROM pg_stat_activity
            WHERE state = 'idle in transaction'
            AND datname = :db_name
        """), {"db_name": DB_NAME})
        
        idle_txns = result.fetchall()
        check_time = time.time() - start
        
        if idle_txns:
            print(f"❌ CRITICAL: Found {len(idle_txns)} 'idle in transaction' in {check_time:.3f}s:")
            for txn in idle_txns:
                print(f"   ⚠️  PID {txn[0]}: idle for {txn[2]}")
        else:
            print(f"✅ No 'idle in transaction' connections in {check_time:.3f}s")
        
        # Check for long-running queries
        print("\n⏱️  Checking for long-running queries...")
        start = time.time()
        
        result = session.execute(text("""
            SELECT 
                pid,
                usename,
                NOW() - query_start AS duration,
                query
            FROM pg_stat_activity
            WHERE state != 'idle'
            AND datname = :db_name
            AND pid != pg_backend_pid()
            ORDER BY duration DESC
        """), {"db_name": DB_NAME})
        
        queries = result.fetchall()
        check_time = time.time() - start
        
        if queries:
            print(f"⚠️  Found {len(queries)} active queries in {check_time:.3f}s:")
            for q in queries[:5]:
                print(f"   📊 PID {q[0]}: {q[2]}")
        else:
            print(f"✅ No long-running queries in {check_time:.3f}s")
        
        session.close()
        engine.dispose()
        
        return len(locks) == 0 and len(idle_txns) == 0
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False


def test_connection_termination():
    """Test the pg_terminate_backend logic that reset uses."""
    print_section("TEST 1: Connection Termination")
    
    try:
        engine = create_engine(
            SQLALCHEMY_CONN_STRING,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 10,
                "options": "-c statement_timeout=30000"
            }
        )
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        print("\n⏱️  Querying connections to terminate...")
        start = time.time()
        
        # Get database name from URL
        db_name = SQLALCHEMY_CONN_STRING.split('/')[-1].split('?')[0]
        
        result = session.execute(text("""
            SELECT 
                pid,
                usename,
                state,
                query
            FROM pg_stat_activity 
            WHERE datname = :db_name 
            AND pid <> pg_backend_pid()
        """), {"db_name": db_name})
        
        connections = result.fetchall()
        query_time = time.time() - start
        
        print(f"✅ Found {len(connections)} other connections in {query_time:.3f}s")
        if connections:
            print(f"   Would terminate: {[c[0] for c in connections]}")
        
        # Try to terminate (only if safe)
        if len(connections) < 5:  # Only if less than 5 connections
            print("\n⏱️  Attempting to terminate connections...")
            start = time.time()
            
            try:
                result = session.execute(text(f"""
                    SELECT pg_terminate_backend(pid) 
                    FROM pg_stat_activity 
                    WHERE datname = :db_name 
                    AND pid <> pg_backend_pid()
                """), {"db_name": db_name})
                
                terminated = result.fetchall()
                term_time = time.time() - start
                
                print(f"✅ Terminated in {term_time:.3f}s: {terminated}")
                
            except Exception as e:
                print(f"⚠️  Termination failed: {e}")
        else:
            print("⚠️  Too many connections to safely terminate. Skipping termination test.")
        
        session.close()
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_drop_create_cycle():
    """Test DROP/CREATE on a test table to measure timing."""
    print_section("TEST 2: DROP/CREATE Cycle Timing")
    
    try:
        engine = create_engine(
            SQLALCHEMY_CONN_STRING,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 10,
                "options": "-c statement_timeout=30000"
            }
        )
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create test table
        print("\n⏱️  Creating test table...")
        start = time.time()
        
        session.execute(text("""
            DROP TABLE IF EXISTS test_reset_cycle CASCADE
        """))
        session.commit()
        
        session.execute(text("""
            CREATE TABLE test_reset_cycle (
                id SERIAL PRIMARY KEY,
                data VARCHAR(100)
            )
        """))
        session.commit()
        
        create_time = time.time() - start
        print(f"✅ Test table created in {create_time:.3f}s")
        
        # Insert some data
        print("\n⏱️  Inserting test data...")
        start = time.time()
        
        for i in range(100):
            session.execute(text(
                "INSERT INTO test_reset_cycle (data) VALUES (:data)"
            ), {"data": f"test_{i}"})
        session.commit()
        
        insert_time = time.time() - start
        print(f"✅ Inserted 100 rows in {insert_time:.3f}s")
        
        # Drop the table
        print("\n⏱️  Dropping test table...")
        start = time.time()
        
        session.execute(text("DROP TABLE test_reset_cycle CASCADE"))
        session.commit()
        
        drop_time = time.time() - start
        print(f"✅ Table dropped in {drop_time:.3f}s")
        
        if drop_time > 5.0:
            print(f"⚠️  WARNING: DROP took {drop_time:.3f}s - might timeout on large tables!")
        
        session.close()
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_actual_reset():
    """Test the actual reset operation on production tables."""
    print_section("TEST 3: Actual Database Reset")
    
    # Import the actual reset function
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        from database.connection import reset_db
        
        print("\n⏱️  Executing reset_db()...")
        start = time.time()
        
        reset_db()
        
        reset_time = time.time() - start
        print(f"\n✅ reset_db() completed in {reset_time:.3f}s")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Could not import reset_db: {e}")
        print("   (Trying manual reset instead)")
        return test_manual_reset()
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_manual_reset():
    """Manual reset implementation to test the actual reset flow."""
    print_section("TEST 3: Manual Database Reset (Without Import)")
    
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
        
        print("\n⏱️  Step 1: Disposing connection pool...")
        start = time.time()
        engine.dispose()
        dispose_time = time.time() - start
        print(f"✅ Pool disposed in {dispose_time:.3f}s")
        
        # Reconnect
        session = Session()
        
        print("\n⏱️  Step 2: Terminating active connections...")
        start = time.time()
        
        db_name = SQLALCHEMY_CONN_STRING.split('/')[-1].split('?')[0]
        
        result = session.execute(text("""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = :db_name 
            AND pid <> pg_backend_pid()
        """), {"db_name": db_name})
        
        session.commit()
        term_time = time.time() - start
        print(f"✅ Connections terminated in {term_time:.3f}s")
        
        print("\n⏱️  Step 3: Getting list of tables...")
        start = time.time()
        
        result = session.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
        """))
        
        tables = [row[0] for row in result]
        list_time = time.time() - start
        print(f"✅ Found {len(tables)} tables in {list_time:.3f}s:")
        for t in tables:
            print(f"   📋 {t}")
        
        print("\n⏱️  Step 4: Dropping all tables...")
        start = time.time()
        
        # Drop all tables in sequence
        for table in tables:
            print(f"   Dropping {table}...", end='', flush=True)
            drop_start = time.time()
            
            session.execute(text(f"DROP TABLE IF EXISTS \"{table}\" CASCADE"))
            session.commit()
            
            drop_duration = time.time() - drop_start
            print(f" ({drop_duration:.3f}s)")
            
            if drop_duration > 10.0:
                print(f"      ⚠️  WARNING: Took {drop_duration:.3f}s!")
        
        drop_time = time.time() - start
        print(f"✅ All tables dropped in {drop_time:.3f}s")
        
        print("\n⏱️  Step 5: Resetting connection pool...")
        session.close()
        engine.dispose()
        reset_pool_time = time.time() - start
        print(f"✅ Pool reset in {reset_pool_time:.3f}s")
        
        total_time = dispose_time + term_time + list_time + drop_time + reset_pool_time
        print(f"\n✅ Total reset time: {total_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DATABASE RESET TESTING SUITE")
    print("=" * 80)
    
    # First diagnose
    print("\n📋 PHASE 1: DIAGNOSIS")
    blockers_ok = diagnose_reset_blockers()
    
    if not blockers_ok:
        print("\n⚠️  Reset blockers detected! Fix them before reset will work:")
        print("   - Kill idle in transaction connections")
        print("   - Wait for long-running queries to complete")
        print("   - Remove any application connections holding locks")
        return
    
    # Then test individual components
    print("\n📋 PHASE 2: COMPONENT TESTS")
    results = {}
    results['Connection Termination'] = test_connection_termination()
    results['DROP/CREATE Cycle'] = test_drop_create_cycle()
    
    # Finally test actual reset
    print("\n📋 PHASE 3: FULL RESET TEST")
    results['Actual Reset'] = test_actual_reset()
    
    # Summary
    print_section("TEST SUMMARY")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above for details.")


if __name__ == "__main__":
    main()
