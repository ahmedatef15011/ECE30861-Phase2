#!/usr/bin/env python
"""Quick test of POST /artifacts with exact name matching."""

from fastapi.testclient import TestClient
from src.api.main import create_app
from src.database.connection import SessionLocal
from src.database.models import Package

# Create test client
app = create_app()
client = TestClient(app)

# Reset the system
print("1. Resetting system...")
response = client.delete("/reset")
print(f"   Reset response: {response.status_code}")

# Ingest an artifact with a specific test name
print("\n2. Ingesting test artifact...")
test_name = "Get Artifact By Name Test Artifact 0"
response = client.post(
    "/artifact/model",
    json={"url": "https://huggingface.co/google-bert/bert-base-uncased"}
)
print(f"   Ingest response: {response.status_code}")
if response.status_code == 201:
    data = response.json()
    print(f"   Created artifact: {data['metadata']}")
    # Get package from DB to update name
    db = SessionLocal()
    pkg = db.query(Package).first()
    if pkg:
        pkg.name = test_name
        db.commit()
        print(f"   Updated name to: {test_name}")
    db.close()
else:
    print(f"   Error: {response.json()}")

# Now query by exact name using POST /artifacts
print(f"\n3. Querying POST /artifacts with exact name: '{test_name}'")
response = client.post(
    "/artifacts",
    json=[{"name": test_name}]
)
print(f"   Query response: {response.status_code}")
print(f"   Response: {response.json()}")

if response.status_code == 200:
    data = response.json()
    if data and len(data) > 0:
        if data[0]['name'] == test_name:
            print(f"   ✅ SUCCESS: Found exact match!")
        else:
            print(f"   ❌ MISMATCH: Got {data[0]['name']} instead of {test_name}")
    else:
        print(f"   ❌ EMPTY: No results returned")
else:
    print(f"   ❌ ERROR: {response.json()}")

# Test with wildcard
print(f"\n4. Testing wildcard name='*'")
response = client.post(
    "/artifacts",
    json=[{"name": "*"}]
)
print(f"   Response: {response.status_code}")
print(f"   Found {len(response.json())} total artifacts")
