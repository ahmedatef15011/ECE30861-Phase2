"""Test script for AWS backend - ingest and retrieve artifacts."""
import requests
import json
import time

BASE_URL = "https://vmqqvhwppq.us-east-1.awsapprunner.com"


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_aws_backend():
    """Test ingest, get by ID, and get by name on AWS backend."""
    
    print_section("STEP 0: Authenticate with Default Admin User")
    auth_data = {
        "user": {
            "name": "ece30861defaultadminuser",
            "is_admin": True
        },
        "secret": {
            "password": (
                "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE "
                "packages;"
            )
        }
    }
    
    try:
        response = requests.put(
            f"{BASE_URL}/authenticate",
            json=auth_data,
            timeout=30
        )
        print(f"Status: {response.status_code}")
        auth_response = response.json()
        print(f"Response: {json.dumps(auth_response, indent=2)}")
        
        if response.status_code == 200:
            if (isinstance(auth_response, dict) and
                    "access_token" in auth_response):
                token = auth_response["access_token"]
                print("✅ Authentication successful")
                print(f"   Token: {token[:50]}...")
            else:
                print("⚠️  No token in response")
        else:
            auth_status = response.status_code
            print(f"❌ Authentication failed: {auth_status}")
    except Exception as e:
        print(f"❌ Authentication request failed: {e}")

    time.sleep(2)
    
    print_section("STEP 1: Reset System")
    try:
        response = requests.delete(f"{BASE_URL}/reset", timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Reset successful")
        else:
            print(f"⚠️  Reset returned {response.status_code}")
    except Exception as e:
        print(f"⚠️  Reset failed: {e}")
    
    time.sleep(2)
    
    print_section("STEP 2: Ingest BERT Model")
    ingest_data = {
        "url": "https://huggingface.co/google-bert/bert-base-uncased"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/artifact/model",
            json=ingest_data,
            timeout=120
        )
        print(f"Status: {response.status_code}")
        artifact = response.json()
        print(f"Response: {json.dumps(artifact, indent=2)}")
        
        if response.status_code in [200, 201, 424]:
            if isinstance(artifact, dict) and "metadata" in artifact:
                artifact_name = artifact["metadata"]["name"]
                artifact_id = artifact["metadata"]["id"]
                artifact_type = artifact["metadata"]["type"]
                print(f"\n✅ Ingested: {artifact_name}")
                print(f"   ID: {artifact_id}")
                print(f"   Type: {artifact_type}")
            else:
                print("⚠️  Could not parse artifact metadata")
                artifact_id = "1"
                artifact_name = "bert-base-uncased"
        else:
            print(f"❌ Ingest failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ingest request failed: {e}")
        return False
    
    time.sleep(2)
    
    print_section("STEP 3: Get Artifact By ID")
    try:
        response = requests.get(
            f"{BASE_URL}/artifacts/model/{artifact_id}",
            timeout=30
        )
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            artifact = response.json()
            print(f"Response: {json.dumps(artifact, indent=2)}")
            print("\n✅ Retrieved by ID")
        else:
            print(f"❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Get by ID failed: {e}")
    
    time.sleep(2)
    
    print_section("STEP 4: Get Artifact By Name - DEBUG")
    print(f"\n🔍 Testing byName with: '{artifact_name}'")
    print(f"   Artifact name extracted from ingest: {artifact_name}")
    print(f"   Name length: {len(artifact_name)}")
    print(f"   Name bytes: {artifact_name.encode('utf-8')}")
    
    # Test 1: Exact name
    test_names = [
        artifact_name,                           # Exact from response
        artifact_name.lower(),                   # Lowercase
        artifact_name.upper(),                   # Uppercase
        "bert-base-uncased",                     # Hardcoded
        "google-bert/bert-base-uncased",         # Full path
    ]
    
    for test_name in test_names:
        print(f"\n   Trying: '{test_name}'")
        try:
            url = f"{BASE_URL}/artifact/byName/{test_name}"
            print(f"   URL: {url}")
            response = requests.get(url, timeout=30)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                results = response.json()
                count = (len(results) if isinstance(results, list)
                         else 1)
                print(f"   ✅ SUCCESS - Found {count} artifact(s)")
                print(f"   Response: {json.dumps(results, indent=6)}")
                break
            elif response.status_code == 404:
                print("   ❌ Not found (404)")
                try:
                    print(f"   Detail: {response.json()}")
                except Exception:
                    print(f"   Response: {response.text[:200]}")
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print_section("STEP 5: Query All Artifacts (Verify Data)")
    try:
        response = requests.post(
            f"{BASE_URL}/artifacts",
            json=[{"name": "*"}],
            timeout=30
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            artifacts = response.json()
            print(f"Total artifacts in database: {len(artifacts)}")
            print(f"Response: {json.dumps(artifacts, indent=2)}")
            for a in artifacts:
                if isinstance(a, dict):
                    name_val = a.get('name')
                    if not name_val:
                        meta = a.get('metadata', {})
                        if isinstance(meta, dict):
                            name_val = meta.get('name')
                    print(f"  - Name: '{name_val}'")
        else:
            print(f"Failed: {response.status_code}")
    except Exception as e:
        print(f"Query failed: {e}")
    
    print_section("SUMMARY")
    print("✅ AWS Backend Test Completed")


if __name__ == "__main__":
    print("\n🌐 " + "="*68)
    print("   AWS BACKEND TEST - byName DEBUGGING")
    print(f"   Target: {BASE_URL}")
    print("🌐 " + "="*68)
    
    try:
        test_aws_backend()
    except requests.exceptions.ConnectionError as e:
        print("\\n❌ ERROR: Cannot connect to AWS backend")
        print(f"   URL: {BASE_URL}")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
