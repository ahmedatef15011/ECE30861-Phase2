# Browser-Based Interface Documentation

## Overview

The ML Model Registry provides a comprehensive **browser-based interface** for interacting with the API without writing code. This allows users to explore, test, and manage ML artifacts entirely through their web browser.

**Production URL**: https://vmqqvhwppq.us-east-1.awsapprunner.com

---

## Interface Components

### 1. Swagger UI (Interactive API Explorer)

**URL**: https://vmqqvhwppq.us-east-1.awsapprunner.com/docs

**Purpose**: Interactive API testing and exploration

#### Features
- **Visual API Documentation**: All endpoints grouped by category
- **Try It Out**: Execute API calls directly from browser
- **Built-in Authentication**: Set JWT token once, use across all requests
- **Request Builder**: Fill forms instead of writing JSON
- **Response Viewer**: Formatted JSON responses with syntax highlighting
- **Schema Explorer**: Browse data models and types

#### Screenshot Walkthrough

**Home Page**:
```
┌──────────────────────────────────────────────────────────┐
│ ML Model Registry API                           [Swagger] │
├──────────────────────────────────────────────────────────┤
│                                                            │
│ ▼ Authentication                                          │
│   PUT /authenticate                                        │
│   POST /register                                           │
│                                                            │
│ ▼ Artifacts                                               │
│   POST /artifact/ingest                                    │
│   GET /artifact/{type}/{id}                               │
│   GET /artifact/{type}/{id}/lineage                       │
│   POST /artifact/{type}/{id}/malicious                    │
│                                                            │
│ ▼ Packages                                                │
│   GET /packages                                            │
│   POST /package                                            │
│   DELETE /package/{id}                                     │
│                                                            │
│ ▼ Admin                                                   │
│   GET /admin/users                                         │
│   POST /admin/reset                                        │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

#### How to Use Swagger UI

**Step 1: Authenticate**

1. Click on `PUT /authenticate`
2. Click "Try it out"
3. Fill in request body:
```json
{
  "user": {
    "name": "ece30861defaultadminuser",
    "is_admin": true
  },
  "secret": {
    "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"
  }
}
```
4. Click "Execute"
5. Copy token from response (e.g., `bearer eyJhbGci...`)
6. Click "Authorize" button at top
7. Paste token → Click "Authorize" → Close

**Step 2: Upload an Artifact**

1. Click on `POST /artifact/ingest`
2. Click "Try it out"
3. Fill in:
```json
{
  "type": "model",
  "artifact": {
    "Name": "test-model",
    "URL": "https://huggingface.co/bert-base-uncased"
  }
}
```
4. Click "Execute"
5. Response shows artifact ID (e.g., `123`)

**Step 3: Search Packages**

1. Click on `GET /packages`
2. Click "Try it out"
3. Enter query: `bert`
4. Select type: `model`
5. Click "Execute"
6. View list of matching packages

**Step 4: View Lineage**

1. Click on `GET /artifact/{type}/{id}/lineage`
2. Click "Try it out"
3. Enter type: `model`, id: `13`
4. Click "Execute"
5. View lineage graph:
```json
{
  "nodes": [
    {"id": 13, "name": "resnet-50"},
    {"id": 14, "name": "trained-gender"},
    {"id": 15, "name": "trained-gender-onnx"}
  ],
  "edges": [
    {"from": 13, "to": 14},
    {"from": 14, "to": 15}
  ]
}
```

**Step 5: Download Artifact**

1. Click on `GET /artifact/{type}/{id}`
2. Click "Try it out"
3. Enter type: `model`, id: `13`
4. Click "Execute"
5. Click "Download file" in response

---

### 2. ReDoc (Searchable Documentation)

**URL**: https://vmqqvhwppq.us-east-1.awsapprunner.com/redoc

**Purpose**: Comprehensive, searchable API reference

#### Features
- **Three-Panel Layout**: Navigation | Content | Examples
- **Search**: Full-text search across all endpoints
- **Organized Hierarchy**: Tags, operations, schemas
- **Code Samples**: Example requests/responses
- **Schema Details**: Complete data models with descriptions

#### Navigation Structure

```
┌─ Navigation ─────┬─ Content ──────────────┬─ Examples ─┐
│                  │                        │            │
│ Authentication   │ PUT /authenticate      │ Request:   │
│ Artifacts        │                        │ {          │
│ Packages         │ Description:           │   "user":  │
│ Admin            │ Authenticates user     │   {...}    │
│                  │ and returns JWT token  │ }          │
│ Schemas          │                        │            │
│ - Package        │ Parameters:            │ Response:  │
│ - Artifact       │ - user (object)        │ "bearer    │
│ - Lineage        │ - secret (object)      │  eyJ..."   │
│                  │                        │            │
└──────────────────┴────────────────────────┴────────────┘
```

#### How to Use ReDoc

**Search Example**:
1. Click search box (top right)
2. Type "lineage"
3. Results show:
   - `GET /artifact/{type}/{id}/lineage`
   - `LineageNode` schema
   - `LineageEdge` schema

**Browse Example**:
1. Click "Artifacts" in left sidebar
2. Scroll to see all artifact endpoints
3. Click on endpoint for details
4. Right panel shows example request/response

---

### 3. Health Dashboard

**URL**: https://vmqqvhwppq.us-east-1.awsapprunner.com/health

**Purpose**: System status monitoring

#### Response Format

```json
{
  "status": "healthy",
  "timestamp": "2024-12-14T10:30:00Z",
  "services": {
    "api": "online",
    "database": {
      "status": "connected",
      "latency_ms": 15
    },
    "s3": {
      "status": "connected",
      "bucket": "ml-registry-artifacts"
    },
    "bedrock": {
      "status": "available",
      "model": "claude-3-haiku"
    }
  },
  "metrics": {
    "total_artifacts": 30,
    "total_packages": 45,
    "disk_usage_mb": 1250
  }
}
```

#### Use Cases
- **Monitoring**: Check if API is up
- **Debugging**: Verify service connectivity
- **Metrics**: Quick stats on system usage

---

## Complete Workflows via Browser

### Workflow 1: Upload & Search

**Goal**: Upload a model and find it via search

**Steps**:
1. Navigate to https://vmqqvhwppq.us-east-1.awsapprunner.com/docs
2. Authenticate (see Step 1 above)
3. POST `/artifact/ingest`:
   ```json
   {
     "type": "model",
     "artifact": {
       "Name": "my-bert-model",
       "URL": "https://huggingface.co/my-bert"
     }
   }
   ```
4. Note artifact ID from response (e.g., `42`)
5. GET `/packages?query=my-bert&type=model`
6. Verify model appears in search results

**Time**: ~2 minutes

---

### Workflow 2: Lineage Exploration

**Goal**: Explore fine-tuning lineage chain

**Steps**:
1. Navigate to Swagger UI
2. Authenticate
3. GET `/artifact/model/13/lineage`
4. View response:
   ```json
   {
     "nodes": [
       {"id": 13, "name": "resnet-50", "type": "base_model"},
       {"id": 14, "name": "trained-gender", "type": "fine_tuned"},
       {"id": 15, "name": "trained-gender-onnx", "type": "converted"}
     ],
     "edges": [
       {"from": 13, "to": 14, "relationship": "fine_tuned_from"},
       {"from": 14, "to": 15, "relationship": "converted_from"}
     ]
   }
   ```
5. Click on parent ID (13) in response
6. GET `/artifact/model/13` to see parent details
7. Repeat for entire lineage chain

**Time**: ~3 minutes

---

### Workflow 3: User Management (Admin)

**Goal**: Create new user and manage permissions

**Steps**:
1. Authenticate as admin
2. POST `/register`:
   ```json
   {
     "user": {"name": "new_user", "is_admin": false},
     "secret": {"password": "secure_password"}
   }
   ```
3. GET `/admin/users` to verify creation
4. Logout (clear token)
5. Authenticate as new user
6. Try admin endpoint → Receive 403 Forbidden (correct)

**Time**: ~4 minutes

---

### Workflow 4: Quality Metrics Review

**Goal**: Review quality scores for multiple artifacts

**Steps**:
1. Authenticate
2. GET `/packages?type=model` → Get all models
3. For each model ID:
   - GET `/artifact/model/{id}`
   - Note metrics:
     ```json
     {
       "NetScore": 0.85,
       "BusFactor": 0.80,
       "CodeQuality": 0.90,
       "License": 1.0,
       ...
     }
     ```
4. Compare scores across models
5. Identify high-quality candidates (NetScore > 0.8)

**Time**: ~5 minutes for 10 models

---

## Browser Compatibility

### Tested Browsers

| Browser | Version | Swagger UI | ReDoc | Health |
|---------|---------|------------|-------|--------|
| Chrome | 120+ | ✅ Full | ✅ Full | ✅ |
| Firefox | 121+ | ✅ Full | ✅ Full | ✅ |
| Safari | 17+ | ✅ Full | ✅ Full | ✅ |
| Edge | 120+ | ✅ Full | ✅ Full | ✅ |
| Mobile Safari | iOS 17+ | ⚠️ Limited | ✅ | ✅ |
| Mobile Chrome | Android 13+ | ⚠️ Limited | ✅ | ✅ |

**Notes**:
- Swagger UI "Try It Out" works best on desktop
- ReDoc fully responsive on mobile
- Health endpoint JSON works on all devices

---

## Authentication in Browser

### Setting Authorization Header

**Swagger UI Method** (Recommended):
1. Get token from `/authenticate` response
2. Click "Authorize" button (top right)
3. Enter: `bearer <your_token_here>`
4. All subsequent requests automatically include header

**Manual Method** (ReDoc, curl):
```bash
# Copy token from Swagger UI
TOKEN="bearer eyJhbGci..."

# Use in curl (can be run from browser DevTools console)
curl -X GET "https://vmqqvhwppq.us-east-1.awsapprunner.com/packages" \
  -H "X-Authorization: $TOKEN"
```

---

## Advanced Features

### 1. Schema Exploration

**Location**: Swagger UI bottom section or ReDoc "Schemas" tab

**Available Schemas**:
- `Package`: Core artifact metadata
- `PackageMetrics`: Quality scores
- `Lineage`: Graph nodes/edges
- `User`: Authentication user
- `ArtifactIngestRequest`: Upload format

**Use Case**: Understand exact JSON structure before making requests

---

### 2. Response Filtering

**Swagger UI**: Click on response field names to expand/collapse

**Example**: Large package list
1. GET `/packages` returns 100 items
2. Click on array items to view one at a time
3. Use browser's Find (Cmd+F) to search within response

---

### 3. Download Artifact Files

**Method**: Swagger UI "Download file" button

**Process**:
1. GET `/artifact/model/13`
2. Response includes file data
3. Click "Download file" button in Swagger UI
4. Browser saves artifact to Downloads folder

**Supported Formats**: ZIP, TAR, JSON, YAML, ONNX

---

## Troubleshooting

### Issue: "401 Unauthorized" on All Requests

**Solution**:
1. Verify you've authenticated (PUT `/authenticate`)
2. Check token was copied correctly (no extra spaces)
3. Click "Authorize" in Swagger UI and paste token
4. Token format must be: `bearer eyJhbGci...` (include "bearer ")

---

### Issue: Swagger UI Not Loading

**Solution**:
1. Check network connection
2. Verify URL: https://vmqqvhwppq.us-east-1.awsapprunner.com/docs
3. Check browser console (F12) for errors
4. Try ReDoc instead: .../redoc
5. Clear browser cache

---

### Issue: Request Returns 422 Validation Error

**Solution**:
1. Check request body matches schema
2. In Swagger UI, click "Schema" tab to see required fields
3. Verify data types (strings in quotes, numbers without)
4. Example errors:
   ```json
   {
     "detail": [
       {
         "loc": ["body", "artifact", "URL"],
         "msg": "field required",
         "type": "value_error.missing"
       }
     ]
   }
   ```
5. Add missing `URL` field

---

## Mobile Usage

### Responsive Features

**ReDoc** (Fully Mobile-Friendly):
- Hamburger menu for navigation
- Tap to expand sections
- Pinch to zoom code examples
- Copy button for sample code

**Swagger UI** (Limited):
- View documentation works
- "Try It Out" difficult on small screens
- Recommendation: Use desktop for testing

**Health Dashboard**:
- JSON response viewable on mobile
- Use JSON viewer browser extension for formatting

---

## Accessibility

### Keyboard Navigation

- **Tab**: Move between interactive elements
- **Enter**: Expand/collapse sections
- **Arrow keys**: Navigate dropdowns
- **Escape**: Close modals

### Screen Reader Support

- All buttons have aria-labels
- Response regions marked as "live"
- Error messages announced

---

## Screenshots

### Swagger UI Main Page
```
╔══════════════════════════════════════════════════════╗
║ ML Model Registry API                    [Authorize] ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║ ▼ Authentication                                     ║
║   PUT /authenticate    [Try it out]                  ║
║   POST /register       [Try it out]                  ║
║                                                      ║
║ ▼ Artifacts                                          ║
║   POST /artifact/ingest                              ║
║   GET /artifact/{type}/{id}                          ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

### Authentication Modal
```
╔══════════════════════════════════════╗
║ Available authorizations             ║
╠══════════════════════════════════════╣
║                                      ║
║ X-Authorization (apiKey)             ║
║                                      ║
║ Value: [bearer eyJhbG...          ] ║
║                                      ║
║        [Authorize]  [Close]          ║
║                                      ║
╚══════════════════════════════════════╝
```

---

## API Endpoint Summary

### Viewable in Browser

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/docs` | GET | Swagger UI | No |
| `/redoc` | GET | ReDoc documentation | No |
| `/health` | GET | System health | No |
| `/openapi.json` | GET | OpenAPI spec | No |
| `/authenticate` | PUT | Login | No |
| `/artifact/ingest` | POST | Upload artifact | Yes |
| `/artifact/{type}/{id}` | GET | Download artifact | Yes |
| `/artifact/{type}/{id}/lineage` | GET | View lineage | Yes |
| `/packages` | GET | Search packages | Yes |
| `/admin/users` | GET | List users | Yes (Admin) |
| `/admin/reset` | POST | Reset database | Yes (Admin) |

---

## Conclusion

The ML Model Registry provides a **fully functional browser-based interface** that allows users to:

✅ **Explore** all API endpoints with interactive documentation  
✅ **Test** API calls without writing code  
✅ **Authenticate** and manage sessions  
✅ **Upload** artifacts via web forms  
✅ **Search** and download packages  
✅ **Visualize** lineage relationships  
✅ **Monitor** system health  
✅ **Manage** users and permissions  

**No command-line or programming knowledge required** – everything is accessible through standard web browsers.

**Primary Interface**: https://vmqqvhwppq.us-east-1.awsapprunner.com/docs
