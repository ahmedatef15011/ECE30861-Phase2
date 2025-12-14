"""Main FastAPI application factory."""

import os
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from src.api.config import settings
from src.api.routes import users, packages, ratings, system, llm
from src.database.connection import init_db, reset_db
from src.database.init_db import create_default_user
from src.api.dependencies import get_db, get_optional_user, get_current_user, validate_id
from src.database.models import Package, User
from src.database import crud
from src.lineage import LineageExtractor
from src.hf_api import HuggingFaceAPI
from src.models import ParsedURL, URLCategory

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Optional S3 storage support
try:
    from src.storage_s3 import get_s3_storage
    from src.model_downloader import ModelDownloader
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logger.warning("S3 storage not available - boto3 not installed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup: Initialize database and create default user
    logger.info("=" * 70)
    logger.info("🚀 STARTING ML REGISTRY API")
    logger.info("=" * 70)
    
    # Log database configuration
    from src.database.connection import DATABASE_URL
    if DATABASE_URL.startswith("postgresql"):
        db_info = DATABASE_URL.split("@")[1] if "@" in DATABASE_URL else "RDS"
        logger.info(f"� Database: PostgreSQL (RDS) - {db_info}")
        logger.info("   ✅ Persistent storage enabled")
    elif DATABASE_URL.startswith("sqlite"):
        logger.warning("⚠️  Database: SQLite (ephemeral)")
        logger.warning("   ❌ Data will be lost on restart!")
    
    # Log S3 configuration
    s3_enabled = os.getenv("ENABLE_S3_STORAGE", "false").lower() == "true"
    if s3_enabled:
        bucket = os.getenv("S3_BUCKET_NAME", "unknown")
        logger.info(f"☁️  S3 Storage: ENABLED - bucket={bucket}")
    else:
        logger.info("☁️  S3 Storage: DISABLED (using local storage)")
    
    logger.info("Initializing database tables...")
    init_db()
    logger.info("✅ Tables created/verified")
    
    # Run database migration for usage_count column if needed
    logger.info("Running database migrations...")
    try:
        from migrate_add_usage_count import migrate_usage_count
        migrate_usage_count()
        logger.info("✅ Migrations complete")
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        logger.error(
            "Authentication may not work! Run migrate_add_usage_count.py "
            "manually."
        )
    
    create_default_user()
    logger.info("✅ Database initialized with default admin user")
    logger.info("=" * 70)
    
    yield
    
    # Shutdown: Clean up resources if needed
    logger.info("=" * 70)
    logger.info("🛑 SHUTTING DOWN ML REGISTRY API")
    logger.info("=" * 70)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.PROJECT_VERSION,
        description="A Trustworthy Model Registry for Machine Learning Models",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all HTTP requests with timing and response status."""
        start_time = time.time()
        request_id = f"{int(start_time * 1000)}"
        
        # Extract client info
        client_ip = request.client.host if request.client else "unknown"
        
        # Log incoming request
        logger.info(
            f"📥 [{request_id}] {request.method} {request.url.path} "
            f"from {client_ip}"
        )
        
        # Log query parameters if present
        if request.query_params:
            logger.info(f"   Query params: {dict(request.query_params)}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response with appropriate emoji
            status_emoji = "✅" if response.status_code < 400 else "❌"
            logger.info(
                f"{status_emoji} [{request_id}] {request.method} "
                f"{request.url.path} → {response.status_code} "
                f"({duration:.3f}s)"
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"💥 [{request_id}] {request.method} {request.url.path} "
                f"→ EXCEPTION ({duration:.3f}s): {str(e)}"
            )
            raise
    
    # Include routers
    app.include_router(
        users.router,
        prefix=f"{settings.API_V1_PREFIX}/user",
        tags=["users"]
    )
    
    app.include_router(
        packages.router,
        prefix=f"{settings.API_V1_PREFIX}/package",
        tags=["packages"]
    )
    
    app.include_router(
        ratings.router,
        prefix=f"{settings.API_V1_PREFIX}/package",
        tags=["ratings"]
    )
    
    app.include_router(
        system.router,
        prefix=f"{settings.API_V1_PREFIX}/system",
        tags=["system"]
    )
    
    # LLM endpoints (AWS Bedrock)
    app.include_router(
        llm.router,
        prefix=f"{settings.API_V1_PREFIX}/llm",
        tags=["llm"]
    )
    
    # Root endpoint
    @app.get("/", tags=["root"])
    def root():
        """Root endpoint with API information."""
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.PROJECT_VERSION,
            "docs": "/docs",
            "health": "/health",
            "frontend": "/frontend"
        }
    
    # Frontend endpoint for autograder
    @app.get("/frontend", response_class=HTMLResponse, tags=["frontend"])
    def frontend():
        """
        Simple frontend interface for the ML Model Registry.
        Provides a basic HTML page for interacting with the API.
        """
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ML Model Registry - Frontend</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }
                .container {
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    max-width: 800px;
                    width: 100%;
                    padding: 40px;
                }
                h1 {
                    color: #667eea;
                    margin-bottom: 10px;
                    font-size: 2.5em;
                }
                .subtitle {
                    color: #666;
                    margin-bottom: 30px;
                    font-size: 1.1em;
                }
                .section {
                    margin: 30px 0;
                }
                .section h2 {
                    color: #333;
                    margin-bottom: 15px;
                    font-size: 1.5em;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }
                .links {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }
                .link-card {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    text-decoration: none;
                    transition: transform 0.2s, box-shadow 0.2s;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    text-align: center;
                }
                .link-card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
                }
                .link-card h3 {
                    font-size: 1.2em;
                    margin-bottom: 10px;
                }
                .link-card p {
                    font-size: 0.9em;
                    opacity: 0.9;
                }
                .status {
                    display: inline-block;
                    background: #10b981;
                    color: white;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-size: 0.9em;
                    font-weight: bold;
                }
                .info-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }
                .info-card {
                    background: #f3f4f6;
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                }
                .info-card strong {
                    display: block;
                    color: #667eea;
                    font-size: 2em;
                    margin-bottom: 5px;
                }
                .info-card span {
                    color: #666;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 ML Model Registry</h1>
                <p class="subtitle">Trustworthy Machine Learning Model Management Platform</p>
                <p><span class="status">✓ System Online</span></p>
                
                <div class="section">
                    <h2>📊 Quick Stats</h2>
                    <div class="info-grid">
                        <div class="info-card">
                            <strong>8</strong>
                            <span>Quality Metrics</span>
                        </div>
                        <div class="info-card">
                            <strong>REST</strong>
                            <span>API Type</span>
                        </div>
                        <div class="info-card">
                            <strong>JWT</strong>
                            <span>Auth Method</span>
                        </div>
                        <div class="info-card">
                            <strong>AWS</strong>
                            <span>Cloud Provider</span>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔗 API Resources</h2>
                    <div class="links">
                        <a href="/docs" class="link-card">
                            <h3>📚 API Docs</h3>
                            <p>Interactive Swagger UI</p>
                        </a>
                        <a href="/redoc" class="link-card">
                            <h3>📖 ReDoc</h3>
                            <p>Alternative API docs</p>
                        </a>
                        <a href="/health" class="link-card">
                            <h3>💚 Health</h3>
                            <p>System status</p>
                        </a>
                        <a href="/tracks" class="link-card">
                            <h3>🛤️ Tracks</h3>
                            <p>Feature tracks</p>
                        </a>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🎯 Features</h2>
                    <ul style="list-style: none; padding-left: 0;">
                        <li style="padding: 10px 0; border-bottom: 1px solid #eee;">✓ User Authentication & Authorization (JWT)</li>
                        <li style="padding: 10px 0; border-bottom: 1px solid #eee;">✓ Package Upload & Management</li>
                        <li style="padding: 10px 0; border-bottom: 1px solid #eee;">✓ Quality Scoring (8 Metrics)</li>
                        <li style="padding: 10px 0; border-bottom: 1px solid #eee;">✓ HuggingFace Model Ingestion</li>
                        <li style="padding: 10px 0; border-bottom: 1px solid #eee;">✓ Regex Search & Filtering</li>
                        <li style="padding: 10px 0;">✓ System Health Monitoring</li>
                    </ul>
                </div>
                
                <div class="section" style="text-align: center; padding-top: 20px; border-top: 2px solid #eee;">
                    <p style="color: #666;">
                        <strong>Team 20</strong> | Ahmed Elbehiry • Zeyad Elshafey • Omar Ahmed • Jacob Walter
                    </p>
                    <p style="color: #999; margin-top: 10px; font-size: 0.9em;">
                        ECE30861 - Software Engineering | Purdue University
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    # Health endpoint for autograder/monitoring (public, no auth required)
    @app.get("/health", tags=["health"])
    def health_check():
        """
        Simple health check endpoint for monitoring and autograder.
        Always returns 200 OK with status 'ok'.
        This is a public endpoint with no authentication required.
        """
        return {"status": "ok"}
    
    # Tracks endpoint for autograder (public, no auth required)
    @app.get("/tracks", tags=["tracks"])
    def get_tracks():
        """
        Return available feature tracks implemented in the system.
        This endpoint is used by the autograder to verify implemented features.
        
        Implemented Tracks:
        - Access control track: Complete user authentication and authorization
          * User authentication with JWT tokens
          * Token storage and validation in database
          * Usage tracking (1000-interaction limit per token)
          * Protected endpoints requiring authentication
          * Admin-only endpoints with role-based access control
        
        Returns:
            Object with plannedTracks array matching OpenAPI spec enum values
        """
        return {
            "plannedTracks": ["Access control track"]
        }
    
    # Authentication endpoint (OpenAPI spec)
    class AuthenticationUser(BaseModel):
        """User info for authentication."""
        name: str
        is_admin: bool

    class AuthenticationSecret(BaseModel):
        """Secret info for authentication."""
        password: str

    class AuthenticationRequest(BaseModel):
        """Authentication request per OpenAPI spec."""
        user: AuthenticationUser
        secret: AuthenticationSecret

    @app.put("/authenticate", tags=["authentication"])
    def authenticate_user(
        auth_request: AuthenticationRequest,
        db: Session = Depends(get_db)
    ):
        """
        Authenticate user and return JWT token (OpenAPI spec).
        
        Args:
            auth_request: Authentication request with user and secret
            db: Database session
            
        Returns:
            JWT token as a string (e.g., "bearer eyJ...")
            
        Raises:
            HTTPException 401: Invalid credentials
            HTTPException 400: Missing fields
        """
        from src.database import crud
        from src.auth.password_hash import verify_password
        from src.auth.jwt_handler import create_access_token
        from datetime import timedelta
        
        # Log the authentication attempt with password for debugging
        logger.info(f"🔐 AUTH ATTEMPT: user={auth_request.user.name}, is_admin={auth_request.user.is_admin}")
        logger.info(f"🔑 PASSWORD RECEIVED: {auth_request.secret.password} (length: {len(auth_request.secret.password)})")
        
        # Get user from database
        user = crud.get_user_by_username(db, auth_request.user.name)
        
        # Verify user exists and password is correct
        if not user or not verify_password(auth_request.secret.password, user.hashed_password):
            raise HTTPException(
                status_code=401,
                detail="The user or password is invalid."
            )
        
        # Create access token
        expires_delta = timedelta(hours=10)
        access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=expires_delta
        )
        
        # Save token to database for usage tracking
        expires_at = datetime.utcnow() + expires_delta
        crud.create_auth_token(
            db=db,
            user_id=user.id,
            token=access_token,
            expires_at=expires_at
        )
        
        # Return token as JSON string per OpenAPI spec example
        token_response = f"bearer {access_token}"
        logger.info(
            f"✅ AUTH SUCCESS: Returning token (length: {len(token_response)})"
        )
        return token_response
    
    # Reset endpoint - requires authentication
    @app.delete("/reset", tags=["system"])
    def reset_system(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
    ):
        """
        Reset the system to default state (empty registry).
        This endpoint is used by the autograder to reset the system
        between tests. Requires authentication.
        
        WARNING: This deletes all data including packages, scores, and users
        except for the default admin user which is recreated.
        
        Returns:
            Success message confirming system reset
        """
        # Check if user is admin (authentication is now required)
        if not current_user.is_admin:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to reset the registry"
            )

        logger.info("🔄 AUTOGRADER RESET: Resetting entire system...")
        
        try:
            # Clean up physical storage artifacts
            import shutil
            from pathlib import Path
            storage_path = Path("storage/artifacts")
            if storage_path.exists():
                # Remove all model files but keep the directory structure
                models_path = storage_path / "models"
                if models_path.exists():
                    shutil.rmtree(models_path)
                    models_path.mkdir(parents=True, exist_ok=True)
                
                # Reset metadata file
                metadata_file = storage_path / "metadata.json"
                if metadata_file.exists():
                    metadata_file.write_text("{}")
            
            logger.info("   Dropping and recreating database tables...")
            # Reset database (drops all tables and recreates them)
            reset_db()
            
            logger.info("   Creating default admin user...")
            # Recreate the default admin user
            create_default_user()
            
            logger.info("✅ RESET COMPLETE: System is now in default state")
            
            return {
                "message": "Registry is reset",
            }
        except Exception as e:
            logger.error(f"❌ RESET FAILED: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "message": f"System reset failed: {str(e)}",
                }
            )
    
    # Artifacts query endpoint (OpenAPI spec)
    # Schemas
    class ArtifactQuery(BaseModel):
        """Query for artifacts."""
        name: str
        types: Optional[list[str]] = None

    class ArtifactMetadata(BaseModel):
        """Artifact metadata response."""
        name: str
        id: int  # Integer per OpenAPI spec
        type: str

    @app.post(
        "/artifacts",
        tags=["artifacts"]
    )
    def query_artifacts(
        queries: list[ArtifactQuery],
        offset: Optional[str] = Query(None),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
    ):
        """
        Query artifacts from the registry.
        Use name="*" to list all artifacts.
        Tries exact name match first (per autograder expectations) then
        falls back to regex search if no exact results are found.
        
        Args:
            queries: List of artifact queries
            offset: Pagination offset
            db: Database session
            
        Returns:
            List of matching artifact metadata with offset header
        """
        from src.database.models import Package
        from fastapi.responses import JSONResponse
        
        logger.info(f"📋 POST /artifacts: Processing {len(queries)} query(ies)")
        for i, q in enumerate(queries, 1):
            logger.info(f"   Query {i}: name='{q.name}', types={q.types}")
        
        results = []
        
        for query in queries:
            if query.name == "*":
                # List all artifacts
                logger.info("   🔍 Wildcard query: listing all artifacts")
                packages = crud.get_packages(db, skip=0, limit=1000)
                logger.info(f"   ✓ Found {len(packages)} total artifact(s)")
                for pkg in packages:
                    artifact_type = getattr(pkg, 'artifact_type', 'model')
                    # Filter by types if specified
                    if query.types and artifact_type not in query.types:
                        continue
                    results.append(
                        ArtifactMetadata(
                            name=pkg.name,
                            id=pkg.id,  # Integer, not string
                            type=artifact_type
                        )
                    )
            else:
                # Try exact match first so we do not return substring hits
                logger.info(f"   🔍 Searching for exact match: '{query.name}'")
                # Execute exact match query for this specific artifact
                if query.types:
                    packages = (
                        db.query(Package)
                        .filter(
                            Package.name == query.name,
                            Package.artifact_type.in_(query.types)
                        )
                        .all()
                    )
                else:
                    packages = (
                        db.query(Package)
                        .filter(
                            Package.name == query.name
                        )
                        .all()
                    )
                if packages:
                    logger.info(
                        f"   ✓ Exact match: {len(packages)} package(s)"
                    )
                else:
                    # Fall back to regex/pattern search only if we did not
                    # find any exact matches so legacy behavior still works.
                    logger.info("   ⚠ No exact match, trying regex...")
                    packages = crud.get_packages(
                        db,
                        skip=0,
                        limit=1000,
                        name_filter=query.name,
                        use_regex=True
                    )
                    logger.info(
                        f"   ✓ Regex: {len(packages)} package(s)"
                    )
                
                for pkg in packages:
                    artifact_type = getattr(pkg, 'artifact_type', 'model')
                    if query.types and artifact_type not in query.types:
                        continue
                        
                    results.append(
                        ArtifactMetadata(
                            name=pkg.name,
                            id=pkg.id,  # Integer, not string
                            type=artifact_type
                        )
                    )
        
        logger.info(
            f"📦 POST /artifacts: Returning {len(results)} total"
        )
        
        # Build response with offset header per OpenAPI spec
        response = JSONResponse(
            content=[
                {
                    "name": r.name,
                    "id": r.id,
                    "type": r.type
                }
                for r in results
            ]
        )
        
        # Add offset header for pagination
        next_offset = str(int(offset or 0) + len(results))
        response.headers["offset"] = next_offset
        
        return response
    
    # Regex search endpoint (OpenAPI spec - BASELINE)
    class ArtifactRegEx(BaseModel):
        """Artifact regex query."""
        regex: str

    @app.post(
        "/artifact/byRegEx",
        response_model=list[ArtifactMetadata],
        tags=["artifacts"]
    )
    def search_artifacts_by_regex(
        regex_query: ArtifactRegEx,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
    ):
        """
        Search for artifacts using regular expression (BASELINE).
        
        Searches artifact names and READMEs using the provided regex pattern.
        Protected against ReDoS attacks via pattern validation and timeout.
        
        Args:
            regex_query: Object containing regex pattern
            db: Database session
            
        Returns:
            List of matching artifact metadata
            
        Raises:
            HTTPException 400: Invalid regex pattern
            HTTPException 404: No artifacts found
        """
        from src.api.dependencies import (
            validate_regex_pattern,
            safe_regex_match,
            RegexTimeoutError,
        )
        
        logger.info(f"🔍 POST /artifact/byRegEx: pattern='{regex_query.regex}'")
        
        # Validate regex pattern first to prevent ReDoS attacks
        try:
            validated_pattern = validate_regex_pattern(regex_query.regex)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ POST /artifact/byRegEx: Invalid pattern - {e}")
            raise HTTPException(
                status_code=400,
                detail=f"There is missing field(s) in the artifact_regex or it is formed improperly, or is invalid"
            )
        
        try:
            # Get all packages and filter with safe regex matching
            all_packages = crud.get_all_packages(db)
            
            logger.info(f"   Checking {len(all_packages)} package(s) against regex")
            
            results = []
            for pkg in all_packages:
                try:
                    # Check if regex matches name OR readme_content
                    name_match = safe_regex_match(validated_pattern, pkg.name)
                    readme_match = False
                    if hasattr(pkg, 'readme_content') and pkg.readme_content:
                        readme_match = safe_regex_match(validated_pattern, pkg.readme_content)
                    
                    if name_match or readme_match:
                        artifact_type = getattr(pkg, 'artifact_type', 'model')
                        results.append(
                            ArtifactMetadata(
                                name=pkg.name,
                                id=str(pkg.id),
                                type=artifact_type
                            )
                        )
                except RegexTimeoutError:
                    # Regex took too long - pattern might be malicious
                    logger.error(f"❌ POST /artifact/byRegEx: Pattern timeout")
                    raise HTTPException(
                        status_code=400,
                        detail="There is missing field(s) in the artifact_regex or it is formed improperly, or is invalid"
                    )
            
            if not results:
                logger.warning("   ⚠ No artifacts found")
                raise HTTPException(
                    status_code=404,
                    detail="No artifact found under this regex."
                )
            
            logger.info(f"✓ POST /artifact/byRegEx: Returning {len(results)}")
            return results
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ POST /artifact/byRegEx: Error - {e}")
            raise HTTPException(
                status_code=400,
                detail=f"There is missing field(s) in the artifact_regex or it is formed improperly, or is invalid"
            )
    
    # Ingest artifact endpoint (OpenAPI spec)
    class ArtifactData(BaseModel):
        """Artifact data for ingest and retrieval."""
        url: str
        name: Optional[str] = None  # Expected artifact name (from autograder)
        download_url: Optional[str] = None  # For GET endpoints

    class ArtifactIngestResponse(BaseModel):
        """Response from artifact ingest."""
        metadata: ArtifactMetadata
        data: ArtifactData

    @app.post(
        "/artifact/{artifact_type}",
        response_model=ArtifactIngestResponse,
        status_code=201,
        tags=["artifacts"]
    )
    async def ingest_artifact(
        artifact_type: str,
        artifact_data: ArtifactData,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
    ):
        """
        Ingest a new artifact from a URL.
        
        Args:
            artifact_type: Type of artifact (model, dataset, code)
            artifact_data: URL to artifact
            db: Database session
            
        Returns:
            Created artifact metadata and data
        """
        from src.ingest import validate_and_ingest
        import uuid
        
        # Log autograder request
        logger.info(
            f"📥 POST /artifact/{artifact_type}: url={artifact_data.url}"
        )
        
        # Use provided name if available, otherwise parse from URL
        url = artifact_data.url
        
        # Check if artifact already exists by source_url
        # This prevents re-streaming from HF to S3 for duplicates
        existing = db.query(Package).filter(
            Package.source_url == url
        ).first()
        
        if existing:
            logger.info(
                f"🔄 Artifact already exists: id={existing.id}, "
                f"name={existing.name}"
            )
            
            # Generate download URL - use direct S3 URL pattern
            s3_bucket = os.getenv("S3_BUCKET_NAME", "ml-registery-artifacts")
            s3_region = os.getenv("AWS_REGION", "us-east-1")
            
            # If S3 key exists, generate S3 URL
            if existing.s3_key:
                download_url = (
                    f"https://{s3_bucket}.s3.{s3_region}.amazonaws.com/"
                    f"{existing.s3_key}"
                )
            else:
                # Generate predictable S3 key for existing artifact
                artifact_name = existing.name
                s3_key = f"{artifact_type}s/{existing.id}/{artifact_name}.tar.gz"
                download_url = (
                    f"https://{s3_bucket}.s3.{s3_region}.amazonaws.com/{s3_key}"
                )
            
            logger.info(f"🔄 Returning existing artifact download_url: {download_url}")
            
            # Return existing artifact info
            return ArtifactIngestResponse(
                metadata=ArtifactMetadata(
                    name=existing.name,
                    id=str(existing.id),
                    type=getattr(existing, 'artifact_type', artifact_type)
                ),
                data=ArtifactData(
                    url=url,
                    download_url=download_url
                )
            )
        
        # Extract full model identifier for validation (e.g., "owner/repo")
        # Clean URL: Remove protocol and domain first
        url_clean = url.strip("/")
        if "://" in url_clean:
            url_clean = url_clean.split("://", 1)[1]  # Remove protocol
        if url_clean.startswith("huggingface.co/"):
            url_clean = url_clean.replace("huggingface.co/", "", 1)
        elif url_clean.startswith("github.com/"):
            url_clean = url_clean.replace("github.com/", "", 1)
        
        url_parts = url_clean.split("/")
        
        # Use autograder-provided name if available
        if artifact_data.name:
            artifact_name = artifact_data.name
        else:
            # Fallback: Extract just the artifact name (WITHOUT owner prefix)
            # Per OpenAPI spec examples: "bert-base-uncased"
            artifact_name = url_parts[-1]
            if artifact_name.endswith('.git'):
                artifact_name = artifact_name[:-4]
        
        # For HuggingFace: "owner/model" or "model"
        # For GitHub: "owner/repo"
        if len(url_parts) >= 2:
            full_model_name = "/".join(url_parts[-2:])
        else:
            full_model_name = url_parts[-1]
        
        # Remove .git suffix if present (for GitHub URLs)
        if full_model_name.endswith('.git'):
            full_model_name = full_model_name[:-4]

        # Create package entry FIRST (before quality gate validation)
        # This ensures we track ALL submission attempts
        logger.info(f"📝 Creating package entry for {artifact_name}...")
        # Get artifact ID
        artifact_id = str(uuid.uuid4())
        
        # Get file size from HuggingFace if available
        file_size_bytes = 0
        try:
            from huggingface_hub import HfApi
            
            # Only fetch size for HuggingFace artifacts (models and datasets)
            if artifact_type in ["model", "dataset"]:
                hf_api_client = HfApi()
                
                if artifact_type == "model":
                    # Use model_info for models
                    model_info = hf_api_client.model_info(full_model_name, files_metadata=True)
                    
                    if hasattr(model_info, 'siblings') and model_info.siblings:
                        total_size = sum(
                            getattr(sibling, 'size', 0) or 0
                            for sibling in model_info.siblings
                        )
                        file_size_bytes = total_size
                        logger.info(f"📊 Calculated size: {file_size_bytes} bytes ({file_size_bytes / (1024*1024):.1f} MB)")
                
                elif artifact_type == "dataset":
                    # Use dataset_info for datasets
                    dataset_info = hf_api_client.dataset_info(full_model_name, files_metadata=True)
                    
                    if hasattr(dataset_info, 'siblings') and dataset_info.siblings:
                        total_size = sum(
                            getattr(sibling, 'size', 0) or 0
                            for sibling in dataset_info.siblings
                        )
                        file_size_bytes = total_size
                        logger.info(f"📊 Calculated dataset size: {file_size_bytes} bytes ({file_size_bytes / (1024*1024):.1f} MB)")
            
            else:
                # For code artifacts (GitHub repos), use GitHub API
                if "github.com" in url.lower():
                    try:
                        import requests
                        import re
                        
                        # Extract owner/repo from GitHub URL
                        # Handles: github.com/owner/repo, github.com/owner/repo.git, github.com/owner/repo/tree/branch
                        match = re.search(r'github\.com[:/]([^/]+)/([^/\.]+)', url)
                        if match:
                            owner, repo = match.groups()
                            
                            # Use GitHub API to get repo info
                            github_api_url = f"https://api.github.com/repos/{owner}/{repo}"
                            headers = {}
                            
                            # Use GITHUB_TOKEN if available (optional)
                            github_token = os.getenv("GITHUB_TOKEN")
                            if github_token:
                                headers["Authorization"] = f"token {github_token}"
                            
                            response = requests.get(github_api_url, headers=headers, timeout=10)
                            
                            if response.status_code == 200:
                                repo_data = response.json()
                                # GitHub returns size in KB
                                file_size_bytes = repo_data.get("size", 0) * 1024
                                logger.info(f"📊 GitHub repo size: {file_size_bytes} bytes ({file_size_bytes / (1024*1024):.1f} MB)")
                            else:
                                logger.warning(f"GitHub API returned {response.status_code} for {owner}/{repo}")
                                file_size_bytes = 0
                        else:
                            logger.warning(f"Could not extract owner/repo from GitHub URL: {url}")
                            file_size_bytes = 0
                    except Exception as e:
                        logger.warning(f"Could not get size from GitHub API: {e}")
                        file_size_bytes = 0
                else:
                    # Non-GitHub code artifact
                    logger.info(f"📊 Code artifact (non-GitHub) - size calculation not implemented, using 0")
                    file_size_bytes = 0
                
        except Exception as e:
            # Handle 401 errors and other failures gracefully
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                logger.warning(f"⚠️  HuggingFace authentication error - file size will be 0. Consider setting HF_TOKEN.")
            else:
                logger.warning(f"Could not get file size from HuggingFace: {e}")
            file_size_bytes = 0
        
        # Fetch README content for searchability
        readme_content = None
        try:
            from src.hf_api import HuggingFaceAPI
            from src.models import ParsedURL, URLCategory
            
            if artifact_type == "model" and "huggingface" in url.lower():
                hf_api = HuggingFaceAPI()
                parsed_url = ParsedURL(
                    url=url,
                    category=URLCategory.MODEL,
                    name=artifact_name,
                    platform="huggingface",
                    owner=url_parts[-2] if len(url_parts) >= 2 else None,
                    repo=url_parts[-1] if url_parts else artifact_name
                )
                readme_content = hf_api.get_readme_content(parsed_url)
                if readme_content:
                    logger.info(f"📄 Fetched README: {len(readme_content)} chars")
        except Exception as e:
            logger.warning(f"Could not fetch README: {e}")
            readme_content = None
        
        from src.database import crud
        package = crud.create_package(
            db,
            name=artifact_name,
            version="1.0.0",
            artifact_type=artifact_type,
            s3_key=artifact_id,
            s3_bucket="",
            file_size_bytes=file_size_bytes,
            source_url=url,
            readme_content=readme_content,  # Store README for regex search
            uploaded_by=1,  # Default admin user
            ingest_status="pending"  # Mark as pending evaluation
        )
        db.commit()
        db.refresh(package)
        
        logger.info(
            f"💾 PACKAGE CREATED: id={package.id}, name={artifact_name}, "
            f"type={artifact_type}, status=pending"
        )
        
        # Only models go through quality gate validation
        # Datasets and code are accepted without validation
        validation_result = None
        
        if artifact_type == "model":
            # Run quality gate validation for models only
            logger.info(f"   Validating {full_model_name}...") 
            passes_gate, validation_result = validate_and_ingest(
                full_model_name
            )
            
            if not passes_gate:
                # Quality gate FAILED - update package status to rejected
                logger.warning(
                    f"❌ QUALITY GATE FAILED: {artifact_name} - "
                    f"Failing metrics: {validation_result.get('failing_metrics')}"
                )
                
                package.ingest_status = "rejected"
                package.quality_gate_result = {
                    "passed": False,
                    "evaluated_at": datetime.utcnow().isoformat(),
                    "failing_metrics": validation_result.get("failing_metrics", [])
                }
                db.commit()
                
                logger.info(
                    f"📊 PACKAGE UPDATED: id={package.id}, status=rejected"
                )
                
                # Save scores even for failed artifacts
                if validation_result and validation_result.get("all_scores"):
                    scores = validation_result["all_scores"]
                    crud.create_or_update_package_score(
                        db,
                        package_id=package.id,
                        ramp_up_time=scores.get("ramp_up_time", 0.0),
                        bus_factor=scores.get("bus_factor", 0.0),
                        performance_claims=scores.get(
                            "performance_claims", 0.0
                        ),
                        license_score=scores.get("license", 0.6),
                        dataset_quality=scores.get(
                            "dataset_quality", 0.0
                        ),
                        dataset_code_linkage=scores.get(
                            "dataset_and_code_score", 0.0
                        ),
                        code_quality=scores.get("code_quality", 0.0),
                        reproducibility=scores.get(
                            "reproducibility", 0.5
                        ),
                        reviewedness=scores.get("reviewedness", 0.6),
                        treescore=scores.get("treescore", 0.6),
                        size_score=scores.get("size_score", 0.0),
                        size_score_raspberry_pi=scores.get(
                            "size_score_raspberry_pi", 0.0
                        ),
                        size_score_jetson_nano=scores.get(
                            "size_score_jetson_nano", 0.0
                        ),
                        size_score_desktop_pc=scores.get(
                            "size_score_desktop_pc", 0.0
                        ),
                        size_score_aws_server=scores.get(
                            "size_score_aws_server", 0.0
                        ),
                        net_score=scores.get("net_score", 0.0)
                    )
                    logger.info(
                        f"💾 Saved scores for rejected artifact "
                        f"{package.id}"
                    )
                
                # Still return 424 per OpenAPI spec
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=424,
                    detail={
                        "message": "Artifact disqualified due to ratings",
                        "failing_metrics": validation_result.get("failing_metrics")
                    }
                )
            
            # Quality gate PASSED - update package status to approved
            logger.info(f"✅ QUALITY GATE PASSED: {artifact_name}")
            
            package.ingest_status = "approved"
            package.quality_gate_result = {
                "passed": True,
                "evaluated_at": datetime.utcnow().isoformat(),
                "net_score": validation_result.get("all_scores", {}).get("net_score", 0.0)
            }
            db.commit()
            db.refresh(package)
            
            logger.info(
                f"📊 PACKAGE UPDATED: id={package.id}, status=approved, "
                f"net_score={package.quality_gate_result.get('net_score', 0.0)}"
            )
        else:
            # Dataset and code artifacts are auto-approved without quality gate
            logger.info(f"✅ AUTO-APPROVED: {artifact_type} artifacts don't require quality gate validation")
            
            package.ingest_status = "approved"
            package.quality_gate_result = {
                "passed": True,
                "evaluated_at": datetime.utcnow().isoformat(),
                "note": f"{artifact_type} artifacts are accepted without quality gate validation"
            }
            db.commit()
            db.refresh(package)
            
            logger.info(
                f"📊 PACKAGE UPDATED: id={package.id}, status=approved (auto)"
            )
        
        # S3 Storage - Queue for async upload to S3
        enable_s3 = os.getenv(
            "ENABLE_S3_STORAGE",
            "false"
        ).lower() == "true"
        
        s3_bucket = os.getenv("S3_BUCKET_NAME", "ml-registery-artifacts")
        s3_region = os.getenv("AWS_REGION", "us-east-1")
        sqs_queue_url = os.getenv("SQS_QUEUE_URL")
        
        # Generate predictable S3 download URL (before upload completes)
        s3_key_prefix = f"{artifact_type}s/{package.id}"
        s3_download_url = (
            f"https://{s3_bucket}.s3.{s3_region}.amazonaws.com/"
            f"{s3_key_prefix}/"
        )
        
        logger.info(
            f"☁️  S3 Config: enable_s3={enable_s3}, "
            f"S3_AVAILABLE={S3_AVAILABLE}, bucket={s3_bucket}"
        )
        
        if enable_s3 and S3_AVAILABLE and sqs_queue_url:
            try:
                # Queue artifact for async upload to S3 via ECS worker
                import json
                import boto3
                
                # Parse repo_id from URL
                # Clean URL: Remove protocol and domain first
                url_clean = url.strip("/")
                if "://" in url_clean:
                    url_clean = url_clean.split("://", 1)[1]
                if url_clean.startswith("huggingface.co/"):
                    url_clean = url_clean.replace("huggingface.co/", "", 1)
                elif url_clean.startswith("github.com/"):
                    url_clean = url_clean.replace("github.com/", "", 1)
                
                url_parts = url_clean.split("/")
                if len(url_parts) >= 2:
                    repo_id = "/".join(url_parts[-2:])
                else:
                    repo_id = url_parts[-1]
                
                # Remove .git suffix if present
                if repo_id.endswith('.git'):
                    repo_id = repo_id[:-4]
                
                # Determine repo_type for HuggingFace
                repo_type = "model" if artifact_type == "model" else "dataset"
                
                # Create SQS message for worker
                message_body = {
                    "artifact_id": package.id,
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "revision": "main",
                    "source_url": url,
                    "artifact_type": artifact_type,
                    "requested_at": datetime.utcnow().isoformat() + "Z"
                }
                
                sqs_client = boto3.client('sqs', region_name=s3_region)
                
                # Send message to SQS (non-blocking, <100ms)
                response = sqs_client.send_message(
                    QueueUrl=sqs_queue_url,
                    MessageBody=json.dumps(message_body),
                    MessageAttributes={
                        'ArtifactType': {
                            'DataType': 'String',
                            'StringValue': artifact_type
                        },
                        'ArtifactId': {
                            'DataType': 'String',
                            'StringValue': str(package.id)
                        }
                    }
                )
                
                message_id = response.get('MessageId')
                
                # Update package with queue status
                package.s3_key = s3_key_prefix
                package.s3_bucket = s3_bucket
                package.ingest_status = "processing"
                package.quality_gate_result["sqs_message_id"] = message_id
                package.quality_gate_result["queued_at"] = (
                    datetime.utcnow().isoformat()
                )
                db.commit()
                
                logger.info(
                    f"✅ Queued for S3 upload: artifact_id={package.id}, "
                    f"message_id={message_id}"
                )
                logger.info(
                    f"☁️  S3 download URL (pending): {s3_download_url}"
                )
                
            except Exception as e:
                logger.error(f"❌ Failed to queue S3 upload: {e}")
                # Fall back to source URL on queue error
                s3_download_url = url
        else:
            logger.info("☁️  S3/SQS disabled, using source URL")
            s3_download_url = url
        
        # Store the quality gate scores in database (only for models)
        if (artifact_type == "model" and validation_result and
                validation_result.get("all_scores")):
            scores = validation_result["all_scores"]
            crud.create_or_update_package_score(
                db,
                package_id=package.id,
                ramp_up_time=scores.get("ramp_up_time", 0.0),
                bus_factor=scores.get("bus_factor", 0.0),
                performance_claims=scores.get(
                    "performance_claims", 0.0
                ),
                license_score=scores.get("license", 0.0),
                dataset_quality=scores.get("dataset_quality", 0.0),
                dataset_code_linkage=scores.get(
                    "dataset_and_code_score", 0.0
                ),
                code_quality=scores.get("code_quality", 0.0),
                reproducibility=scores.get("reproducibility", 0.0),
                reviewedness=scores.get("reviewedness", 0.0),
                treescore=scores.get("treescore", 0.0),
                size_score=scores.get("size_score", 0.0),
                size_score_raspberry_pi=scores.get(
                    "size_score_raspberry_pi", 0.0
                ),
                size_score_jetson_nano=scores.get(
                    "size_score_jetson_nano", 0.0
                ),
                size_score_desktop_pc=scores.get(
                    "size_score_desktop_pc", 0.0
                ),
                size_score_aws_server=scores.get(
                    "size_score_aws_server", 0.0
                ),
                net_score=scores.get("net_score", 0.0)
            )
            
            # Extract and persist lineage relationships
            logger.info("🌳 Extracting artifact lineage...")
            try:
                # Use LLM to analyze dependencies
                from src.llm.analyzer import (
                    analyze_artifact_dependencies
                )
                
                # Get config data if available
                config_data = None
                try:
                    from src.hf_api import HuggingFaceAPI
                    from src.models import ParsedURL, URLCategory
                    
                    if "huggingface" in url.lower():
                        hf_api = HuggingFaceAPI()
                        parsed_url = ParsedURL(
                            url=url,
                            category=URLCategory.MODEL,
                            name=artifact_name,
                            platform="huggingface",
                            owner=(url_parts[-2]
                                   if len(url_parts) >= 2
                                   else None),
                            repo=(url_parts[-1]
                                  if url_parts
                                  else artifact_name)
                        )
                        config_data = hf_api.get_model_config(
                            parsed_url
                        )
                except Exception as e:
                    logger.debug(f"Could not fetch config: {e}")
                
                # Analyze dependencies with LLM
                lineage_metadata = analyze_artifact_dependencies(
                    config_data=config_data,
                    readme_content=readme_content,
                    model_url=url
                )
                
                # Store lineage metadata in package
                package.lineage_metadata = lineage_metadata
                db.commit()
                
                # Create PackageLineage entries for parent models
                parent_models = lineage_metadata.get(
                    "parent_models", []
                )
                for parent_info in parent_models:
                    parent_id_str = parent_info.get("id")
                    relationship = parent_info.get(
                        "relationship", "depends_on"
                    )
                    
                    # Look up parent package by name
                    if "/" in parent_id_str:
                        parent_name = parent_id_str.split("/")[-1]
                    else:
                        parent_name = parent_id_str
                    
                    # Find parent package in database
                    parent_packages = db.query(Package).filter(
                        Package.name == parent_name
                    ).all()
                    
                    if parent_packages:
                        # Use first match
                        parent_pkg = parent_packages[0]
                        
                        # Create lineage relationship
                        crud.create_lineage(
                            db,
                            parent_package_id=parent_pkg.id,
                            child_package_id=package.id,
                            relationship_type=relationship
                        )
                        logger.info(
                            f"   ✓ Linked parent: {parent_name} "
                            f"({relationship})"
                        )
                
                logger.info(
                    f"✅ Lineage extracted: "
                    f"{len(parent_models)} parent(s)"
                )
                
            except Exception as e:
                logger.warning(f"Lineage extraction failed: {e}")
            
            # Trigger cascade tree score updates for dependent
            # artifacts
            logger.info("🔄 Updating dependent tree scores...")
            try:
                updated_count = crud.update_dependent_tree_scores(
                    db,
                    package_id=package.id,
                    max_depth=3
                )
                if updated_count > 0:
                    logger.info(
                        f"✅ Updated tree scores for "
                        f"{updated_count} dependent artifact(s)"
                    )
            except Exception as e:
                logger.warning(
                    f"Cascade tree score update failed: {e}"
                )
        
        # s3_download_url is already set above
        # (either S3 URL or source URL fallback)
        logger.info(
            f"📦 Returning artifact with download_url: "
            f"{s3_download_url}"
        )
        
        # Return response with download_url (per OpenAPI spec)
        return ArtifactIngestResponse(
            metadata=ArtifactMetadata(
                name=artifact_name,
                id=str(package.id),
                type=artifact_type
            ),
            data=ArtifactData(
                url=url,
                download_url=s3_download_url
            )
        )
    
    # GET endpoints for listing artifacts by type
    @app.get(
        "/artifact/model",
        response_model=list[ArtifactMetadata],
        tags=["artifacts"]
    )
    def get_all_models(db: Session = Depends(get_db)):
        """
        Get all model artifacts.
        
        Returns:
            List of all model artifact metadata
        """
        logger.info("🔍 AUTOGRADER: GET all models")
        packages = db.query(Package).filter(
            Package.artifact_type == "model"
        ).all()
        
        result = [
            ArtifactMetadata(
                name=p.name,
                id=str(p.id),  # Already string
                type="model"
            )
            for p in packages
        ]
        logger.info(f"✅ Found {len(result)} models")
        return result
    
    @app.get(
        "/artifact/code",
        response_model=list[ArtifactMetadata],
        tags=["artifacts"]
    )
    def get_all_code(db: Session = Depends(get_db)):
        """
        Get all code artifacts.
        
        Returns:
            List of all code artifact metadata
        """
        logger.info("🔍 AUTOGRADER: GET all code")
        packages = db.query(Package).filter(
            Package.artifact_type == "code"
        ).all()
        
        result = [
            ArtifactMetadata(
                name=p.name,
                id=str(p.id),  # Already string
                type="code"
            )
            for p in packages
        ]
        logger.info(f"✅ Found {len(result)} code artifacts")
        return result
    
    @app.get(
        "/artifact/dataset",
        response_model=list[ArtifactMetadata],
        tags=["artifacts"]
    )
    def get_all_datasets(db: Session = Depends(get_db)):
        """
        Get all dataset artifacts.
        
        Returns:
            List of all dataset artifact metadata
        """
        logger.info("🔍 AUTOGRADER: GET all datasets")
        packages = db.query(Package).filter(
            Package.artifact_type == "dataset"
        ).all()
        
        result = [
            ArtifactMetadata(
                name=p.name,
                id=str(p.id),  # Already string
                type="dataset"
            )
            for p in packages
        ]
        logger.info(f"✅ Found {len(result)} datasets")
        return result
    
    # Rate endpoint - GET /artifact/model/{id}/rate (BASELINE)
    @app.get(
        "/artifact/model/{id}/rate",
        tags=["artifacts"],
        status_code=200
    )
    def get_model_rating(
        id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
    ):
        """
        Get ratings/scores for a model artifact (BASELINE).
        
        Per OpenAPI spec: Returns ModelRating with all quality metrics.
        All field names must match spec exactly (snake_case).
        
        Args:
            id: Artifact ID
            db: Database session
            
        Returns:
            ModelRating with all scores per spec format
        """
        from fastapi import HTTPException
        
        # Validate ID format
        validate_id(id)
        
        # Try to convert to int for database lookup
        # If it's a valid format but not a number, treat as not found (404)
        try:
            package_id = int(id)
        except ValueError:
            # Valid format (passed validate_id) but not an integer
            # This means it won't exist in our integer-based DB
            raise HTTPException(
                status_code=404,
                detail=f"Artifact with ID {id} not found"
            )
        
        package = crud.get_package_by_id(db, package_id)
        if not package:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact with ID {id} not found"
            )
        
        # Get scores
        scores = crud.get_package_scores(db, package_id)
        if not scores:
            raise HTTPException(
                status_code=404,
                detail=f"No ratings found for artifact {id}"
            )
        
        # Determine category from artifact type or use generic "model"
        category = getattr(package, 'artifact_type', 'model')
        
        # Calculate latencies in seconds (convert from ms if available)
        latency_seconds = (
            scores.scoring_latency_ms / 1000.0
            if scores.scoring_latency_ms
            else 0.0
        )
        
        # For now, use same latency for all metrics (could track separately)
        individual_latency = (
            latency_seconds / 11.0
            if latency_seconds > 0
            else 0.0
        )
        
        # Build size_score object with 4 platform scores from database
        size_score_obj = {
            "raspberry_pi": (
                scores.size_score_raspberry_pi
                if scores.size_score_raspberry_pi is not None
                else 0.0
            ),
            "jetson_nano": (
                scores.size_score_jetson_nano
                if scores.size_score_jetson_nano is not None
                else 0.0
            ),
            "desktop_pc": (
                scores.size_score_desktop_pc
                if scores.size_score_desktop_pc is not None
                else 0.0
            ),
            "aws_server": (
                scores.size_score_aws_server
                if scores.size_score_aws_server is not None
                else 0.0
            )
        }
        
        # Return ModelRating per OpenAPI spec (lines 1063-1216)
        return {
            # Required metadata fields
            "name": package.name,
            "category": category,
            
            # Net score (overall)
            "net_score": (
                scores.net_score if scores.net_score is not None else 0.0
            ),
            "net_score_latency": latency_seconds,
            
            # Phase 1 metrics (8 metrics)
            "ramp_up_time": (
                scores.ramp_up_time if scores.ramp_up_time is not None else 0.0
            ),
            "ramp_up_time_latency": individual_latency,
            
            "bus_factor": (
                scores.bus_factor if scores.bus_factor is not None else 0.0
            ),
            "bus_factor_latency": individual_latency,
            
            "performance_claims": (
                scores.performance_claims
                if scores.performance_claims is not None
                else 0.0
            ),
            "performance_claims_latency": individual_latency,
            
            "license": (
                scores.license_score
                if scores.license_score is not None
                else 0.0
            ),
            "license_latency": individual_latency,
            
            "dataset_and_code_score": (
                scores.dataset_code_linkage
                if scores.dataset_code_linkage is not None
                else 0.0
            ),
            "dataset_and_code_score_latency": individual_latency,
            
            "dataset_quality": (
                scores.dataset_quality
                if scores.dataset_quality is not None
                else 0.0
            ),
            "dataset_quality_latency": individual_latency,
            
            "code_quality": (
                scores.code_quality if scores.code_quality is not None else 0.0
            ),
            "code_quality_latency": individual_latency,
            
            "size_score": size_score_obj,  # Object with 4 platform scores
            "size_score_latency": individual_latency,
            
            # Phase 2 metrics (3 additional)
            "reproducibility": (
                scores.reproducibility
                if scores.reproducibility is not None
                else 0.0
            ),
            "reproducibility_latency": individual_latency,
            
            "reviewedness": (
                scores.reviewedness
                if scores.reviewedness is not None
                else 0.0
            ),
            "reviewedness_latency": individual_latency,
            
            "tree_score": (
                scores.treescore
                if scores.treescore is not None
                else 0.0
            ),
            "tree_score_latency": individual_latency,
        }
    
    # Get artifact by name endpoint (NON-BASELINE per spec)
    @app.get(
        "/artifact/byName/{name:path}",
        response_model=list[ArtifactMetadata],
        tags=["artifacts"]
    )
    def get_artifact_by_name(
        name: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
    ):
        """
        List artifact metadata for this name.
        
        Returns all artifacts (models, datasets, code) that match the
        given name. Multiple artifacts can share the same name but have
        different IDs.
        
        Note: Uses :path converter to allow names with slashes (/).
        Requires authentication per OpenAPI spec.
        
        Args:
            name: Artifact name to search for
            db: Database session
            current_user: Authenticated user
            
        Returns:
            List of artifact metadata entries matching the name
        """
        from fastapi import HTTPException
        
        # Log query
        logger.info(f"🔍 GET /artifact/byName/{name}")
        
        # Search for packages with this exact name (all statuses - for transparency)
        packages = db.query(Package).filter(
            Package.name == name
        ).all()
        
        logger.info(f"   Found {len(packages)} package(s) with name '{name}'")
        
        if not packages:
            logger.warning("   ⚠ No artifact found")
            raise HTTPException(
                status_code=404,
                detail=f"No artifact found with name: {name}"
            )
        
        # Return all matching artifacts
        results = []
        for pkg in packages:
            artifact_type = getattr(pkg, 'artifact_type', 'model')
            results.append(
                ArtifactMetadata(
                    name=pkg.name,
                    id=str(pkg.id),  # Convert to string
                    type=artifact_type
                )
            )
        
        logger.info(f"✓ GET /artifact/byName: Returning {len(results)}")
        return results
    
    # Get artifact by type and ID endpoint (BASELINE)
    class Artifact(BaseModel):
        """Complete artifact with metadata and data."""
        metadata: ArtifactMetadata
        data: ArtifactData
    
    @app.get(
        "/artifacts/{artifact_type}/{id}",
        response_model=Artifact,
        tags=["artifacts"]
    )
    def get_artifact_by_id(
        artifact_type: str,
        id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
    ):
        """
        Retrieve artifact by type and ID.
        
        Per OpenAPI spec: Returns artifact with metadata and data (url).
        Requires authentication.
        
        Args:
            artifact_type: Type of artifact (model, dataset, code)
            id: Artifact ID
            db: Database session
            current_user: Authenticated user
            
        Returns:
            Artifact with metadata and data
        """
        from fastapi import HTTPException
        
        # Log query
        logger.info(f"🔍 GET /artifacts/{artifact_type}/{id}")
        
        # Validate ID format
        validate_id(id)
        
        # Validate artifact type
        if artifact_type not in ["model", "dataset", "code"]:
            logger.warning(f"   ⚠ Invalid type: {artifact_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid artifact type: {artifact_type}"
            )
        
        # Try to convert to int for database lookup
        try:
            package_id = int(id)
        except ValueError:
            logger.warning(f"❌ NOT FOUND: {id} (valid format but not in DB)")
            raise HTTPException(
                status_code=404,
                detail=f"Artifact with ID {id} not found"
            )
        
        package = crud.get_package_by_id(db, package_id)
        if not package:
            logger.warning(f"❌ NOT FOUND: No artifact with id={id}")
            raise HTTPException(
                status_code=404,
                detail=f"Artifact with ID {id} not found"
            )
        
        # Check if artifact type matches
        pkg_artifact_type = getattr(package, 'artifact_type', 'model')
        if pkg_artifact_type != artifact_type:
            logger.warning(
                f"❌ TYPE MISMATCH: id={id} is '{pkg_artifact_type}' "
                f"not '{artifact_type}'"
            )
            raise HTTPException(
                status_code=404,
                detail=f"Artifact {id} is not of type {artifact_type}"
            )
        
        logger.info(
            f"   ✅ Found: id={package.id}, name={package.name}, "
            f"type={pkg_artifact_type}"
        )
        
        # Return artifact with metadata and data
        url = getattr(package, 'source_url', '')
        if not url:
            # Fallback to constructed URL if no source_url
            url = f"https://huggingface.co/{package.name}"
        
        # Build download URL - use S3 URL
        s3_bucket = os.getenv("S3_BUCKET_NAME", "ml-registery-artifacts")
        s3_region = os.getenv("AWS_REGION", "us-east-1")
        
        if package.s3_key:
            download_url = (
                f"https://{s3_bucket}.s3.{s3_region}.amazonaws.com/"
                f"{package.s3_key}"
            )
        else:
            # Generate predictable S3 key
            s3_key = f"{artifact_type}s/{id}/{package.name}.tar.gz"
            download_url = (
                f"https://{s3_bucket}.s3.{s3_region}.amazonaws.com/{s3_key}"
            )
        
        return Artifact(
            metadata=ArtifactMetadata(
                name=package.name,
                id=str(package.id),  # Convert to string
                type=pkg_artifact_type
            ),
            data=ArtifactData(
                url=url,
                download_url=download_url
            )
        )
    
    # Download endpoint - Proxy to original source or S3
    @app.get(
        "/download/{artifact_type}/{id}",
        tags=["artifacts"]
    )
    def download_artifact(
        artifact_type: str,
        id: str,
        request: Request,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """
        Download artifact by redirecting to S3 or original source.
        Records download in audit trail.
        
        If S3 storage is enabled and artifact is in S3, returns presigned
        URL. Otherwise, redirects to original HuggingFace/GitHub URL.
        
        Args:
            artifact_type: Type of artifact (model, dataset, code)
            id: Artifact ID
            request: FastAPI request for IP and user agent
            db: Database session
            current_user: Optional authenticated user
            
        Returns:
            Redirect to artifact download URL
        """
        # Validate ID format
        validate_id(id)
        
        # Validate artifact type
        if artifact_type not in ["model", "dataset", "code"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid artifact type: {artifact_type}"
            )
        
        # Try to convert to int for database lookup
        try:
            package_id = int(id)
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact not found: {id}"
            )
        
        package = crud.get_package_by_id(db, package_id)
        if not package:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact not found: {id}"
            )
        
        # Check if artifact type matches
        pkg_artifact_type = getattr(package, 'artifact_type', 'model')
        if pkg_artifact_type != artifact_type:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact {id} is not of type {artifact_type}"
            )
        
        # Check if S3 storage is enabled and artifact is in S3
        enable_s3 = os.getenv(
            "ENABLE_S3_STORAGE",
            "false"
        ).lower() == "true"
        
        if (enable_s3 and S3_AVAILABLE and
                package.s3_key and package.s3_bucket):
            try:
                # Generate presigned S3 URL (valid for 1 hour)
                s3_storage = get_s3_storage()
                download_url = s3_storage.generate_download_url(
                    package.s3_key,
                    expiration=3600
                )
                
                # Record S3 download in audit trail
                try:
                    client_ip = request.client.host if request.client else None
                    user_agent = request.headers.get("user-agent")
                    user_id = current_user.id if current_user else None
                    
                    crud.record_download(
                        db=db,
                        package_id=package.id,
                        user_id=user_id,
                        ip_address=client_ip,
                        user_agent=user_agent,
                        access_granted=True,
                        access_control_result={"source": "s3"}
                    )
                    logger.info(
                        f"📝 Recorded S3 download: artifact={id}, user={user_id}"
                    )
                except Exception as audit_error:
                    logger.error(f"Failed to record download audit: {audit_error}")
                
                logger.info(
                    f"Generated S3 download URL for artifact {id}"
                )
                return RedirectResponse(url=download_url)
            except Exception as e:
                logger.error(f"Failed to generate S3 URL: {e}")
                # Fall through to original URL
        
        # Record download in audit trail
        try:
            client_ip = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
            user_id = current_user.id  # Authentication is now required
            
            crud.record_download(
                db=db,
                package_id=package.id,
                user_id=user_id,
                ip_address=client_ip,
                user_agent=user_agent,
                access_granted=True,
                access_control_result=None
            )
            logger.info(
                f"📝 Recorded download audit: artifact={id}, "
                f"user={user_id}, ip={client_ip}"
            )
        except Exception as e:
            logger.error(f"Failed to record download audit: {e}")
            # Continue with download even if audit recording fails
        
        # Fall back to original source URL
        url = getattr(package, 'source_url', '')
        if not url:
            url = f"https://huggingface.co/{package.name}"
        
        logger.info(f"Redirecting to original URL for artifact {id}")
        return RedirectResponse(url=url)
    
    # Audit trail endpoint (NON-BASELINE)
    @app.get(
        "/artifact/{artifact_type}/{id}/audit",
        response_model=list,
        tags=["artifacts"]
    )
    def get_artifact_audit(
        artifact_type: str,
        id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
    ):
        """
        Retrieve audit entries for this artifact (NON-BASELINE).
        
        Returns historical information about the artifact including
        what changed, when, and by whom. Tracks CREATE, UPDATE,
        DOWNLOAD, RATE, and AUDIT actions.
        
        Args:
            artifact_type: Type of artifact (model, dataset, code)
            id: Artifact ID
            db: Database session
            current_user: Authenticated user
            
        Returns:
            List of ArtifactAuditEntry objects
        """
        from src.api.schemas import ArtifactAuditEntry, AuditUser
        from src.api.schemas import AuditArtifactMetadata
        
        # Validate ID format
        validate_id(id)
        
        # Validate artifact type
        if artifact_type not in ["model", "dataset", "code"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid artifact type: {artifact_type}"
            )
        
        # Convert to int for database lookup
        try:
            package_id = int(id)
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact not found: {id}"
            )
        
        # Check if artifact exists
        package = crud.get_package_by_id(db, package_id)
        if not package:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact not found: {id}"
            )
        
        # Get download history (DOWNLOAD actions)
        download_history = crud.get_download_history(
            db, package_id=package_id, limit=1000
        )
        
        audit_entries = []
        
        # Add CREATE action (from package creation)
        if package.uploaded_by:
            creator = crud.get_user_by_id(db, package.uploaded_by)
            if creator:
                audit_entries.append(ArtifactAuditEntry(
                    user=AuditUser(
                        name=creator.username,
                        is_admin=creator.is_admin
                    ),
                    date=package.uploaded_at,
                    artifact=AuditArtifactMetadata(
                        name=package.name,
                        id=str(package.id),
                        type=getattr(package, 'artifact_type', artifact_type)
                    ),
                    action="CREATE"
                ))
        
        # Add UPDATE action if package was updated
        if hasattr(package, 'updated_at') and package.updated_at:
            if package.updated_at != package.uploaded_at:
                updater = None
                if package.uploaded_by:
                    updater = crud.get_user_by_id(db, package.uploaded_by)
                if updater:
                    audit_entries.append(ArtifactAuditEntry(
                        user=AuditUser(
                            name=updater.username,
                            is_admin=updater.is_admin
                        ),
                        date=package.updated_at,
                        artifact=AuditArtifactMetadata(
                            name=package.name,
                            id=str(package.id),
                            type=getattr(package, 'artifact_type', artifact_type)
                        ),
                        action="UPDATE"
                    ))
        
        # Add DOWNLOAD actions
        for download in download_history:
            if download.user_id:
                downloader = crud.get_user_by_id(db, download.user_id)
                if downloader:
                    audit_entries.append(ArtifactAuditEntry(
                        user=AuditUser(
                            name=downloader.username,
                            is_admin=downloader.is_admin
                        ),
                        date=download.downloaded_at,
                        artifact=AuditArtifactMetadata(
                            name=package.name,
                            id=str(package.id),
                            type=getattr(package, 'artifact_type', artifact_type)
                        ),
                        action="DOWNLOAD"
                    ))
        
        # Sort by date (newest first)
        audit_entries.sort(key=lambda x: x.date, reverse=True)
        
        logger.info(
            f"📋 Retrieved {len(audit_entries)} audit entries for "
            f"artifact {id}"
        )
        
        return audit_entries
    
    # PUT endpoint - Update artifact (BASELINE)
    @app.put(
        "/artifacts/{artifact_type}/{id}",
        response_model=None,
        tags=["artifacts"],
        status_code=200
    )
    async def update_artifact(
        artifact_type: str,
        id: str,
        artifact: Artifact,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
    ):
        """
        Update artifact content (BASELINE).
        
        The name and id must match. The artifact source will replace 
        the previous contents.
        
        Args:
            artifact_type: Type of artifact to update
            id: Artifact ID
            artifact: New artifact data
            db: Database session
            current_user: Authenticated user
            
        Returns:
            Success message
        """
        logger.info(f"🔄 UPDATE ARTIFACT: type={artifact_type}, id={id}")
        
        # Validate ID format
        validate_id(id)
        
        # Validate artifact type
        if artifact_type not in ["model", "dataset", "code"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid artifact type: {artifact_type}"
            )
        
        # Try to convert to int for database lookup
        try:
            package_id = int(id)
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact does not exist: {id}"
            )
        
        package = crud.get_package_by_id(db, package_id)
        if not package:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact does not exist: {id}"
            )
        
        # Verify name and id match
        if artifact.metadata.id != id:
            raise HTTPException(
                status_code=400,
                detail="Artifact ID in body must match URL parameter"
            )
        
        # Check artifact type matches
        pkg_artifact_type = getattr(package, 'artifact_type', 'model')
        if pkg_artifact_type != artifact_type:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact {id} is not of type {artifact_type}"
            )
        
        # Update the artifact with new data
        package.source_url = artifact.data.url
        package.updated_at = datetime.utcnow()
        
        # If new URL provided, could re-download and validate (optional)
        # For now, just update the metadata
        db.commit()
        db.refresh(package)
        
        logger.info(f"✅ UPDATED: artifact {id}")
        return {"message": "Artifact is updated."}
    
    # DELETE endpoint - Delete artifact (BASELINE - with proper auth)
    @app.delete(
        "/artifacts/{artifact_type}/{id}",
        tags=["artifacts"],
        status_code=200
    )
    def delete_artifact_by_type_and_id(
        artifact_type: str,
        id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
    ):
        """
        Delete artifact (BASELINE).
        
        Args:
            artifact_type: Type of artifact to delete
            id: Artifact ID
            db: Database session
            current_user: Authenticated user
            
        Returns:
            Success message
        """
        logger.info(f"🗑️  DELETE ARTIFACT: type={artifact_type}, id={id}")
        
        # Validate ID format
        validate_id(id)
        
        # Validate artifact type
        if artifact_type not in ["model", "dataset", "code"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid artifact type: {artifact_type}"
            )
        
        # Try to convert to int for database lookup
        try:
            package_id = int(id)
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact does not exist: {id}"
            )
        
        package = crud.get_package_by_id(db, package_id)
        if not package:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact does not exist: {id}"
            )
        
        # Check artifact type matches
        pkg_artifact_type = getattr(package, 'artifact_type', 'model')
        if pkg_artifact_type != artifact_type:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact {id} is not of type {artifact_type}"
            )
        
        # Delete the artifact
        crud.delete_package(db, package_id)
        
        logger.info(f"✅ DELETED: artifact {id}")
        return {"message": "Artifact is deleted."}
    
    # Cost endpoint - Get artifact cost (BASELINE)
    class ArtifactCostResponse(BaseModel):
        """Cost response for artifacts."""
        pass
    
    @app.get(
        "/artifact/{artifact_type}/{id}/cost",
        tags=["artifacts"],
        status_code=200
    )
    def get_artifact_cost(
        artifact_type: str,
        id: str,
        dependency: bool = Query(False),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
    ):
        """
        Get the cost of an artifact (BASELINE).
        
        Requires authentication.
        
        Cost is measured in KB (kilobytes) based on content size.
        Formula: standalone_cost = max(1.0, content_size_bytes / 1024.0)
        Minimum cost is 1.0 KB (even for empty content).
        When dependency=true, total_cost = standalone_cost * 2.0
        
        Args:
            artifact_type: Type of artifact (model/dataset/code)
            id: Artifact ID
            dependency: Include dependencies in cost calculation
                (default: False)
            db: Database session
            current_user: Authenticated user
            
        Returns:
            Cost information with standalone_cost and total_cost in KB
            
        Raises:
            HTTPException 400: Invalid artifact type or ID
            HTTPException 404: Artifact not found or type mismatch
            
        Example calculations:
            - Content size = 512 bytes, dependency=false
              → standalone_cost = max(1.0, 512/1024) = 1.0 KB
              → total_cost = 1.0 KB
              
            - Content size = 5120 bytes, dependency=true
              → standalone_cost = max(1.0, 5120/1024) = 5.0 KB
              → total_cost = 10.0 KB
              
            - Content size = 0 bytes, dependency=false
              → standalone_cost = max(1.0, 0/1024) = 1.0 KB (minimum)
              → total_cost = 1.0 KB
        """
        user_str = current_user.username  # Authentication is now required
        logger.info(
            f"💰 COST QUERY: type={artifact_type}, id={id}, "
            f"deps={dependency}, user={user_str}"
        )
        
        # Validate ID format
        validate_id(id)
        
        # Validate artifact type
        if artifact_type not in ["model", "dataset", "code"]:
            logger.warning(f"❌ Invalid artifact type: {artifact_type}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid artifact type: {artifact_type}"
            )
        
        # Try to convert to int for database lookup
        try:
            package_id = int(id)
            logger.info(f"   📋 Looking up package ID: {package_id}")
        except ValueError:
            logger.warning(f"❌ NOT FOUND: {id} (valid format but not in DB)")
            raise HTTPException(
                status_code=404,
                detail="Invalid artifact ID format"
            )
        
        package = crud.get_package_by_id(db, package_id)
        if not package:
            logger.warning(f"❌ Package {package_id} not found")
            raise HTTPException(
                status_code=404,
                detail=f"Artifact does not exist: {id}"
            )
        
        pkg_type = getattr(package, 'artifact_type', 'model')
        logger.info(f"   ✓ Found package: name={package.name}, "
                    f"type={pkg_type}")
        
        # Check artifact type matches
        pkg_artifact_type = getattr(package, 'artifact_type', 'model')
        if pkg_artifact_type != artifact_type:
            logger.warning(
                f"❌ Type mismatch: expected {artifact_type}, "
                f"got {pkg_artifact_type}")
            raise HTTPException(
                status_code=404,
                detail=f"Artifact {id} is not of type {artifact_type}"
            )
        
        # Calculate cost based on content size (using file_size_bytes
        # as proxy for content size)
        # Cost is measured in KB with minimum of 1.0 KB
        content_size_bytes = getattr(package, 'file_size_bytes', 0)
        logger.info(f"   📊 Content size: {content_size_bytes} bytes")
        
        # Formula: standalone_cost = max(1.0, content_size_bytes / 1024.0)
        # Cost in KB with minimum of 1.0 KB (even for empty content)
        standalone_cost = max(1.0, content_size_bytes / 1024.0)
        logger.info(f"   💵 Calculated standalone_cost: {standalone_cost} KB")
        
        # If dependency=true, double the cost (simple approximation)
        # Formula: total_cost = standalone_cost * 2.0 if dependency
        # else standalone_cost
        total_cost = standalone_cost * 2.0 if dependency else standalone_cost
        
        # Always return both standalone_cost and total_cost fields
        # Round to 2 decimal places
        result = {
            id: {
                "standalone_cost": float(round(standalone_cost, 2)),
                "total_cost": float(round(total_cost, 2))
            }
        }
        logger.info(f"   📦 Response: {result}")
        
        logger.info(f"✅ COST: {standalone_cost} KB "
                    f"(total: {total_cost} KB) → {result}")
        return result

    
    # Lineage endpoint - Get artifact lineage graph (BASELINE)
    class ArtifactLineageNode(BaseModel):
        """Lineage graph node."""
        artifact_id: str
        name: str
        source: str
        metadata: Optional[dict] = None
    
    class ArtifactLineageEdge(BaseModel):
        """Lineage graph edge."""
        from_node_artifact_id: str
        to_node_artifact_id: str
        relationship: str
    
    class ArtifactLineageGraph(BaseModel):
        """Complete lineage graph."""
        nodes: list[ArtifactLineageNode]
        edges: list[ArtifactLineageEdge]
    
    @app.get(
        "/artifact/model/{id}/lineage",
        response_model=ArtifactLineageGraph,
        tags=["artifacts"],
        status_code=200
    )
    def get_artifact_lineage(
        id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
    ):
        """
        Get artifact lineage graph (BASELINE).
        
        Extracts lineage from structured metadata (config.json, model cards, etc.)
        showing relationships between models, datasets, and code.
        
        Args:
            id: Artifact ID
            db: Database session
            current_user: Authenticated user
            
        Returns:
            Lineage graph with nodes and edges
        """
        logger.info(f"🌳 LINEAGE QUERY: id={id}")
        
        # Validate ID format
        validate_id(id)
        
        # Try to convert to int for database lookup
        try:
            package_id = int(id)
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail="Invalid artifact ID format"
            )
        
        package = crud.get_package_by_id(db, package_id)
        if not package:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact does not exist: {id}"
            )
        
        # Extract model info for lineage analysis
        nodes = []
        edges = []
        
        # Add the root node (the artifact itself)
        root_node = ArtifactLineageNode(
            artifact_id=str(package.id),
            name=package.name,
            source="database",
            metadata={
                "type": getattr(package, 'artifact_type', 'model'),
                "uploaded_at": (
                    package.uploaded_at.isoformat() 
                    if package.uploaded_at else None
                )
            }
        )
        nodes.append(root_node)
        
        # Try to extract lineage from HuggingFace metadata
        try:
            # Get the model URL from package
            model_url = package.url
            if model_url and "huggingface.co" in model_url:
                # Parse the URL to get owner/repo
                import re
                match = re.search(
                    r'huggingface\.co/([^/]+)/([^/\s]+)', 
                    model_url
                )
                if match:
                    owner = match.group(1)
                    repo = match.group(2).rstrip('/')
                    
                    # Create ParsedURL for lineage extraction
                    parsed_url = ParsedURL(
                        url=model_url,
                        category=URLCategory.MODEL,
                        name=repo,
                        platform="huggingface",
                        owner=owner,
                        repo=repo,
                    )
                    
                    # Use LineageExtractor to get parent models
                    hf_api = HuggingFaceAPI()
                    extractor = LineageExtractor(hf_api=hf_api)
                    
                    # Fetch config and readme
                    config_data = hf_api.get_model_config(parsed_url)
                    readme_content = hf_api.get_readme_content(parsed_url)
                    
                    # Extract lineage (non-recursive for API response speed)
                    lineage_graph = extractor.extract_lineage(
                        model_url=parsed_url,
                        config_data=config_data,
                        readme_content=readme_content,
                        max_depth=2,
                        recursive=False,  # Don't recurse for API speed
                    )
                    
                    # Convert lineage nodes to API format
                    for node in lineage_graph.nodes:
                        # Skip root node (already added)
                        if node.metadata.get("is_root"):
                            continue
                        
                        api_node = ArtifactLineageNode(
                            artifact_id=node.artifact_id,
                            name=node.name,
                            source=node.source,
                            metadata=node.metadata
                        )
                        nodes.append(api_node)
                    
                    # Convert lineage edges to API format
                    for edge in lineage_graph.edges:
                        # Map the root model ID to database ID
                        to_id = edge.to_node_id
                        model_id = f"{owner}/{repo}"
                        if to_id == model_id:
                            to_id = str(package.id)
                        
                        api_edge = ArtifactLineageEdge(
                            from_node_artifact_id=edge.from_node_id,
                            to_node_artifact_id=to_id,
                            relationship=edge.relationship
                        )
                        edges.append(api_edge)
                    
                    logger.info(
                        f"✅ LINEAGE: Extracted {len(nodes)} nodes, "
                        f"{len(edges)} edges from HuggingFace"
                    )
        except Exception as e:
            logger.warning(f"Could not extract HuggingFace lineage: {e}")
            # Continue with just the root node
        
        logger.info(f"✅ LINEAGE: {len(nodes)} nodes, {len(edges)} edges")
        return ArtifactLineageGraph(nodes=nodes, edges=edges)
    
    # License check endpoint - Check license compatibility (BASELINE)
    class SimpleLicenseCheckRequest(BaseModel):
        """License check request."""
        github_url: str
    
    @app.post(
        "/artifact/model/{id}/license-check",
        tags=["artifacts"],
        status_code=200
    )
    async def check_license_compatibility(
        id: str,
        request: SimpleLicenseCheckRequest,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
    ):
        """
        Check license compatibility (BASELINE).
        
        Assesses whether the model's license is compatible with the
        intended usage (fine-tuning, inference) given the GitHub project's license.
        
        Args:
            id: Artifact ID
            request: GitHub URL to check compatibility with
            db: Database session
            current_user: Authenticated user
            
        Returns:
            Boolean indicating compatibility
        """
        logger.info(f"⚖️  LICENSE CHECK: id={id}, github={request.github_url}")
        
        # Validate ID format
        validate_id(id)
        
        # Try to convert to int for database lookup
        try:
            package_id = int(id)
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail="Invalid artifact ID format"
            )
        
        package = crud.get_package_by_id(db, package_id)
        if not package:
            raise HTTPException(
                status_code=404,
                detail=f"Artifact does not exist: {id}"
            )
        
        # Get model license from scores
        scores = crud.get_package_scores(db, package_id)
        model_has_license = scores and scores.license_score and scores.license_score > 0.5
        
        # Check GitHub project license
        try:
            # Extract owner/repo from URL
            github_url = request.github_url.rstrip('/')
            parts = github_url.split('github.com/')
            if len(parts) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid GitHub URL format"
                )
            
            repo_path = parts[1].rstrip('/')
            
            # Use GitHub API to check license
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://api.github.com/repos/{repo_path}",
                    headers={"Accept": "application/vnd.github.v3+json"},
                    timeout=10.0
                )
                
                if response.status_code == 404:
                    raise HTTPException(
                        status_code=404,
                        detail="GitHub project not found"
                    )
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=502,
                        detail="Could not retrieve GitHub license information"
                    )
                
                repo_data = response.json()
                github_has_license = repo_data.get('license') is not None
            
            # Simple compatibility check: both must have licenses
            is_compatible = model_has_license and github_has_license
            
            logger.info(f"✅ LICENSE COMPATIBLE: {is_compatible}")
            return is_compatible
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"License check error: {e}")
            raise HTTPException(
                status_code=502,
                detail=f"External license information could not be retrieved: {str(e)}"
            )
    
    return app


# Create the app instance for uvicorn
app = create_app()