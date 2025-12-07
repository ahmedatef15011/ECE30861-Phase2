"""LLM-powered endpoints using AWS Bedrock.

These endpoints provide AI-powered analysis capabilities:
- Code analysis for reproducibility
- License compatibility checking
- README/model card summarization
- General text generation

All endpoints require AWS Bedrock to be properly configured.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
import logging

from src.api.dependencies import get_db, get_optional_user
from src.api.config import settings
from src.database.models import User, Package
from src.database import crud

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request/Response Schemas
# ============================================================================

class GenerateRequest(BaseModel):
    """Request for text generation."""
    prompt: str = Field(..., min_length=1, max_length=10000)
    system: Optional[str] = Field(None, max_length=2000)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    """Response from text generation."""
    response: str
    model: str
    usage: Optional[Dict[str, Any]] = None


class AnalyzeCodeRequest(BaseModel):
    """Request for code analysis."""
    code: str = Field(..., min_length=1, max_length=50000)
    analysis_type: str = Field(
        default="reproducibility",
        description="Type of analysis: reproducibility, quality, security"
    )


class AnalyzeCodeResponse(BaseModel):
    """Response from code analysis."""
    score: float
    issues: List[str] = []
    suggestions: List[str] = []
    details: Optional[Dict[str, Any]] = None


class LicenseCheckRequest(BaseModel):
    """Request for license compatibility check."""
    model_license: str
    project_license: str
    usage_type: str = Field(
        default="inference",
        description="Usage type: inference, fine-tuning, training, commercial"
    )


class LicenseCheckResponse(BaseModel):
    """Response from license check."""
    compatible: bool
    confidence: float
    explanation: str
    restrictions: List[str] = []


class SummarizeRequest(BaseModel):
    """Request to summarize README content."""
    content: str = Field(..., min_length=1, max_length=100000)


class SummarizeResponse(BaseModel):
    """Response from README summarization."""
    summary: str
    quality_score: float
    has_usage_examples: bool
    has_performance_metrics: bool
    has_dataset_info: bool
    has_limitations: bool
    missing_sections: List[str] = []


class LLMHealthResponse(BaseModel):
    """LLM health check response."""
    status: str
    model: str
    region: str
    message: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def get_bedrock_client():
    """Get Bedrock client with lazy loading."""
    if not settings.BEDROCK_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bedrock LLM is not enabled"
        )
    
    try:
        from src.llm import get_bedrock_client as get_client
        return get_client()
    except ImportError as e:
        logger.error(f"Failed to import Bedrock client: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM module not available"
        )
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock client: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initialize LLM: {str(e)}"
        )


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/health", response_model=LLMHealthResponse)
def llm_health_check():
    """
    Check if AWS Bedrock LLM is available and working.
    
    Returns:
        LLM health status
    """
    if not settings.BEDROCK_ENABLED:
        return LLMHealthResponse(
            status="disabled",
            model=settings.BEDROCK_MODEL_ID,
            region=settings.AWS_REGION or "unknown",
            message="Bedrock is disabled via BEDROCK_ENABLED=false"
        )
    
    try:
        client = get_bedrock_client()
        result = client.health_check()
        
        return LLMHealthResponse(
            status=result.get("status", "unknown"),
            model=result.get("model", settings.BEDROCK_MODEL_ID),
            region=result.get("region", settings.AWS_REGION or "unknown"),
            message=result.get("error") or result.get("response")
        )
    except HTTPException:
        raise
    except Exception as e:
        return LLMHealthResponse(
            status="error",
            model=settings.BEDROCK_MODEL_ID,
            region=settings.AWS_REGION or "unknown",
            message=str(e)
        )


@router.post("/generate", response_model=GenerateResponse)
def generate_text(
    request: GenerateRequest,
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Generate text using AWS Bedrock Claude model.
    
    This is a general-purpose text generation endpoint. Use specific
    endpoints (analyze, summarize, etc.) for structured outputs.
    
    Args:
        request: Generation request with prompt
        current_user: Optional authenticated user
        
    Returns:
        Generated text response
    """
    client = get_bedrock_client()
    
    try:
        response = client.generate(
            prompt=request.prompt,
            system=request.system,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return GenerateResponse(
            response=response,
            model=client.model_id
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@router.post("/analyze-code", response_model=AnalyzeCodeResponse)
def analyze_code(
    request: AnalyzeCodeRequest,
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Analyze code for reproducibility, quality, or security.
    
    Supported analysis types:
    - reproducibility: Will the code run without modifications?
    - quality: Code quality metrics (complexity, readability)
    - security: Security vulnerability scan
    
    Args:
        request: Code and analysis type
        current_user: Optional authenticated user
        
    Returns:
        Analysis results with score and issues
    """
    if request.analysis_type not in ["reproducibility", "quality", "security"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid analysis_type. Use: reproducibility, quality, security"
        )
    
    client = get_bedrock_client()
    
    try:
        result = client.analyze_code(
            code=request.code,
            analysis_type=request.analysis_type
        )
        
        return AnalyzeCodeResponse(
            score=result.get("score", 0.5),
            issues=result.get("issues", []),
            suggestions=result.get("suggestions", []),
            details=result
        )
    except Exception as e:
        logger.error(f"Code analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/check-license", response_model=LicenseCheckResponse)
def check_license_compatibility(
    request: LicenseCheckRequest,
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Check if model license is compatible with project license.
    
    Analyzes legal compatibility between model and project licenses
    for the specified usage type.
    
    Args:
        request: License information and usage type
        current_user: Optional authenticated user
        
    Returns:
        Compatibility analysis with explanation
    """
    client = get_bedrock_client()
    
    try:
        result = client.check_license_compatibility(
            model_license=request.model_license,
            project_license=request.project_license,
            usage_type=request.usage_type
        )
        
        return LicenseCheckResponse(
            compatible=result.get("compatible", False),
            confidence=result.get("confidence", 0.0),
            explanation=result.get("explanation", "Unable to determine"),
            restrictions=result.get("restrictions", [])
        )
    except Exception as e:
        logger.error(f"License check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"License check failed: {str(e)}"
        )


@router.post("/summarize-readme", response_model=SummarizeResponse)
def summarize_readme(
    request: SummarizeRequest,
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Summarize and analyze a README/model card.
    
    Extracts key information and evaluates completeness of
    the documentation.
    
    Args:
        request: README content to summarize
        current_user: Optional authenticated user
        
    Returns:
        Summary and quality analysis
    """
    client = get_bedrock_client()
    
    try:
        result = client.summarize_readme(content=request.content)
        
        return SummarizeResponse(
            summary=result.get("summary", ""),
            quality_score=result.get("quality_score", 0.5),
            has_usage_examples=result.get("has_usage_examples", False),
            has_performance_metrics=result.get("has_performance_metrics", False),
            has_dataset_info=result.get("has_dataset_info", False),
            has_limitations=result.get("has_limitations", False),
            missing_sections=result.get("missing_sections", [])
        )
    except Exception as e:
        logger.error(f"README summarization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )


@router.post("/analyze-artifact/{artifact_id}")
def analyze_artifact(
    artifact_id: int,
    analysis_type: str = "reproducibility",
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Analyze an artifact from the registry using LLM.
    
    Fetches the artifact's README and code examples, then
    performs the specified analysis.
    
    Args:
        artifact_id: Package/artifact ID to analyze
        analysis_type: Type of analysis to perform
        db: Database session
        current_user: Optional authenticated user
        
    Returns:
        Analysis results for the artifact
    """
    # Get package from database
    package = crud.get_package_by_id(db, artifact_id)
    if not package:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {artifact_id} not found"
        )
    
    client = get_bedrock_client()
    
    results = {
        "artifact_id": artifact_id,
        "artifact_name": package.name,
        "analyses": {}
    }
    
    # Analyze README if available
    if package.readme_content:
        try:
            readme_analysis = client.summarize_readme(package.readme_content)
            results["analyses"]["readme"] = readme_analysis
        except Exception as e:
            results["analyses"]["readme"] = {"error": str(e)}
    
    return results
