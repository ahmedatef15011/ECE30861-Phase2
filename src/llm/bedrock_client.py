"""AWS Bedrock LLM client for ML Model Registry.

This module provides integration with AWS Bedrock for LLM capabilities.
Supports Claude 3 models (Sonnet, Haiku) for:
- Code analysis and reproducibility scoring
- License compatibility checking
- README quality assessment
- Model card generation

AWS Bedrock Authentication:
    On AWS App Runner, authentication is handled via IAM roles.
    Ensure your App Runner service has the following IAM permissions:
    - bedrock:InvokeModel
    - bedrock:InvokeModelWithResponseStream (optional, for streaming)
    
    For local development, use AWS credentials via:
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - AWS credentials file (~/.aws/credentials)
    - AWS SSO
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


# Default configuration
DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# Available Claude 3 models on Bedrock
AVAILABLE_MODELS = {
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
}


class BedrockError(Exception):
    """Base exception for Bedrock-related errors."""
    pass


class BedrockAuthError(BedrockError):
    """Authentication/authorization error with Bedrock."""
    pass


class BedrockModelError(BedrockError):
    """Model invocation error."""
    pass


class BedrockClient:
    """AWS Bedrock client for LLM operations.
    
    This client wraps AWS Bedrock's runtime API to provide a simple
    interface for invoking Claude models.
    
    Attributes:
        model_id: The Bedrock model identifier to use
        region: AWS region for Bedrock
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-1.0)
    
    Example:
        ```python
        client = BedrockClient()
        response = client.generate(
            prompt="Analyze this code for reproducibility",
            system="You are a code quality expert."
        )
        print(response)
        ```
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        region: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ):
        """Initialize Bedrock client.
        
        Args:
            model_id: Bedrock model ID (default: Claude 3 Haiku)
            region: AWS region (default: us-east-1)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (lower = more deterministic)
        """
        self.region = region or os.getenv("AWS_REGION", DEFAULT_REGION)
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize Bedrock runtime client
        self._client = self._create_client()
        
        logger.info(
            f"Initialized Bedrock client: model={self.model_id}, "
            f"region={self.region}"
        )
    
    def _create_client(self):
        """Create boto3 Bedrock runtime client.
        
        Returns:
            boto3 Bedrock runtime client
            
        Raises:
            BedrockAuthError: If AWS credentials are not available
        """
        try:
            # Configure with retries and timeout
            config = Config(
                region_name=self.region,
                retries={
                    "max_attempts": 3,
                    "mode": "adaptive"
                },
                connect_timeout=10,
                read_timeout=60,
            )
            
            return boto3.client(
                "bedrock-runtime",
                config=config
            )
        except NoCredentialsError as e:
            logger.error(f"AWS credentials not found: {e}")
            raise BedrockAuthError(
                "AWS credentials not configured. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, "
                "or configure IAM role for App Runner."
            ) from e
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate text using Claude model.
        
        Args:
            prompt: User prompt/question
            system: System prompt (instructions for the model)
            max_tokens: Override default max tokens
            temperature: Override default temperature
            stop_sequences: Optional stop sequences
            
        Returns:
            Generated text response
            
        Raises:
            BedrockModelError: If model invocation fails
        """
        try:
            # Build request body for Claude 3 (Messages API)
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature if temperature is not None else self.temperature,
                "messages": messages,
            }
            
            if system:
                body["system"] = system
                
            if stop_sequences:
                body["stop_sequences"] = stop_sequences
            
            logger.debug(f"Invoking Bedrock model: {self.model_id}")
            
            response = self._client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            
            # Parse response
            response_body = json.loads(response["body"].read())
            
            # Extract text from Claude 3 response format
            if "content" in response_body and response_body["content"]:
                return response_body["content"][0]["text"]
            
            return ""
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            
            if error_code in ["AccessDeniedException", "UnauthorizedException"]:
                logger.error(f"Bedrock access denied: {error_msg}")
                raise BedrockAuthError(
                    f"Access denied to Bedrock model {self.model_id}. "
                    "Ensure IAM role has bedrock:InvokeModel permission."
                ) from e
            elif error_code == "ModelNotReadyException":
                logger.error(f"Model not ready: {error_msg}")
                raise BedrockModelError(
                    f"Model {self.model_id} is not ready. Try again later."
                ) from e
            elif error_code == "ThrottlingException":
                logger.warning(f"Bedrock throttled: {error_msg}")
                raise BedrockModelError(
                    "Request throttled by Bedrock. Reduce request rate."
                ) from e
            else:
                logger.error(f"Bedrock error: {error_code} - {error_msg}")
                raise BedrockModelError(f"Bedrock error: {error_msg}") from e
                
        except Exception as e:
            logger.error(f"Unexpected Bedrock error: {e}")
            raise BedrockModelError(f"Failed to invoke model: {e}") from e
    
    def analyze_code(
        self,
        code: str,
        analysis_type: str = "reproducibility"
    ) -> Dict[str, Any]:
        """Analyze code using LLM.
        
        Args:
            code: Code to analyze
            analysis_type: Type of analysis (reproducibility, quality, security)
            
        Returns:
            Analysis results dictionary
        """
        system_prompts = {
            "reproducibility": """You are an expert at evaluating ML model reproducibility.
Analyze the provided code and determine if it would run successfully.
Return a JSON object with:
- score: 0.0, 0.5, or 1.0 (0=broken, 0.5=needs fixes, 1=works)
- issues: list of issues found
- suggestions: list of fixes needed""",
            
            "quality": """You are a code quality expert.
Analyze the provided code for quality issues.
Return a JSON object with:
- score: 0.0 to 1.0
- issues: list of quality issues
- metrics: dict with complexity, readability, maintainability scores""",
            
            "security": """You are a security expert.
Analyze the provided code for security vulnerabilities.
Return a JSON object with:
- score: 0.0 to 1.0 (1=secure)
- vulnerabilities: list of security issues found
- severity: overall severity (low, medium, high, critical)"""
        }
        
        prompt = f"""Analyze this code:

```python
{code}
```

Provide your analysis as a valid JSON object."""

        response = self.generate(
            prompt=prompt,
            system=system_prompts.get(analysis_type, system_prompts["quality"]),
            temperature=0.1  # Low temperature for consistent analysis
        )
        
        # Try to parse JSON response
        try:
            # Find JSON in response (may have markdown formatting)
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]
            
            return json.loads(json_match.strip())
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            return {
                "score": 0.5,
                "raw_response": response,
                "parse_error": True
            }
    
    def check_license_compatibility(
        self,
        model_license: str,
        project_license: str,
        usage_type: str = "inference"
    ) -> Dict[str, Any]:
        """Check if licenses are compatible.
        
        Args:
            model_license: License of the ML model
            project_license: License of the project using the model
            usage_type: How the model will be used (inference, fine-tuning, etc.)
            
        Returns:
            Compatibility analysis results
        """
        prompt = f"""Analyze license compatibility:

Model License: {model_license}
Project License: {project_license}
Intended Usage: {usage_type}

Determine if the model can be legally used in the project.
Return a JSON object with:
- compatible: true/false
- confidence: 0.0 to 1.0
- explanation: brief explanation
- restrictions: list of any restrictions or requirements"""

        system = """You are a software licensing expert. Analyze license compatibility
for machine learning models and provide accurate legal guidance.
Always err on the side of caution when unsure."""

        response = self.generate(prompt=prompt, system=system, temperature=0.1)
        
        try:
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]
            else:
                json_match = response
            return json.loads(json_match.strip())
        except json.JSONDecodeError:
            return {
                "compatible": False,
                "confidence": 0.0,
                "explanation": "Unable to determine compatibility",
                "raw_response": response
            }
    
    def summarize_readme(self, readme_content: str) -> Dict[str, Any]:
        """Summarize and analyze a model README/model card.
        
        Args:
            readme_content: Full README content
            
        Returns:
            Summary and quality analysis
        """
        # Truncate if too long
        max_chars = 8000
        content = readme_content[:max_chars]
        if len(readme_content) > max_chars:
            content += "\n...[truncated]"
        
        prompt = f"""Analyze this ML model README/model card:

{content}

Return a JSON object with:
- summary: 2-3 sentence summary of the model
- has_usage_examples: true/false
- has_performance_metrics: true/false
- has_dataset_info: true/false
- has_limitations: true/false
- quality_score: 0.0 to 1.0
- missing_sections: list of recommended sections to add"""

        system = """You are an ML documentation expert. Analyze model cards
and READMEs for completeness and quality."""

        response = self.generate(prompt=prompt, system=system, temperature=0.2)
        
        try:
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]
            else:
                json_match = response
            return json.loads(json_match.strip())
        except json.JSONDecodeError:
            return {
                "summary": "Unable to parse README",
                "quality_score": 0.5,
                "raw_response": response
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check if Bedrock is accessible and working.
        
        Returns:
            Health check results
        """
        try:
            response = self.generate(
                prompt="Say 'OK' to confirm you are working.",
                max_tokens=10,
                temperature=0.0
            )
            return {
                "status": "healthy",
                "model": self.model_id,
                "region": self.region,
                "response": response.strip()
            }
        except BedrockAuthError as e:
            return {
                "status": "auth_error",
                "error": str(e),
                "model": self.model_id
            }
        except BedrockModelError as e:
            return {
                "status": "model_error",
                "error": str(e),
                "model": self.model_id
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "model": self.model_id
            }


# Singleton instance for reuse
_bedrock_client: Optional[BedrockClient] = None


def get_bedrock_client(
    model_id: Optional[str] = None,
    force_new: bool = False
) -> BedrockClient:
    """Get or create a Bedrock client instance.
    
    Uses a singleton pattern for efficiency, but allows creating
    new instances with different configurations.
    
    Args:
        model_id: Optional model ID override
        force_new: Force creation of new client
        
    Returns:
        BedrockClient instance
    """
    global _bedrock_client
    
    if force_new or _bedrock_client is None or model_id:
        _bedrock_client = BedrockClient(model_id=model_id)
    
    return _bedrock_client


# Convenience function for quick generation
def generate_text(
    prompt: str,
    system: Optional[str] = None,
    model: str = "claude-3-haiku"
) -> str:
    """Quick text generation using Bedrock.
    
    Args:
        prompt: User prompt
        system: Optional system prompt
        model: Model shortname (claude-3-haiku, claude-3-sonnet, etc.)
        
    Returns:
        Generated text
    """
    model_id = AVAILABLE_MODELS.get(model, DEFAULT_MODEL_ID)
    client = get_bedrock_client(model_id=model_id)
    return client.generate(prompt=prompt, system=system)
