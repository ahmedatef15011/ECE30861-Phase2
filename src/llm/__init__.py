"""LLM integration module for AWS Bedrock and other providers."""

from .bedrock_client import BedrockClient, get_bedrock_client

__all__ = ["BedrockClient", "get_bedrock_client"]
