# AWS Bedrock LLM Integration

This document describes how to set up and use AWS Bedrock LLM capabilities in the ML Model Registry.

## Overview

The ML Model Registry now includes integration with AWS Bedrock for AI-powered features:
- **Code Analysis**: Analyze code for reproducibility, quality, and security
- **License Checking**: AI-powered license compatibility verification  
- **README Summarization**: Automatic model card analysis and summarization
- **General Text Generation**: Custom prompts for various use cases

## AWS App Runner Configuration

### Required Environment Variables

Add these environment variables to your AWS App Runner service:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_REGION` | No | `us-east-1` | AWS region for Bedrock |
| `BEDROCK_MODEL_ID` | No | `anthropic.claude-3-haiku-20240307-v1:0` | Bedrock model to use |
| `BEDROCK_ENABLED` | No | `true` | Enable/disable Bedrock LLM |
| `BEDROCK_MAX_TOKENS` | No | `4096` | Max tokens in response |
| `BEDROCK_TEMPERATURE` | No | `0.3` | Sampling temperature |

### Your Current Environment Variables
Based on your setup, you already have:
- `HF_TOKEN` - HuggingFace API token
- `GITHUB_TOKEN` - GitHub API token  
- `GEN_AI_STUDIO_API_KEY` - (can be removed if switching to Bedrock)
- `DATABASE_URL` - Aurora PostgreSQL connection string

### Additional Variables to Add
```
AWS_REGION=us-east-1
BEDROCK_ENABLED=true
```

## IAM Permissions

Your App Runner service needs IAM permissions to access Bedrock. Create an IAM role with this policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": [
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
                "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
            ]
        }
    ]
}
```

### Attach Role to App Runner

1. Go to **AWS App Runner Console**
2. Select your service
3. Go to **Configuration** → **Security**
4. Under **Instance role**, select or create a role with the above policy
5. Save and deploy

## Available Models

| Model | ID | Best For |
|-------|-----|----------|
| Claude 3 Haiku | `anthropic.claude-3-haiku-20240307-v1:0` | Fast, cost-effective (default) |
| Claude 3 Sonnet | `anthropic.claude-3-sonnet-20240229-v1:0` | Balanced performance |
| Claude 3.5 Sonnet | `anthropic.claude-3-5-sonnet-20240620-v1:0` | Best quality |
| Claude 3 Opus | `anthropic.claude-3-opus-20240229-v1:0` | Most capable (expensive) |

## API Endpoints

### Health Check
```bash
GET /api/v1/llm/health
```

Response:
```json
{
    "status": "healthy",
    "model": "anthropic.claude-3-haiku-20240307-v1:0",
    "region": "us-east-1",
    "message": "OK"
}
```

### Generate Text
```bash
POST /api/v1/llm/generate
Content-Type: application/json

{
    "prompt": "Explain machine learning in one sentence",
    "system": "You are a helpful AI assistant",
    "max_tokens": 100,
    "temperature": 0.5
}
```

### Analyze Code
```bash
POST /api/v1/llm/analyze-code
Content-Type: application/json

{
    "code": "from transformers import pipeline\nclassifier = pipeline('sentiment-analysis')\nresult = classifier('I love this!')",
    "analysis_type": "reproducibility"
}
```

Response:
```json
{
    "score": 1.0,
    "issues": [],
    "suggestions": [],
    "details": {
        "score": 1.0,
        "issues": [],
        "suggestions": ["Consider adding error handling"]
    }
}
```

### Check License Compatibility
```bash
POST /api/v1/llm/check-license
Content-Type: application/json

{
    "model_license": "Apache-2.0",
    "project_license": "MIT",
    "usage_type": "inference"
}
```

Response:
```json
{
    "compatible": true,
    "confidence": 0.95,
    "explanation": "Apache-2.0 is compatible with MIT for inference use",
    "restrictions": ["Must include Apache-2.0 license notice"]
}
```

### Summarize README
```bash
POST /api/v1/llm/summarize-readme
Content-Type: application/json

{
    "content": "# My Model\n\nThis is a BERT-based model for sentiment analysis..."
}
```

Response:
```json
{
    "summary": "A BERT-based sentiment analysis model",
    "quality_score": 0.7,
    "has_usage_examples": true,
    "has_performance_metrics": false,
    "has_dataset_info": true,
    "has_limitations": false,
    "missing_sections": ["Performance Metrics", "Limitations"]
}
```

## Python Usage Example

```python
from src.llm import get_bedrock_client

# Get client (singleton)
client = get_bedrock_client()

# Generate text
response = client.generate(
    prompt="What is transfer learning?",
    system="You are an ML expert. Be concise."
)
print(response)

# Analyze code
result = client.analyze_code(
    code="import torch\nmodel = torch.load('model.pt')",
    analysis_type="security"
)
print(f"Security score: {result['score']}")

# Check license
compat = client.check_license_compatibility(
    model_license="GPL-3.0",
    project_license="MIT",
    usage_type="fine-tuning"
)
print(f"Compatible: {compat['compatible']}")
```

## Cost Considerations

AWS Bedrock pricing (as of 2024):

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|------------------------|
| Claude 3 Haiku | $0.00025 | $0.00125 |
| Claude 3 Sonnet | $0.003 | $0.015 |
| Claude 3.5 Sonnet | $0.003 | $0.015 |
| Claude 3 Opus | $0.015 | $0.075 |

**Recommendation**: Use Claude 3 Haiku for most operations to minimize costs.

## Troubleshooting

### "Access denied to Bedrock model"
- Ensure IAM role has `bedrock:InvokeModel` permission
- Verify the model is enabled in your AWS account (Bedrock Console → Model access)

### "Model not ready"
- The model may need to be enabled first in Bedrock Console
- Go to AWS Bedrock Console → Model access → Request access

### "AWS credentials not found"
- App Runner should use IAM role (not env variables)
- For local development, configure AWS CLI or set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

### "Request throttled"
- Reduce request rate
- Consider using a queue for batch operations
- Request quota increase from AWS

## Local Development

For local testing without AWS credentials:

```bash
# Disable Bedrock (endpoints will return 503)
export BEDROCK_ENABLED=false

# Or use AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

## Files Added

```
src/
├── llm/
│   ├── __init__.py          # Module exports
│   └── bedrock_client.py    # AWS Bedrock client
└── api/
    └── routes/
        └── llm.py            # LLM API endpoints
```

## Configuration Updated

- `src/api/config.py` - Added Bedrock configuration settings
- `src/api/routes/__init__.py` - Added llm router export
- `src/api/main.py` - Registered LLM router at `/api/v1/llm`
