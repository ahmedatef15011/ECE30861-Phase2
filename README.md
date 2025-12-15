# ECE30861 - ML Model Registry System

[![CI/CD Pipeline](https://github.com/ahmedatef15011/ECE30861-Phase2/actions/workflows/cicd.yml/badge.svg)](https://github.com/ahmedatef15011/ECE30861-Phase2/actions/workflows/cicd.yml)
[![Coverage](https://img.shields.io/badge/coverage-60%25-brightgreen.svg)](https://github.com/ahmedatef15011/ECE30861-Phase2)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Deployment](https://img.shields.io/badge/deployment-AWS%20App%20Runner-orange.svg)](https://vmqqvhwppq.us-east-1.awsapprunner.com/)

## Team Members

**Phase 2 (Team 20 - Current)**
- Ahmed Elbehiry
- Zeyad Elshafey  
- Omar Ahmed
- Jacob Walter

**Phase 1 (Team 19 - Foundation)**
- Sanya Dod
- Spoorthi Koppula
- Romita Pakrasi
- Suhani Mathur

---

## Overview

The **ML Model Registry System** is a comprehensive machine learning artifact management platform that provides intelligent auditing, quality scoring, and lifecycle management for ML models, datasets, and code repositories.

### Key Capabilities
- **Intelligent Quality Scoring**: 10+ automated metrics
- **LLM-Powered Analysis**: AWS Bedrock integration (Claude 3)
- **Artifact Lineage Tracking**: Parent-child relationship detection
- **Security & Access Control**: JavaScript-based access control
- **Production Deployment**: AWS App Runner + RDS + S3

### Technology Stack
- Backend: Python 3.11, FastAPI, SQLAlchemy
- Database: PostgreSQL (AWS RDS)
- Storage: AWS S3
- LLM: AWS Bedrock (Claude 3 Haiku, Sonnet)
- Deployment: AWS App Runner, Docker
- CI/CD: GitHub Actions
- Testing: pytest (60% coverage)

---

## Quick Start

### Installation

1. Clone and setup:
```bash
git clone https://github.com/ahmedatef15011/ECE30861-Phase2.git
cd ECE30861-Phase2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. Initialize database:
```bash
python -m src.database.init_db
```

4. Run application:
```bash
# API Server
uvicorn src.api.main:app --reload --port 8000

# CLI Tool
./run install
./run URL_FILE
./run test
```

---

## Configuration

### Environment Variables

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ml_registry

# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET_NAME=ml-registry-artifacts

# Bedrock LLM
LLM_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048

# Authentication
JWT_SECRET_KEY=your-secret-key
JWT_EXPIRATION_HOURS=24

# GitHub (for CLI)
GITHUB_TOKEN=your_token
LOG_LEVEL=1
```

### Metric Weights

Edit `config/weights.yaml`:

```yaml
weights:
  bus_factor: 0.10
  code_quality: 0.12
  license: 0.12
  # ... other metrics
```

---

## Deployment

### AWS App Runner

**Production URL**: https://vmqqvhwppq.us-east-1.awsapprunner.com

**Architecture**:
```
Client → App Runner (FastAPI) → RDS PostgreSQL
                  ↓
            AWS S3 + Bedrock
```

**Deployment Steps**:

1. Build Docker image:
```bash
docker build -t ml-registry:latest .
```

2. Push to ECR:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ecr-url>
docker tag ml-registry:latest <ecr-url>:latest
docker push <ecr-url>:latest
```

3. Deploy to App Runner:
```bash
aws apprunner update-service --service-arn <arn>
```

4. Verify:
```bash
curl https://vmqqvhwppq.us-east-1.awsapprunner.com/health
```

---

## API Usage

### Authentication

```bash
curl -X PUT "https://vmqqvhwppq.us-east-1.awsapprunner.com/authenticate" \
  -H "Content-Type: application/json" \
  -d '{
    "user": {"name": "ece30861defaultadminuser", "is_admin": true},
    "secret": {"password": "correcthorsebatterystaple123(!__+@**(A'\''\"`;DROP TABLE packages;"}
  }'
```

Returns: `"bearer eyJhbGci..."`

### Core Endpoints

**Upload Artifact**:
```bash
curl -X POST ".../artifact/ingest" \
  -H "X-Authorization: bearer <token>" \
  -d '{"type": "model", "artifact": {"Name": "my-model", "URL": "https://..."}}'
```

**Download Artifact**:
```bash
curl -X GET ".../artifact/model/123" \
  -H "X-Authorization: bearer <token>"
```

**Search**:
```bash
curl -X GET ".../packages?query=bert&type=model" \
  -H "X-Authorization: bearer <token>"
```

**Get Lineage**:
```bash
curl -X GET ".../artifact/model/13/lineage" \
  -H "X-Authorization: bearer <token>"
```

**API Documentation**: https://vmqqvhwppq.us-east-1.awsapprunner.com/docs

---

## Testing

### Run Tests

```bash
# All tests
python -m pytest tests/

# With coverage
python -m pytest --cov=src --cov-report=term-missing tests/

# Specific test
python -m pytest tests/test_lineage_api.py -v
```

### Coverage: 60%

**Test Areas**:
- ✅ API endpoints (authentication, CRUD)
- ✅ All 10 quality metrics
- ✅ Lineage extraction
- ✅ Database operations
- ✅ Security & access control
- ✅ LLM integration (mocked)

### Manual Testing

Comprehensive manual testing performed on:
1. Artifact upload/download (30+ HuggingFace models)
2. Quality scoring (all metrics validated)
3. Lineage tracking (fine-tuning chains)
4. Access control (JavaScript execution)
5. LLM analysis (README/dependencies)
6. End-to-end API testing

---

## LLM Integration

### Overview

**AWS Bedrock Integration** for intelligent analysis:

**Use Cases**:
1. README quality analysis
2. Artifact lineage extraction
3. Dependency mapping

**Models**:
- Claude 3 Haiku (cost-effective, fast)
- Claude 3.5 Sonnet (high accuracy)

### Inference Parameter Tuning

**Factual Extraction** (lineage, dependencies):
```python
{
    "model": "claude-3-haiku",
    "temperature": 0.1,    # Low for accuracy
    "max_tokens": 2048,
    "top_p": 0.9          # Focus on high probability
}
```

**README Analysis** (quality assessment):
```python
{
    "model": "claude-3-haiku",
    "temperature": 0.3,    # Slightly higher for nuance
    "max_tokens": 4096,
    "top_p": 0.95         # Broader sampling
}
```

### Structured Prompts

All prompts include:
- **Role definition**: Expert system identity
- **Task specification**: Clear instructions
- **Output format**: JSON schema
- **Context**: Relevant metadata

**Example**:
```python
system = """You are an expert at analyzing ML model metadata.

Your task: Identify parent models, datasets, and code repos.

Return JSON:
{
  "parent_models": [{"id": "owner/model", "relationship": "type"}],
  "datasets": [...],
  "confidence": "high|medium|low"
}

Relationship types:
- fine_tuned_from, based_on, merged_from
- trained_on, evaluated_on

Be conservative with confidence."""
```

### Safeguards

1. **JSON Schema Validation**: Validates LLM output structure
2. **Fallback to Heuristics**: Uses regex if LLM fails
3. **Transitive Filtering**: Removes indirect ancestors
4. **Self-Reference Detection**: Blocks invalid relationships
5. **Confidence Scoring**: Flags low-confidence results

**Code Locations**:
- `src/llm/analyzer.py`: Dependency analysis
- `src/llm/bedrock_client.py`: Bedrock interface
- `src/api/main.py` (L1390-1470): Lineage with LLM
- `src/metrics/llm_scoring.py`: README analysis

### Development Usage

LLMs used extensively during development:
- GitHub Copilot: Code generation
- Claude: Security review
- ChatGPT: Debugging
- LLMs: Documentation

---

## Features

### Quality Metrics (10)

1. **Bus Factor**: Contributor diversity
2. **Code Quality**: Structure, documentation
3. **Dataset Quality**: Documentation, structure
4. **Dataset-Code Linkage**: Alignment
5. **License Score**: OSI compliance
6. **Performance Claims**: Benchmarks
7. **Ramp-Up Time**: Getting started ease
8. **Size Score**: Deployment feasibility
9. **Reproducibility**: Environment, determinism
10. **Reviewedness**: Code review coverage
11. **Tree Score**: Lineage-based propagation

### Lineage Tracking

- Automatic parent-child detection
- Fine-tuning chain tracking
- Transitive parent filtering
- Tree score propagation

**Example**: ResNet-50 → trained-gender → trained-gender-ONNX

### Security

- JWT authentication
- Role-based permissions
- JavaScript access control scripts
- Malicious artifact reporting
- CVE monitoring

---

## Project Structure

```
ECE30861-Phase2/
├── .github/workflows/cicd.yml    # CI/CD pipeline
├── src/
│   ├── api/                       # FastAPI application
│   │   ├── main.py
│   │   ├── routes/
│   │   └── schemas.py
│   ├── auth/                      # Authentication
│   ├── database/                  # Database layer
│   │   ├── models.py
│   │   ├── crud.py
│   │   └── init_db.py
│   ├── llm/                       # LLM integration
│   │   ├── bedrock_client.py
│   │   └── analyzer.py
│   ├── metrics/                   # Quality metrics
│   │   ├── bus_factor.py
│   │   ├── code_quality.py
│   │   ├── treescore.py
│   │   └── ...
│   ├── cli.py                     # CLI interface
│   ├── lineage.py                 # Lineage extraction
│   ├── scoring.py                 # Scoring orchestrator
│   └── storage_s3.py              # S3 integration
├── tests/                         # Test suite (60% coverage)
├── config/weights.yaml            # Metric weights
├── Dockerfile                     # Container image
├── requirements.txt               # Dependencies
├── pytest.ini                     # Test configuration
├── .env.example                   # Environment template
├── ece461_fall_2025_openapi_spec.yaml  # API spec
└── README.md                      # This file
```

---

## CI/CD Pipeline

### GitHub Actions (`.github/workflows/cicd.yml`)

**CI (Continuous Integration)**:
- Trigger: Push/PR to `main`
- Steps:
  1. Checkout code
  2. Setup Python 3.11
  3. Install dependencies
  4. Lint (flake8, pylint)
  5. Run tests with coverage
  6. Security scan (bandit)

**CD (Continuous Deployment)**:
- Trigger: Successful CI on `main`
- Steps:
  1. Build Docker image
  2. Push to Amazon ECR
  3. Update App Runner service
  4. Health check
  5. Rollback on failure

**Status**: All CI/CD checks passing ✅

---

## Browser Interface

### Web-Accessible Features

1. **Swagger UI**: https://vmqqvhwppq.us-east-1.awsapprunner.com/docs
   - Interactive API testing
   - Try endpoints in browser
   - Built-in authentication

2. **ReDoc**: https://vmqqvhwppq.us-east-1.awsapprunner.com/redoc
   - Searchable API reference
   - Organized documentation

3. **Health Dashboard**: https://vmqqvhwppq.us-east-1.awsapprunner.com/health
   - System status
   - Database connectivity
   - S3 status

**Core Operations via Browser**:
- Upload artifacts
- Search packages
- Download files
- View lineage graphs
- Manage users

---

## Documentation Standards

### File-Level Documentation

All source files include:
- Module-level docstring
- Function/method docstrings
- Type hints
- Parameter/return documentation

**Example** (`src/lineage.py`):
```python
"""
Lineage extraction module for HuggingFace models.

Extracts parent model references from config.json and model cards
to build a lineage graph showing derivation relationships.
"""
```

### Code Style

- **PEP 8** compliance (flake8 enforced)
- **snake_case**: functions/variables
- **PascalCase**: classes
- **Max 100 chars**: line length
- **Google-style**: docstrings

---

## Contributing

### Git Workflow

1. Create feature branch
2. Make changes with descriptive commits
3. Run tests locally
4. Push and create PR
5. CI must pass
6. Require code review

### Quality Checklist

- [ ] Tests pass
- [ ] Coverage > 50%
- [ ] No lint errors
- [ ] Docstrings updated
- [ ] Type hints added
- [ ] Manual testing done

---

## Troubleshooting

**Database connection fails**:
- Check `DATABASE_URL` in `.env`
- Ensure PostgreSQL running

**AWS Bedrock access denied**:
- Verify AWS credentials
- Check IAM permissions

**S3 upload fails**:
- Verify bucket exists
- Check AWS credentials

**Tests timeout**:
- Node.js tests auto-skip if unavailable

---

## Contact

- GitHub Issues: https://github.com/ahmedatef15011/ECE30861-Phase2/issues
- API Docs: https://vmqqvhwppq.us-east-1.awsapprunner.com/docs

---

## License

ECE30861 Course Project - All rights reserved

**Last Updated**: December 14, 2025
