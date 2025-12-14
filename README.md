# ECE30861 - ML Model Registry

[![CI/CD Pipeline](https://github.com/Ziad-elshafey/ECE30861-Phase2/actions/workflows/cicd.yml/badge.svg)](https://github.com/Ziad-elshafey/ECE30861-Phase2/actions/workflows/cicd.yml)
[![Coverage](https://img.shields.io/badge/coverage-79%25-brightgreen.svg)](https://github.com/Ziad-elshafey/ECE30861-Phase2)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Deployment](https://img.shields.io/badge/deployment-AWS%20App%20Runner-orange.svg)](https://vmqqvhwppq.us-east-1.awsapprunner.com/)

## Team

### Phase 1 (Team 19)
- Sanya Dod
- Spoorthi Koppula
- Romita Pakrasi
- Suhani Mathur

### Phase 2 (Team 20 - Current)
- Ahmed Elbehiry
- Zeyad Elshafey
- Omar Ahmed
- Jacob Walter

## Overview
The system is a comprehensive machine learning model auditing platform that evaluates the quality and trustworthiness of ML models hosted on platforms like Hugging Face through eight distinct metrics. The system accepts input files containing triplets of URLs (code repository, dataset, model) and processes each model asynchronously to compute weighted quality scores across multiple dimensions including ramp-up time (ease of getting started), bus factor (contributor diversity), performance claims (documented benchmarks), license compatibility, deployment size feasibility, dataset-code linkage quality, dataset documentation quality, and code structure quality. Each metric is computed in parallel using a configurable scoring system with weights defined in a YAML configuration file, and the results are outputted as NDJSON format for easy integration with other tools. The platform includes robust error handling, comprehensive logging, environment validation (including GitHub token verification), and a complete test suite with coverage requirements, making it suitable for automated ML model evaluation pipelines in research and production environments.

## Features
- CLI interface with ./run supporting:
    - ./run install – installs dependencies.

    - ./run URL_FILE – evaluates models/datasets/code URLs and outputs NDJSON with required fields.

    - ./run test – runs the test suite.

- Outputs 8 different metrics.
    - bus factor
    - code quality
    - dataset and code quality
    - dataset quality
    - license 
    - performance
    - ramp up
    - size

- Parallel metric calculation for better latency.

## Structure
```bash
├── src/                # main code: metrics, utilities
├── tests/              # unit tests and integration tests
├── requirements.txt    # Python dependencies
├── run                 # executable wrapper / entrypoint
├── sample_urls.txt     # example model URLs for testing
└── README.md           
```

## Installation
Install requirements given.
```bash
pip install -r requirements.txt
```

## Testing
Create virtual environment.
```bash
python3 -m venv venv
```

Open virtual environment (macOS and Linux).
```bash
source venv/bin/activate
```

Run test with sample urls.
```bash
./run test sample_urls.txt
```

## CI/CD

### ✅ Continuous Integration (CI)
Automated testing pipeline runs on every push and pull request:
- **Test Suite**: 218 tests with 79% coverage
- **Type Checking**: MyPy static analysis
- **Fast Feedback**: ~2-3 minute runtime

### 🚀 Continuous Deployment (CD)
Automated deployment to AWS on merge to main:
- **Docker**: Containerized FastAPI application
- **AWS ECR**: Container registry for Docker images
- **AWS App Runner**: Serverless deployment (0.25 vCPU, 0.5GB RAM)
- **Auto-Deploy**: Triggered on push to main branch
- **Live API**: https://vmqqvhwppq.us-east-1.awsapprunner.com/

**Cost**: ~$10/month (~$15 for 1.5 months) ✅ Well under $100 budget

**See**: [Complete CI/CD Guide](docs/CI_CD_GUIDE.md)

## Phase 2 Features

### Database Layer
- **8 Tables**: Users, Permissions, Tokens, Packages, Scores, Lineage, Downloads, Health
- **Full CRUD Operations**: Complete database management
- **SQLAlchemy ORM**: Type-safe database access
- **Migration Support**: Alembic-ready for schema changes

### REST API
- **User Management**: Registration, authentication, JWT tokens
- **Package Operations**: Upload, download, search, delete
- **Rating System**: Submit and retrieve package scores
- **Health Monitoring**: System health and metrics endpoints

### Security
- **JWT Authentication**: Secure token-based auth
- **Password Hashing**: Bcrypt encryption
- **Role-Based Access**: Admin and user permissions
- **Audit Trail**: Download history tracking
- **Sensitive Module History**: Track changes to sensitive modules (what changed, when, and by whom)
- **Malicious Model Detection**: Report and track models suspected to be malicious

#### Security API Endpoints
- `GET /api/v1/security/sensitive-modules/{package_id}/history` - Get history for a sensitive module
- `GET /api/v1/security/sensitive-modules/history` - Get all sensitive module history (admin only)
- `GET /api/v1/security/malicious-models` - Get list of suspected malicious models
- `POST /api/v1/security/malicious-models/report` - Report a potentially malicious model
- `GET /api/v1/security/malicious-models/{package_id}` - Get malicious reports for a package
- `PATCH /api/v1/security/malicious-models/reports/{report_id}/status` - Update report status (admin only)

**See**: [Database Documentation](src/database/README.md)
