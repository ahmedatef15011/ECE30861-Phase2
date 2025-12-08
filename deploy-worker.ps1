#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Automated deployment script for HuggingFace ingestion worker
    
.DESCRIPTION
    This script automates the entire deployment process:
    1. Builds Docker image for ECS worker
    2. Authenticates with AWS ECR
    3. Pushes image to ECR
    4. Updates ECS service (optional)
    
.PARAMETER ScaleWorker
    If specified, scales ECS service to 1 task after deployment
    
.EXAMPLE
    .\deploy-worker.ps1
    Deploy worker image without scaling
    
.EXAMPLE
    .\deploy-worker.ps1 -ScaleWorker
    Deploy worker image and scale to 1 task
#>

[CmdletBinding()]
param(
    [switch]$ScaleWorker,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

# Configuration from Terraform outputs
$AWS_ACCOUNT_ID = "576316822080"
$AWS_REGION = "us-east-1"
$ECR_REPO_NAME = "hf-ingestion-worker"
$ECR_REPO_URL = "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"
$ECS_CLUSTER = "hf-ingestion-cluster"
$ECS_SERVICE = "hf-ingestion-worker-service"
$IMAGE_TAG = "latest"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  HuggingFace Worker Deployment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  ECR Repository: $ECR_REPO_URL" -ForegroundColor Gray
Write-Host "  ECS Cluster:    $ECS_CLUSTER" -ForegroundColor Gray
Write-Host "  ECS Service:    $ECS_SERVICE" -ForegroundColor Gray
Write-Host "  Image Tag:      $IMAGE_TAG" -ForegroundColor Gray
Write-Host ""

# Check if Docker is available
Write-Host "[1/6] Checking Docker installation..." -ForegroundColor Cyan
try {
    $dockerVersion = docker --version
    Write-Host "  ✓ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Docker Desktop:" -ForegroundColor Yellow
    Write-Host "  https://www.docker.com/products/docker-desktop/" -ForegroundColor Gray
    Write-Host ""
    exit 1
}

# Check if AWS CLI is available (for ECR login)
Write-Host "[2/6] Checking AWS configuration..." -ForegroundColor Cyan
try {
    # Check if AWS credentials are set via environment variables
    if ($env:AWS_ACCESS_KEY_ID -and $env:AWS_SECRET_ACCESS_KEY) {
        Write-Host "  ✓ AWS credentials found (environment variables)" -ForegroundColor Green
    } else {
        # Try AWS CLI
        $awsIdentity = aws sts get-caller-identity 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ AWS CLI configured" -ForegroundColor Green
        } else {
            throw "AWS not configured"
        }
    }
} catch {
    Write-Host "  ✗ AWS credentials not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please set AWS credentials:" -ForegroundColor Yellow
    Write-Host "  Option 1: Environment variables" -ForegroundColor Gray
    Write-Host "    `$env:AWS_ACCESS_KEY_ID='your-key'" -ForegroundColor Gray
    Write-Host "    `$env:AWS_SECRET_ACCESS_KEY='your-secret'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  Option 2: AWS CLI" -ForegroundColor Gray
    Write-Host "    aws configure" -ForegroundColor Gray
    Write-Host ""
    exit 1
}

if (-not $SkipBuild) {
    # Build Docker image
    Write-Host "[3/6] Building Docker image..." -ForegroundColor Cyan
    Write-Host "  Command: docker build -t $ECR_REPO_NAME`:$IMAGE_TAG ." -ForegroundColor Gray
    
    Push-Location worker
    try {
        docker build -t "$ECR_REPO_NAME`:$IMAGE_TAG" .
        if ($LASTEXITCODE -ne 0) {
            throw "Docker build failed"
        }
        Write-Host "  ✓ Image built successfully" -ForegroundColor Green
    } finally {
        Pop-Location
    }
    
    # Tag image for ECR
    Write-Host "[4/6] Tagging image for ECR..." -ForegroundColor Cyan
    docker tag "$ECR_REPO_NAME`:$IMAGE_TAG" "$ECR_REPO_URL`:$IMAGE_TAG"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Tagging failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✓ Image tagged: $ECR_REPO_URL`:$IMAGE_TAG" -ForegroundColor Green
} else {
    Write-Host "[3/6] Skipping Docker build (--SkipBuild)" -ForegroundColor Yellow
    Write-Host "[4/6] Skipping image tagging (--SkipBuild)" -ForegroundColor Yellow
}

# Authenticate with ECR
Write-Host "[5/6] Authenticating with ECR..." -ForegroundColor Cyan
try {
    # Get ECR login password and pipe to docker login
    if ($env:AWS_ACCESS_KEY_ID -and $env:AWS_SECRET_ACCESS_KEY) {
        # Use environment variables
        Write-Host "  Using AWS credentials from environment variables" -ForegroundColor Gray
        $ecrPassword = aws ecr get-login-password --region $AWS_REGION 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to get ECR password: $ecrPassword"
        }
        
        # Docker login
        $ecrPassword | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker login failed"
        }
    } else {
        # Use AWS CLI
        aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "ECR authentication failed"
        }
    }
    Write-Host "  ✓ ECR authentication successful" -ForegroundColor Green
} catch {
    Write-Host "  ✗ ECR authentication failed: $_" -ForegroundColor Red
    exit 1
}

if (-not $SkipBuild) {
    # Push image to ECR
    Write-Host "[6/6] Pushing image to ECR..." -ForegroundColor Cyan
    Write-Host "  This may take several minutes for large images..." -ForegroundColor Gray
    docker push "$ECR_REPO_URL`:$IMAGE_TAG"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ✗ Push failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "  ✓ Image pushed successfully" -ForegroundColor Green
} else {
    Write-Host "[6/6] Skipping image push (--SkipBuild)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Deployment Complete! ✓" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Scale ECS service if requested
if ($ScaleWorker) {
    Write-Host "Scaling ECS service to 1 task..." -ForegroundColor Cyan
    try {
        aws ecs update-service `
            --cluster $ECS_CLUSTER `
            --service $ECS_SERVICE `
            --desired-count 1 `
            --region $AWS_REGION | Out-Null
        
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to scale ECS service"
        }
        
        Write-Host "  ✓ Worker scaled to 1 task" -ForegroundColor Green
        Write-Host ""
        Write-Host "Monitor logs:" -ForegroundColor Yellow
        Write-Host "  aws logs tail /ecs/hf-ingestion-worker --follow" -ForegroundColor Gray
    } catch {
        Write-Host "  ✗ Failed to scale worker: $_" -ForegroundColor Red
    }
} else {
    Write-Host "Worker Status: STOPPED (desired_count=0)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To start worker:" -ForegroundColor Yellow
    Write-Host "  aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE --desired-count 1" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To monitor logs:" -ForegroundColor Yellow
    Write-Host "  aws logs tail /ecs/hf-ingestion-worker --follow" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Configure App Runner environment variables (see configure-apprunner.ps1)" -ForegroundColor Gray
Write-Host "  2. Test ingestion: POST to /artifact/model with HuggingFace URL" -ForegroundColor Gray
Write-Host "  3. Check SQS queue for messages in AWS Console" -ForegroundColor Gray
Write-Host "  4. Scale worker to 1 to process queue" -ForegroundColor Gray
Write-Host ""
