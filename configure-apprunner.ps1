#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Configure AWS App Runner with environment variables and IAM permissions
    
.DESCRIPTION
    This script automates App Runner configuration:
    1. Adds required environment variables (SQS_QUEUE_URL, etc.)
    2. Creates IAM inline policy for SQS access
    
.PARAMETER AppRunnerServiceARN
    ARN of the App Runner service (optional - will auto-detect if not provided)
    
.EXAMPLE
    .\configure-apprunner.ps1
    Auto-detect App Runner service and configure it
#>

[CmdletBinding()]
param(
    [string]$AppRunnerServiceARN,
    [switch]$SkipIAM
)

$ErrorActionPreference = "Stop"

# Configuration
$AWS_REGION = "us-east-1"
$SQS_QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/576316822080/hf-ingestion-queue"
$S3_BUCKET_NAME = "ml-registery-artifacts"
$ENABLE_S3_STORAGE = "true"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  App Runner Configuration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check AWS credentials
Write-Host "[1/4] Checking AWS configuration..." -ForegroundColor Cyan
try {
    if ($env:AWS_ACCESS_KEY_ID -and $env:AWS_SECRET_ACCESS_KEY) {
        Write-Host "  ✓ AWS credentials found (environment variables)" -ForegroundColor Green
    } else {
        $awsIdentity = aws sts get-caller-identity 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "AWS not configured"
        }
        Write-Host "  ✓ AWS CLI configured" -ForegroundColor Green
    }
} catch {
    Write-Host "  ✗ AWS credentials not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please set AWS credentials first" -ForegroundColor Yellow
    exit 1
}

# Find App Runner service if not provided
if (-not $AppRunnerServiceARN) {
    Write-Host "[2/4] Finding App Runner service..." -ForegroundColor Cyan
    try {
        $services = aws apprunner list-services --region $AWS_REGION --query "ServiceSummaryList[?ServiceName=='ml-registry-api' || contains(ServiceName, 'ml-registry')].ServiceArn" --output text 2>&1
        
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($services)) {
            throw "No App Runner service found"
        }
        
        # Take first service if multiple found
        $AppRunnerServiceARN = ($services -split "`n")[0].Trim()
        Write-Host "  ✓ Found service: $AppRunnerServiceARN" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Could not find App Runner service" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please provide service ARN manually:" -ForegroundColor Yellow
        Write-Host "  .\configure-apprunner.ps1 -AppRunnerServiceARN 'arn:aws:apprunner:...'" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Or configure manually in AWS Console:" -ForegroundColor Yellow
        Write-Host "  1. Go to App Runner → Your Service → Configuration → Environment variables" -ForegroundColor Gray
        Write-Host "  2. Add these variables:" -ForegroundColor Gray
        Write-Host "     SQS_QUEUE_URL = $SQS_QUEUE_URL" -ForegroundColor Gray
        Write-Host "     ENABLE_S3_STORAGE = $ENABLE_S3_STORAGE" -ForegroundColor Gray
        Write-Host "     S3_BUCKET_NAME = $S3_BUCKET_NAME" -ForegroundColor Gray
        Write-Host "     AWS_REGION = $AWS_REGION" -ForegroundColor Gray
        exit 1
    }
} else {
    Write-Host "[2/4] Using provided App Runner service ARN" -ForegroundColor Cyan
    Write-Host "  $AppRunnerServiceARN" -ForegroundColor Gray
}

# Note: App Runner doesn't support updating env vars via CLI easily
# Provide manual instructions
Write-Host "[3/4] Environment Variables Configuration" -ForegroundColor Cyan
Write-Host "  ⚠ App Runner env vars must be configured via AWS Console" -ForegroundColor Yellow
Write-Host ""
Write-Host "Manual Steps:" -ForegroundColor Yellow
Write-Host "  1. Open AWS Console: https://console.aws.amazon.com/apprunner" -ForegroundColor Gray
Write-Host "  2. Select your service (ml-registry-api or similar)" -ForegroundColor Gray
Write-Host "  3. Go to: Configuration → Environment variables → Edit" -ForegroundColor Gray
Write-Host "  4. Add these variables:" -ForegroundColor Gray
Write-Host ""
Write-Host "     SQS_QUEUE_URL = $SQS_QUEUE_URL" -ForegroundColor White
Write-Host "     ENABLE_S3_STORAGE = $ENABLE_S3_STORAGE" -ForegroundColor White
Write-Host "     S3_BUCKET_NAME = $S3_BUCKET_NAME" -ForegroundColor White
Write-Host "     AWS_REGION = $AWS_REGION" -ForegroundColor White
Write-Host "     DATABASE_URL = postgresql://username:password@your-db-endpoint:5432/mlregistry" -ForegroundColor Yellow
Write-Host ""
Write-Host "  ⚠️  CRITICAL: Set DATABASE_URL to your RDS PostgreSQL connection string!" -ForegroundColor Red
Write-Host "     Format: postgresql://username:password@endpoint:port/database" -ForegroundColor Gray
Write-Host "     If you created RDS using setup-rds-postgres.sh, check /tmp/rds-connection-info.txt" -ForegroundColor Gray
Write-Host ""
Write-Host "  5. Click Save → Deploy" -ForegroundColor Gray
Write-Host "  6. Wait for deployment to complete (~3-5 minutes)" -ForegroundColor Gray
Write-Host ""

# IAM Policy Configuration
if (-not $SkipIAM) {
    Write-Host "[4/4] IAM Policy Configuration" -ForegroundColor Cyan
    Write-Host "  ⚠ IAM policy must be added manually via AWS Console" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Manual Steps:" -ForegroundColor Yellow
    Write-Host "  1. Open AWS Console: https://console.aws.amazon.com/iam" -ForegroundColor Gray
    Write-Host "  2. Go to: Roles → Search for 'AppRunner'" -ForegroundColor Gray
    Write-Host "  3. Find role for your service (contains 'ml-registry' or similar)" -ForegroundColor Gray
    Write-Host "  4. Click: Add permissions → Create inline policy" -ForegroundColor Gray
    Write-Host "  5. Switch to JSON tab and paste this policy:" -ForegroundColor Gray
    Write-Host ""
    
    $policy = @"
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sqs:SendMessage",
                "sqs:GetQueueUrl",
                "sqs:GetQueueAttributes"
            ],
            "Resource": "arn:aws:sqs:us-east-1:576316822080:hf-ingestion-queue"
        }
    ]
}
"@
    
    Write-Host $policy -ForegroundColor White
    Write-Host ""
    Write-Host "  6. Name the policy: 'SQSPublishPolicy'" -ForegroundColor Gray
    Write-Host "  7. Click Create policy" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "[4/4] Skipping IAM configuration (--SkipIAM)" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "  Configuration Instructions Provided" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Summary of Manual Steps Required:" -ForegroundColor Cyan
Write-Host "  ✓ [1/4] AWS credentials verified" -ForegroundColor Green
Write-Host "  ✓ [2/4] App Runner service found" -ForegroundColor Green
Write-Host "  ⚠ [3/4] Add environment variables in AWS Console" -ForegroundColor Yellow
Write-Host "  ⚠ [4/4] Add IAM inline policy in AWS Console" -ForegroundColor Yellow
Write-Host ""
Write-Host "After configuration, test the setup:" -ForegroundColor Cyan
Write-Host "  1. Ingest a model:" -ForegroundColor Gray
Write-Host "     curl -X POST https://your-app.awsapprunner.com/artifact/model \" -ForegroundColor Gray
Write-Host "       -H 'Content-Type: application/json' \" -ForegroundColor Gray
Write-Host "       -d '{\"url\":\"https://huggingface.co/bert-base-uncased\"}'" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Check SQS queue in AWS Console" -ForegroundColor Gray
Write-Host "     https://console.aws.amazon.com/sqs" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Scale worker to process queue" -ForegroundColor Gray
Write-Host "     aws ecs update-service --cluster hf-ingestion-cluster \" -ForegroundColor Gray
Write-Host "       --service hf-ingestion-worker-service --desired-count 1" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. Monitor worker logs" -ForegroundColor Gray
Write-Host "     aws logs tail /ecs/hf-ingestion-worker --follow" -ForegroundColor Gray
Write-Host ""
