# Quick Start Guide: Async S3 Upload Worker

This guide walks you through deploying the async S3 upload worker in **5 minutes**.

## What You Get

✅ **Automatic S3 uploads** - No more App Runner timeouts  
✅ **Cost-optimized** - Worker runs only when needed (~$1/month)  
✅ **Fully automated** - One-command deployment  
✅ **Production-ready** - Error handling, retries, monitoring  

---

## Prerequisites

1. **Docker Desktop** installed and running
2. **AWS credentials** configured (environment variables or AWS CLI)
3. **Terraform infrastructure** already deployed (SQS, ECR, ECS)

---

## Step 1: Deploy Worker (2 minutes)

Run the automated deployment script:

```powershell
.\deploy-worker.ps1
```

This will:
- ✓ Build Docker image from `worker/Dockerfile`
- ✓ Authenticate with AWS ECR
- ✓ Push image to ECR repository
- ✓ Verify deployment

**Optional:** To automatically start the worker after deployment:
```powershell
.\deploy-worker.ps1 -ScaleWorker
```

---

## Step 2: Configure App Runner (2 minutes)

Run the configuration helper:

```powershell
.\configure-apprunner.ps1
```

This will show you:
- ✓ Environment variables to add in AWS Console
- ✓ IAM policy to attach for SQS access

**Then follow the manual steps** (AWS Console doesn't support CLI updates):

### Add Environment Variables
1. Go to [App Runner Console](https://console.aws.amazon.com/apprunner)
2. Select your service → **Configuration** → **Environment variables** → **Edit**
3. Add these variables:
   ```
   SQS_QUEUE_URL = https://sqs.us-east-1.amazonaws.com/576316822080/hf-ingestion-queue
   ENABLE_S3_STORAGE = true
   S3_BUCKET_NAME = ml-registery-artifacts
   AWS_REGION = us-east-1
   ```
4. Click **Save** → **Deploy** (wait 3-5 minutes)

### Add IAM Policy
1. Go to [IAM Console](https://console.aws.amazon.com/iam)
2. Search for **Roles** → Find your App Runner role (contains "ml-registry")
3. **Add permissions** → **Create inline policy** → **JSON tab**
4. Paste this policy:
   ```json
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
   ```
5. Name it **SQSPublishPolicy** → **Create policy**

---

## Step 3: Test End-to-End (1 minute)

### Test 1: Queue a Model for Upload
```bash
curl -X POST https://your-app.awsapprunner.com/artifact/model \
  -H "Content-Type: application/json" \
  -d '{"url":"https://huggingface.co/bert-base-uncased"}'
```

**Expected:** Immediate response with artifact ID and S3 URL (before upload completes)

### Test 2: Verify Message in SQS
1. Go to [SQS Console](https://console.aws.amazon.com/sqs)
2. Select `hf-ingestion-queue`
3. **Send and receive messages** → **Poll for messages**
4. You should see 1 message with your artifact details

### Test 3: Start Worker and Watch Processing
```powershell
# Start worker
aws ecs update-service --cluster hf-ingestion-cluster `
  --service hf-ingestion-worker-service --desired-count 1

# Watch logs in real-time
aws logs tail /ecs/hf-ingestion-worker --follow
```

**Expected logs:**
```
📬 Polling for messages...
📨 Received 1 message(s)
🔨 PROCESSING MESSAGE
   Artifact ID: 123
   Repo: model/bert-base-uncased
📥 Step 1/4: Downloading from HuggingFace...
✅ Downloaded to: /tmp/...
📦 Step 2/4: Creating tarball...
✅ Tarball created: 420.50 MB
☁️  Step 3/4: Uploading to S3...
   Using multipart upload (420.50 MB)
✅ Uploaded successfully: https://...
📝 Step 4/4: Uploading metadata...
✅ PROCESSING COMPLETE
🗑️  Message deleted from queue
```

### Test 4: Verify S3 Upload
```powershell
aws s3 ls s3://ml-registery-artifacts/models/
```

**Expected:** You should see your artifact directory with `*.tar.gz` and `metadata.json`

---

## Step 4: Stop Worker (Save Money!)

After testing, scale worker back to 0 to avoid charges:

```powershell
aws ecs update-service --cluster hf-ingestion-cluster `
  --service hf-ingestion-worker-service --desired-count 0
```

**Worker costs:**
- Idle (desired_count=0): **~$0.75/month** (SQS + S3 only)
- Running 1 hour/month: **~$1.00/month** (includes ECS Fargate Spot)

---

## Common Issues

### Issue: Docker not found
**Solution:** Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and restart PowerShell

### Issue: AWS credentials not found
**Solution:** Set environment variables:
```powershell
$env:AWS_ACCESS_KEY_ID = "your-key"
$env:AWS_SECRET_ACCESS_KEY = "your-secret"
$env:AWS_REGION = "us-east-1"
```

### Issue: ECR authentication failed
**Solution:** Re-run authentication:
```powershell
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 576316822080.dkr.ecr.us-east-1.amazonaws.com
```

### Issue: Worker not processing messages
**Checklist:**
1. ✓ Worker scaled to 1? (`aws ecs describe-services --cluster hf-ingestion-cluster --services hf-ingestion-worker-service`)
2. ✓ SQS_QUEUE_URL set in worker? (Check ECS task definition environment variables)
3. ✓ Worker has S3/SQS permissions? (Check IAM task role)
4. ✓ Messages in queue? (Check SQS console)

### Issue: Database connection errors in worker
**Solution:** Worker can run without DB - it just won't update `ingest_status`. To enable DB updates, add `DATABASE_URL` to ECS task definition environment variables.

---

## Architecture Recap

```
User Request → FastAPI → SQS Queue
                              ↓
                         ECS Worker → HuggingFace → S3
                              ↓
                         Update DB Status
```

**Flow:**
1. User POSTs to `/artifact/model` with HuggingFace URL
2. FastAPI validates model (quality gate)
3. FastAPI queues message to SQS
4. FastAPI returns immediate response (before S3 upload)
5. ECS worker polls SQS (long polling, 20s)
6. Worker downloads from HuggingFace (with git-lfs for large files)
7. Worker creates tarball (compression)
8. Worker uploads to S3 (multipart for large files)
9. Worker updates DB status to "completed"
10. Worker deletes SQS message

**Error handling:**
- Message stays in queue if worker fails
- After 3 retries, message goes to Dead Letter Queue (DLQ)
- CloudWatch logs capture all errors

---

## Monitoring Commands

```powershell
# View recent worker logs
aws logs tail /ecs/hf-ingestion-worker --since 10m

# Follow worker logs in real-time
aws logs tail /ecs/hf-ingestion-worker --follow

# Check worker status
aws ecs describe-services --cluster hf-ingestion-cluster --services hf-ingestion-worker-service

# Check queue depth
aws sqs get-queue-attributes --queue-url https://sqs.us-east-1.amazonaws.com/576316822080/hf-ingestion-queue --attribute-names ApproximateNumberOfMessages

# Check DLQ (failed messages)
aws sqs get-queue-attributes --queue-url https://sqs.us-east-1.amazonaws.com/576316822080/hf-ingestion-dlq --attribute-names ApproximateNumberOfMessages

# List S3 uploads
aws s3 ls s3://ml-registery-artifacts/models/ --recursive
```

---

## Next Steps

- **Production use:** Set up CloudWatch alarms for DLQ messages
- **Auto-scaling:** Add EventBridge rule to auto-scale worker on queue depth
- **Cost optimization:** Use S3 Lifecycle policies to archive old artifacts
- **Monitoring:** Set up CloudWatch dashboard for queue metrics

---

## Need Help?

Check the logs first:
```powershell
aws logs tail /ecs/hf-ingestion-worker --follow
```

Common log patterns:
- `📬 Polling for messages...` = Worker running, waiting for messages
- `📨 Received 1 message(s)` = Processing started
- `✅ PROCESSING COMPLETE` = Success!
- `❌ PROCESSING FAILED` = Check error details

---

**That's it!** Your async S3 upload worker is now deployed and ready to handle HuggingFace model ingestion without App Runner timeouts. 🚀
