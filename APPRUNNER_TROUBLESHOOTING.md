# AppRunner Health Check Troubleshooting

## Current Issue: Health Check Timeout

Your AppRunner deployment is failing because the health check at `/api/v1/system/health` is timing out.

## Root Causes & Solutions

### 1. ✅ **FIXED: Database Blocking Startup**

**Problem:** The app was trying to connect to PostgreSQL during startup, blocking all HTTP requests.

**Solution Applied:**
- Added error handling in startup sequence (wraps DB init in try/catch)
- Created lightweight `/health` endpoint that doesn't require database
- Reduced connection timeouts from 10s → 5s to fail faster

### 2. ⚠️ **CRITICAL: RDS Security Group**

**Problem:** Your RDS database likely isn't allowing connections from App Runner.

**How to Fix:**

#### Step 1: Get AppRunner Security Group ID
```bash
# Get your App Runner service details
aws apprunner describe-service \
  --service-arn <your-service-arn> \
  --region us-east-1 \
  --query 'Service.NetworkConfiguration.EgressConfiguration.VpcConnectorArn'
```

If you're using default App Runner networking (no VPC connector), you need to allow **public access** to RDS:

#### Step 2: Update RDS Security Group
```bash
# Find your RDS security group
aws rds describe-db-instances \
  --db-instance-identifier ml-registry-db \
  --region us-east-1 \
  --query 'DBInstances[0].VpcSecurityGroups[0].VpcSecurityGroupId' \
  --output text

# Allow all inbound PostgreSQL traffic (5432) temporarily for testing
aws ec2 authorize-security-group-ingress \
  --group-id <your-sg-id> \
  --protocol tcp \
  --port 5432 \
  --cidr 0.0.0.0/0 \
  --region us-east-1
```

**⚠️ WARNING:** Opening to `0.0.0.0/0` is for testing only! For production:
- Use a VPC connector in App Runner
- Restrict to specific CIDR ranges
- Use AWS PrivateLink

#### Step 3: Verify RDS is Publicly Accessible
```bash
aws rds describe-db-instances \
  --db-instance-identifier ml-registry-db \
  --region us-east-1 \
  --query 'DBInstances[0].PubliclyAccessible'
```

If it returns `false`, make it publicly accessible:
```bash
aws rds modify-db-instance \
  --db-instance-identifier ml-registry-db \
  --publicly-accessible \
  --region us-east-1 \
  --apply-immediately
```

### 3. **Health Check Path Configuration**

The health check should now work on **either** path:
- `/health` - Simple, no database required ✅ **USE THIS FOR APPRUNNER**
- `/api/v1/system/health` - Full health check with database status

#### Update AppRunner Health Check:
1. Go to AWS Console → App Runner → Your Service
2. Configuration → Health check → Edit
3. Change:
   - **Path:** `/health` (instead of `/api/v1/system/health`)
   - **Interval:** 10 seconds
   - **Timeout:** 5 seconds
   - **Healthy threshold:** 2
   - **Unhealthy threshold:** 3

### 4. **Verify Environment Variables**

Your current config looks good:
```
DATABASE_URL=postgresql://mlregistry_admin:t8U%3Age%21%23G%23%3D_GGMhML-TJN%26soLuVD%26Ru@ml-registry-db.cwxieem041of.us-east-1.rds.amazonaws.com:5432/mlregistry
```

The URL encoding is correct (`%3A` = `:`, `%21` = `!`, etc.)

## Quick Deployment Steps

### Option A: Fix RDS Security Group (Recommended)
1. Allow App Runner to connect to RDS (see Step 2 above)
2. Change health check path to `/health`
3. Redeploy

### Option B: Use SQLite (Quick Test)
Remove `DATABASE_URL` env var temporarily to test if the app starts:
1. Remove DATABASE_URL from App Runner env vars
2. Deploy (will use SQLite - data will be lost on restart)
3. If it works, the issue is definitely RDS connectivity

## Testing After Fix

### Test the new health endpoint locally:
```bash
curl https://your-app.awsapprunner.com/health
```

Expected response:
```json
{
  "status": "ok",
  "timestamp": "2025-12-14T03:56:30.123456",
  "service": "ml-registry-api"
}
```

### Test database health:
```bash
curl https://your-app.awsapprunner.com/api/v1/system/health
```

Expected response (if DB connected):
```json
{
  "status": "ok",
  "timestamp": "2025-12-14T03:56:30.123456",
  "database_status": "healthy",
  "uptime_seconds": 42.5
}
```

## Common Issues

### Issue: "Connection timeout"
- RDS security group not allowing traffic
- RDS not publicly accessible
- Wrong DATABASE_URL

### Issue: "Authentication failed"
- Check URL encoding in DATABASE_URL
- Verify username/password

### Issue: "Database does not exist"
- The database name in RDS doesn't match URL
- Create database: `CREATE DATABASE mlregistry;`

## Monitoring Logs

```bash
# View AppRunner logs
aws logs tail /aws/apprunner/ml-registry-service/application --follow

# Look for these messages:
# ✅ "🚀 STARTING ML REGISTRY API"
# ✅ "✅ Database initialized"
# ❌ "❌ Database initialization failed" (RDS connection issue)
```

## Next Steps

1. **Fix RDS security group** (most likely issue)
2. **Change health check path to `/health`**
3. **Redeploy and monitor logs**
4. If still failing, temporarily remove DATABASE_URL to test with SQLite

Your deployment will succeed once App Runner can reach the RDS database!
